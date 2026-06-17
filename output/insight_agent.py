from __future__ import annotations

import json
import logging
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import Config, get_config
from core.groq_rate_limiter import groq_invoke
from core.state import AgentResult

log = logging.getLogger(__name__)


# ── Per-table metric pre-computation (all numbers from code, not LLM) ────────

def _compute_per_table_metrics(
    universal:        list,
    ent_results:      list,
    wow_results:      list,
    table_row_counts: dict[str, int] | None = None,
) -> dict[str, dict]:
    """Compute ALL numeric insight fields from actual row data stored in AgentResult.

    Three guarantees that make this permanent and schema-independent:

    1. best_col is chosen by the highest full-table mean among anomalous columns.
       This selects the primary financial amount column for any table/schema —
       e.g. po_amount beats po_qty because its mean is larger.

    2. named_entities are read from the all_values dict of the EXACT ROW that
       carries top_amount.  Entity and amount always come from the same row —
       cross-row contamination is impossible regardless of dataset.

    3. deviation_multiple = top_amount / amount_avg is pre-computed here in
       Python so the LLM never divides anything itself.

    Returns a dict keyed by table name.  Each entry contains:
      total_flagged  – count of flagged rows for this table only
      signal_counts  – {human_label: count} from this table's signals only
      top_amount     – max anomalous value of primary_col across flagged rows
      amount_avg     – full-table mean of that exact same column
      deviation      – top_amount / amount_avg, rounded to 1 decimal
      primary_col    – column selected as primary financial metric
      col_stats      – {col: {max_anomalous, mean, deviation}} per anomalous col
      named_entities – string values from the same DB row as top_amount
    """
    all_results = universal + ent_results + wow_results
    by_table: dict[str, list] = {}
    for r in all_results:
        by_table.setdefault(r.get("table", "unknown"), []).append(r)

    metrics: dict[str, dict] = {}

    for table, rows in by_table.items():
        sig_counts: dict[str, int] = {}

        # ── Collect col_means from AgentResult ───────────────────────────────
        col_means: dict[str, float] = {}
        for r in rows:
            cm = r.get("analysis", {}).get("col_means", {})
            if cm:
                col_means.update(cm)

        # ── Find max anomalous value per column + remember which row had it ──
        col_max:     dict[str, float] = {}
        col_top_row: dict[str, dict]  = {}  # col -> full AgentResult of the row with max

        for r in rows:
            analysis  = r.get("analysis", {})
            values    = analysis.get("values", {})
            all_vals  = analysis.get("all_values", {})

            for col in analysis.get("anomalous_columns", []):
                raw = values.get(col) if values.get(col) is not None else all_vals.get(col)
                if raw is None:
                    continue
                try:
                    v = float(raw)
                except (ValueError, TypeError):
                    continue
                if col not in col_max or v > col_max[col]:
                    col_max[col]     = round(v, 2)
                    col_top_row[col] = r   # store full result — gives row_index + all_values

            # ── Signal type counts (count ROWS, not signal instances) ────────
            # Each row is counted once per unique signal type it exhibits.
            # sig_counts[type] = "number of rows with this type" — never exceeds
            # the number of flagged rows regardless of how many columns are flagged.
            # Without this dedup, a row with 5 datetime columns generates 5 "dt"
            # signals, inflating counts far beyond the actual row count.
            row_labels: set[str] = set()

            for sig in analysis.get("all_signals", []):
                prefix, _col, _val = _parse_signal(sig)
                if prefix:
                    row_labels.add(_sig_label(prefix))

            for sig in analysis.get("signals", []):
                prefix = sig.split(":")[0] if ":" in sig else "other"
                row_labels.add(_sig_label(prefix))

            for sig_type in analysis.get("signal_types", []):
                row_labels.add(_sig_label(sig_type))

            for label in row_labels:
                sig_counts[label] = sig_counts.get(label, 0) + 1

        # ── Primary column: highest full-table mean among anomalous columns ──
        # Selects the primary financial amount column dynamically for any schema.
        # e.g. for cheque_clearing: amount(mean=6800) beats userid(mean=100)
        # e.g. for price_variance: po_amount(mean=28062) beats po_qty(mean=22)
        best_col = (
            max(col_max, key=lambda c: col_means.get(c, 0))
            if col_max else None
        )
        top_amount = col_max.get(best_col) if best_col else None
        amount_avg = (
            round(col_means[best_col], 4)
            if best_col and best_col in col_means
            else None
        )
        deviation = (
            round(top_amount / amount_avg, 1)
            if top_amount and amount_avg and amount_avg > 0
            else None
        )

        # ── top_row_index: row_index of the exact row carrying top_amount ────────
        # col_top_row[best_col] is the full AgentResult of that row (set above).
        # This is the authoritative row reference — used to override key_findings
        # in _merge_ground_truth so row + column + anomalous_value always agree.
        top_result    = col_top_row.get(best_col) if best_col else None
        top_row_index = (
            top_result.get("analysis", {}).get("row_index")
            if top_result else None
        )

        # ── named_entities: string context columns from the SAME row as top_amount ─
        #
        # Permanent approach — no column names, no regex patterns, no hardcoding:
        #
        #   1. col_top_row[best_col] is already the full AgentResult of the row
        #      that carries top_amount — direct lookup, no value-matching loop needed.
        #
        #   2. Read all_values from that row (every column value as stored in DB).
        #
        #   3. The pipeline already knows which columns are anomalous for this row
        #      (stored in anomalous_columns).  Also collect cols flagged in signals.
        #      Non-anomalous columns = context columns = the entity carriers.
        #
        #   4. From those non-anomalous columns, keep only genuine string values.
        #      Structural filters only — no hardcoded word lists:
        #        • skip numbers, skip null/empty sentinels
        #        • skip datetime strings (pandas parse attempt)
        #        • skip short uppercase+digit codes ≤5 chars (plant codes, IDs)
        #        • skip reference codes: letters then 4+ digits
        #        • skip compound join keys (ABC-123 pattern)
        #        • skip values with 2+ consecutive spaces (padding/code fields)
        #        • skip sentence-like values: end with ! or ? or end with . + space
        #          (these are alert/status text, not entity names)
        #
        # Fallback: if col_top_row has no entry for best_col, use highest signal_count row.
        source_row = top_result or (
            max(rows, key=lambda r: r.get("analysis", {}).get("signal_count", 0))
            if rows else None
        )
        named_entities: list[str] = []
        if source_row:
            analysis_top    = source_row.get("analysis", {})
            all_vals_top    = analysis_top.get("all_values", {})
            # Collect all columns that are themselves anomalous on this row
            anomalous_cols: set[str] = set(analysis_top.get("anomalous_columns", []))
            for sig in analysis_top.get("all_signals", []):
                pfx, col_name, _ = _parse_signal(sig)
                if pfx in ("num", "dt", "txt") and col_name:
                    anomalous_cols.add(col_name)   # these cols are the flags, not entities

            for col, val in all_vals_top.items():
                if col in anomalous_cols:
                    continue          # skip flagged columns — they are the anomalies
                if val is None:
                    continue
                val_str = str(val).strip()

                # Skip empty / null sentinels (including pandas <NA>)
                if not val_str or val_str.upper() in (
                    "MISSING", "NONE", "NULL", "NAN", "NAT", "NA", "<NA>", ""
                ):
                    continue

                # Skip numeric values
                try:
                    float(val_str)
                    continue
                except (ValueError, TypeError):
                    pass

                # Skip datetime strings — check using pandas (works for any locale/format)
                try:
                    import pandas as pd
                    pd.to_datetime(val_str, errors="raise")
                    continue          # successfully parsed as date → skip
                except Exception:
                    pass              # not a date → keep processing

                # Skip purely uppercase+digits of ≤5 chars
                # covers: plant codes (AJM1), currency (EUR, AED), flags (T, F),
                #         short IDs (V0007, M1006) — all non-human-readable labels
                if re.match(r"^[A-Z\d]{1,5}$", val_str):
                    continue

                # Skip internal reference codes: start with letters then digits
                # e.g. CM2026000065, CHQ5364515, PO001099, R003_Cheque_Clearing
                if re.match(r"^[A-Z]{1,4}\d{4,}", val_str):
                    continue

                # Skip compound join keys e.g. PO001099-83
                if re.match(r"^[A-Z0-9]+-[A-Z0-9]+$", val_str):
                    continue

                # Skip values with 2+ consecutive spaces — padding / code fields e.g. "L   2"
                if re.search(r"\s{2,}", val_str):
                    continue

                # Skip sentence-like alert text — entity names never end with ! or ?
                # or end with a period that follows a space (e.g. "Take Action Immediately.")
                if val_str.endswith("!") or val_str.endswith("?"):
                    continue
                if val_str.endswith(".") and " " in val_str:
                    continue

                named_entities.append(val_str)
        named_entities = sorted(set(named_entities))

        # ── Per-column stats for all anomalous columns ────────────────────────
        col_stats: dict[str, dict] = {}
        for col, mx in col_max.items():
            mean = col_means.get(col)
            col_stats[col] = {
                "max_anomalous": mx,
                "mean":          round(mean, 4) if mean is not None else None,
                "deviation":     round(mx / mean, 2) if mean and mean > 0 else None,
            }

        # Count UNIQUE flagged DB rows — permanent ground-truth hierarchy:
        #   1. table_row_counts from DB (passed in by orchestrator) — most accurate,
        #      direct DB read, unaffected by AgentResult metadata completeness.
        #   2. Deduplicated row_index set — reliable when all agents set row_index.
        #   3. len(rows) fallback — last resort only; may overcount multi-agent results.
        if table_row_counts and table in table_row_counts:
            unique_flagged = table_row_counts[table]
        else:
            unique_row_indices = {
                r.get("analysis", {}).get("row_index")
                for r in rows
                if r.get("analysis", {}).get("row_index") is not None
            }
            unique_flagged = len(unique_row_indices) if unique_row_indices else len(rows)

        metrics[table] = {
            "total_flagged":  unique_flagged,
            "signal_counts":  sig_counts,
            "top_amount":     top_amount,
            "amount_avg":     amount_avg,
            "deviation":      deviation,
            "primary_col":    best_col,
            "top_row_index":  top_row_index,   # row_index of the row carrying top_amount
            "col_stats":      col_stats,
            "named_entities": named_entities,
        }

    return metrics


def _fmt_amount(v) -> str:
    """Format a numeric amount for display — always produces clean output.

    Permanent rules (schema-independent):
      - Float that is a whole number → integer string:  172600.0  → '172600'
      - Float with decimals          → float string:    5.1784    → '5.1784'
      - None / non-numeric           → empty string

    Using this helper everywhere in _merge_ground_truth prevents the
    double-.0 artifact (e.g. '172600.0.0') that occurs when a regex replaces
    only the digit part of an already-formatted float string.
    """
    if v is None:
        return ""
    try:
        fv = float(v)
        return str(int(fv)) if fv.is_integer() else str(fv)
    except (TypeError, ValueError):
        return str(v)


def _replace_placeholders(text: str, m: dict) -> str:
    """Replace [TOKEN] placeholders in LLM-generated text with Python-verified values.

    This is the permanent number-injection layer — the LLM prompt instructs it
    to write tokens instead of actual numbers; Python replaces them here with
    pre-computed values from per_table_metrics (_compute_per_table_metrics).
    No regex guessing, no post-hoc patching — the LLM never writes a number at all.

    Per-table tokens (m = one table's entry from per_table_metrics):
      [FLAGGED]   → total_flagged count for this table
      [TOP_AMT]   → top_amount formatted with _fmt_amount (no trailing .0)
      [DEV]       → deviation multiple rounded to 1 dp
      [AVG_AMT]   → amount_avg formatted with _fmt_amount
      [SIG_COUNT] → total exception count across all signal types for this table
    """
    if not text:
        return text
    text = text.replace("[FLAGGED]",   str(m.get("total_flagged", "")))
    text = text.replace("[TOP_AMT]",   _fmt_amount(m.get("top_amount")))
    text = text.replace("[DEV]",       str(m.get("deviation") or ""))
    text = text.replace("[AVG_AMT]",   _fmt_amount(m.get("amount_avg")))
    text = text.replace("[SIG_COUNT]", str(sum(m.get("signal_counts", {}).values())))
    return text


def _strip_column_names(text: str, col_names: set[str]) -> str:
    """Remove internal DB column names from audit-facing text.

    Permanent approach — no hardcoded column list:
      col_names is built dynamically from per_table_metrics (primary_col +
      col_stats keys) so it works for any schema without configuration.

    Patterns removed:
      'po_amount column'     → ''   (column name + the word 'column')
      'in po_amount'         → ''   (preposition + column name)
      'the po_amount'        → ''   (article + column name)
    Only removes exact matches of known column names — never touches
    entity names or qualitative text.
    """
    for col in col_names:
        if not col or len(col) < 3:   # skip very short names to avoid false matches
            continue
        # Remove "X column" / "X columns"
        text = re.sub(rf'\b{re.escape(col)}\s+columns?\b', '', text, flags=re.IGNORECASE)
        # Remove "in X" / "the X" when X is a column name followed by space+word or punctuation
        text = re.sub(rf'\b(in|the)\s+{re.escape(col)}\b', '', text, flags=re.IGNORECASE)
    # Clean up double spaces left by removals
    text = re.sub(r'  +', ' ', text).strip()
    return text


def _merge_ground_truth(parsed: dict, per_table_metrics: dict) -> dict:
    """Override every numeric field in the LLM response with pre-computed values.

    This is the permanent validation layer — the LLM writes narrative text,
    Python owns every number.  Runs after json.loads() on the LLM output.
    Works for any dataset or schema because it derives values from
    per_table_metrics which was built dynamically from the pipeline signals.
    """
    # ── Lift misplaced keys the LLM may have nested inside insights ──────────
    insights = parsed.get("insights", {})
    if isinstance(insights, dict):
        for key in ("recommendations", "data_quality_score", "overall_narrative",
                    "audit_priority", "wow_highlights"):
            if key in insights and key not in parsed:
                parsed[key] = insights.pop(key)

    # ── key_findings — override row, column, anomalous_value, statistical_context ─
    # Python owns all four fields so row + column + anomalous_value always agree.
    # Description is ALSO rebuilt from Python values to prevent text-number mismatch.
    valid_signal_labels = set(_SIGNAL_LABEL_MAP.values())

    for finding in parsed.get("key_findings", []):
        tbl = finding.get("table")
        m   = per_table_metrics.get(tbl, {})
        if m:
            finding["anomalous_value"]     = m.get("top_amount")
            finding["statistical_context"] = m.get("amount_avg")
            if m.get("primary_col"):
                finding["column"] = m["primary_col"]
            if m.get("top_row_index") is not None:
                finding["row"] = f"row_{m['top_row_index']}"

            # Rebuild description so its numbers always match the overridden fields.
            # Strategy: keep the qualitative control-risk tail the LLM wrote (everything
            # after the first " — "), replace only the numeric opening sentence.
            # Also replace any placeholder tokens the LLM wrote in the tail.
            amt = finding.get("anomalous_value")
            avg = finding.get("statistical_context")
            if amt is not None and avg and avg > 0:
                dev = round(amt / avg, 1)
                old = finding.get("description", "")
                # Replace placeholders in the LLM tail before keeping it
                old = _replace_placeholders(old, m)
                tail_idx = old.find(" — ")
                tail = (" — " + old[tail_idx + 3:]) if tail_idx >= 0 else "."
                finding["description"] = (
                    f"A transaction in the {tbl} register of {_fmt_amount(amt)} represents "
                    f"{dev}x the register average of {_fmt_amount(avg)}{tail}"
                )
            elif finding.get("description"):
                # No rebuild — still replace placeholders in whatever the LLM wrote
                finding["description"] = _replace_placeholders(finding["description"], m)

    # Deduplicate key_findings: keep only the first (Python-canonical) entry per table.
    # After the override loop above, every entry for the same table has identical
    # structured fields and an identical rebuilt description, so later duplicates are
    # pure noise introduced by the LLM writing secondary findings for the same table.
    _seen_kf_tables: set[str] = set()
    _deduped_kf: list[dict] = []
    for finding in parsed.get("key_findings", []):
        tbl = finding.get("table")
        if tbl not in _seen_kf_tables:
            _seen_kf_tables.add(tbl)
            _deduped_kf.append(finding)
    parsed["key_findings"] = _deduped_kf

    # Ensure every table has at least one key_finding entry.
    # If the LLM duplicated one table and skipped others, Python fills in the gaps.
    covered = {f.get("table") for f in parsed.get("key_findings", [])}
    for tbl, m in per_table_metrics.items():
        if tbl in covered or not m.get("top_amount"):
            continue
        top_amt = m["top_amount"]
        avg_amt = m.get("amount_avg")
        dev     = m.get("deviation")
        sig_lbl = next(iter(m.get("signal_counts", {})), "anomalous activity")
        parsed.setdefault("key_findings", []).append({
            "table":               tbl,
            "column":              m.get("primary_col", "unknown"),
            "row":                 f"row_{m['top_row_index']}" if m.get("top_row_index") is not None else "unknown",
            "anomalous_value":     top_amt,
            "statistical_context": avg_amt,
            "description": (
                f"A transaction of {top_amt:,.2f} in {tbl} represents "
                f"{dev}x the register average of {avg_amt:,.4f}, "
                f"indicating {sig_lbl} requiring immediate management review."
            ) if avg_amt else (
                f"Anomalous transactions detected in {tbl} — "
                f"top value {top_amt:,.2f} requires immediate review."
            ),
        })

    # ── risk_signals — fix invalid signal_type, override value + statistical_context ─
    # If the LLM invented a label not in the pipeline's _SIGNAL_LABEL_MAP, replace it
    # with the dominant valid label from the table's own signal_counts.
    # Description is ALSO rebuilt so the count and exposure amount match Python values.
    for sig in parsed.get("risk_signals", []):
        tbl = sig.get("table")
        m   = per_table_metrics.get(tbl, {})
        if not m:
            continue
        sig_type = sig.get("signal_type", "")
        if sig_type not in valid_signal_labels:
            # Replace with the most frequent valid label for this table
            sig["signal_type"] = (
                max(m["signal_counts"].items(), key=lambda x: x[1])[0]
                if m.get("signal_counts") else "unclassified anomaly"
            )
            sig_type = sig["signal_type"]
        sig["value"]               = m["signal_counts"].get(sig_type, sig.get("value"))
        sig["statistical_context"] = m.get("amount_avg") or 0

        # Replace placeholder tokens the LLM wrote in the description FIRST
        if sig.get("description"):
            sig["description"] = _replace_placeholders(sig["description"], m)

        # Rebuild description so the exception count and exposure amount always
        # match the Python-overridden value and top_amount fields.
        # Strategy: keep the qualitative control-risk clause the LLM wrote
        # (the part between "represent" and "exposing"), replace the bookend numbers.
        val     = sig.get("value")
        top_amt = m.get("top_amount")
        if val is not None and tbl and sig_type:
            old = sig.get("description", "")
            represent_idx = old.lower().find(" represent ")
            if represent_idx >= 0:
                # Rebuild the prefix; keep everything after "represent"
                tail = old[represent_idx:]
                if top_amt is not None:
                    # Use _fmt_amount to prevent double-.0 (e.g. 172600.0 → '172600')
                    tail = re.sub(r"exposing\s+[\d,]+(?:\.\d+)?", f"exposing {_fmt_amount(top_amt)}", tail)
                sig["description"] = (
                    f"{val} {sig_type} exceptions across the {tbl} register{tail}"
                )
            elif top_amt is not None:
                sig["description"] = (
                    f"{val} {sig_type} exceptions across the {tbl} register represent "
                    f"a control failure, exposing {_fmt_amount(top_amt)} to financial or regulatory "
                    f"risk if not immediately remediated."
                )

    # ── insights — override ALL numeric fields and ALL number-bearing text ─────
    #
    # Permanent principle: Python owns every number in every text field.
    # The LLM is only trusted for qualitative phrases (control_weakness,
    # the "represent a..." clause in narrative sentences).
    # Every sentence containing a count, amount, or deviation is either
    # rebuilt from Python or regex-corrected so it always matches structured fields.
    #
    # Three text fields fixed per table:
    #   A. condition              — fully rebuilt from Python (no LLM text kept)
    #   B. narrative              — stale counts replaced throughout via regex
    #   C. financial_consequence  — amount corrected to Python top_amount
    for tbl, val in parsed.get("insights", {}).items():
        if not isinstance(val, dict):
            continue
        m = per_table_metrics.get(tbl, {})
        if not m:
            continue

        flagged = m.get("total_flagged", 0)
        top_amt = m.get("top_amount")
        dev     = m.get("deviation")
        dom_sig = next(iter(m.get("signal_counts", {})), None)

        # Structured numeric fields — always Python
        val["total_flagged"]      = flagged   # stored so StoryAgent can read it directly
        val["top_amount"]         = top_amt
        val["deviation_multiple"] = dev
        val["named_entities"]     = m.get("named_entities", [])

        # Replace placeholder tokens FIRST — LLM wrote [FLAGGED], [TOP_AMT] etc.;
        # Python injects the pre-computed values before any regex correction runs.
        # This is the permanent solution: numbers never originate from the LLM.
        for _field in ("condition", "narrative", "financial_consequence"):
            if val.get(_field) and isinstance(val[_field], str):
                val[_field] = _replace_placeholders(val[_field], m)

        # A. condition — fully rebuilt: zero LLM numbers in this field ever
        if flagged and dom_sig:
            val["condition"] = (
                f"{dom_sig.capitalize()} exceptions detected in {flagged} flagged "
                f"transactions across the {tbl} register"
                + (f", with {_fmt_amount(top_amt)} being the top anomalous value." if top_amt else ".")
            )

        # B. narrative — regex-replace every stale count pattern the LLM may write.
        # Patterns targeted (all contain audit count words, not amounts or ratios):
        #   "totaling N"  /  "across N rows"  /  "N flagged transactions"  /
        #   "N anomalies detected"  /  "across N exceptions"
        # top_amount exposures are also corrected.
        # Deviation-ratio phrases ("8.6 times", "25.4x") are NOT touched.
        if val.get("narrative") and flagged:
            n = val["narrative"]
            n = re.sub(r'totaling\s+[\d,]+',              f'totaling {flagged}',             n)
            n = re.sub(r'across\s+[\d,]+\s+rows?',        f'across {flagged} rows',           n)
            n = re.sub(r'[\d,]+\s+flagged\s+transactions?',f'{flagged} flagged transactions', n)
            n = re.sub(r'[\d,]+\s+anomalies\s+detected',  f'{flagged} anomalies detected',   n)
            n = re.sub(r'across\s+[\d,]+\s+exceptions?',  f'across {flagged} exceptions',    n)
            if top_amt is not None:
                n = re.sub(r'exposing\s+[\d,]+(?:\.\d+)?\s+to', f'exposing {_fmt_amount(top_amt)} to', n)
            # Strip internal DB column names from narrative — permanent, schema-independent
            all_col_names: set[str] = set()
            for _m2 in per_table_metrics.values():
                if _m2.get("primary_col"):
                    all_col_names.add(_m2["primary_col"])
                all_col_names.update(_m2.get("col_stats", {}).keys())
            val["narrative"] = _strip_column_names(n, all_col_names)

        # C. financial_consequence — replace amount with Python top_amount (clean format)
        if val.get("financial_consequence") and top_amt is not None:
            val["financial_consequence"] = re.sub(
                r'\b[\d,]+(?:\.\d+)?\b',
                _fmt_amount(top_amt),
                val["financial_consequence"],
                count=1,
            )

    # ── data_quality_score — Python-computed, never delegated to LLM ────────────
    # Formula (schema-independent, calibrated: 100=clean, ~40=typical audit findings):
    #   Each affected table        → -4 pts  (capped at -20 for 5+ tables)
    #   Each distinct signal type  → -5 pts  (capped at -25 for 5+ types)
    #   Fraud signal present       → flat -15 pts (most severe risk type)
    # Why Python: score is derived from DB row counts and signal counts —
    # both are facts the LLM cannot verify; delegating to LLM produces inconsistent
    # and often inflated scores across runs.
    _distinct_sig_types: set[str] = set()
    _tables_affected = 0
    _has_fraud = False
    for _m in per_table_metrics.values():
        _sc = _m.get("signal_counts", {})
        if _sc:
            _tables_affected += 1
            _distinct_sig_types.update(_sc.keys())
            if "fraud structuring indicator" in _sc:
                _has_fraud = True
    parsed["data_quality_score"] = max(0, min(100,
        100
        - min(_tables_affected * 4, 20)
        - min(len(_distinct_sig_types) * 5, 25)
        - (15 if _has_fraud else 0)
    ))

    # ── overall_narrative — fix ALL counts (global total + per-table) ───────────
    # The LLM may write stale totals or per-table exception counts in the narrative.
    # Strategy (permanent, schema-independent):
    #   1. Replace global "N flagged transactions" with sum of per_table_metrics totals.
    #   2. For each table, replace "N exceptions" / "N anomalies" patterns that appear
    #      in the same sentence as the table name with the correct per-table count.
    # Qualitative text (control-risk phrases, team names, etc.) is never touched.
    _total_flagged_true = sum(m.get("total_flagged", 0) for m in per_table_metrics.values())
    # Build set of all DB column names to strip from overall_narrative text
    _all_col_names_nar: set[str] = set()
    for _mn in per_table_metrics.values():
        if _mn.get("primary_col"):
            _all_col_names_nar.add(_mn["primary_col"])
        _all_col_names_nar.update(_mn.get("col_stats", {}).keys())

    if _total_flagged_true and parsed.get("overall_narrative"):
        nar = parsed["overall_narrative"]
        # Replace global placeholder tokens first (LLM wrote [TOTAL], [FLAGGED:table] etc.)
        nar = nar.replace("[TOTAL]", str(_total_flagged_true))
        for _tbl_n, _m_n in per_table_metrics.items():
            nar = nar.replace(f"[FLAGGED:{_tbl_n}]", str(_m_n.get("total_flagged", "")))
            nar = nar.replace(f"[TOP_AMT:{_tbl_n}]", _fmt_amount(_m_n.get("top_amount")))
            nar = nar.replace(f"[DEV:{_tbl_n}]",     str(_m_n.get("deviation") or ""))
            nar = nar.replace(f"[AVG_AMT:{_tbl_n}]", _fmt_amount(_m_n.get("amount_avg")))
        # Replace bare [TOP_AMT]/[DEV]/[AVG_AMT] in overall_narrative using the
        # critical table (highest top_amount) — LLM may use bare tokens here
        if per_table_metrics:
            _crit_m = max(per_table_metrics.values(), key=lambda x: x.get("top_amount") or 0)
            nar = nar.replace("[TOP_AMT]",   _fmt_amount(_crit_m.get("top_amount")))
            nar = nar.replace("[DEV]",       str(_crit_m.get("deviation") or ""))
            nar = nar.replace("[AVG_AMT]",   _fmt_amount(_crit_m.get("amount_avg")))
            nar = nar.replace("[SIG_COUNT]", str(sum(_crit_m.get("signal_counts", {}).values())))
        # Fix global total (fallback regex for any raw numbers the LLM still wrote)
        nar = re.sub(r'\b\d[\d,]*\s+flagged\s+transactions', f'{_total_flagged_true} flagged transactions', nar)
        # Fix per-table exception counts + exposure amounts: sentence by sentence
        sentences = re.split(r'(?<=[.!?])\s+', nar)
        fixed_sentences = []
        for sent in sentences:
            for tbl, m in per_table_metrics.items():
                if tbl in sent:
                    count   = m.get("total_flagged", 0)
                    top_amt = m.get("top_amount")
                    if count:
                        sent = re.sub(r'\b\d[\d,]+\s+(off-hours|statistical|categorical|fraud|missing|seasonal|temporal)',
                                      lambda mo: f'{count} {mo.group(1)}', sent)
                        sent = re.sub(r'\b\d[\d,]+\s+exceptions?', f'{count} exceptions', sent)
                        sent = re.sub(r'\b\d[\d,]+\s+anomalies?',  f'{count} anomalies',  sent)
                    if top_amt is not None:
                        sent = re.sub(r'exposing\s+[\d,]+(?:\.\d+)?\s+to',
                                      f'exposing {_fmt_amount(top_amt)} to', sent)
            fixed_sentences.append(sent)
        nar = " ".join(fixed_sentences)
        # Strip column names from overall_narrative as well
        parsed["overall_narrative"] = _strip_column_names(nar, _all_col_names_nar)

    # ── overall_narrative fallback ─────────────────────────────────────────────
    # When the LLM response is truncated (max tokens hit), overall_narrative
    # will be empty or null.  Build it here from per_table_metrics — never empty.
    # Fully dynamic: uses only pre-computed values, no hardcoded column names.
    if not parsed.get("overall_narrative") and per_table_metrics:
        critical_tbl, critical_m = max(
            per_table_metrics.items(),
            key=lambda x: x[1].get("top_amount") or 0,
        )
        total_flagged_all = sum(m.get("total_flagged", 0) for m in per_table_metrics.values())
        tables_list = list(per_table_metrics.keys())

        top_amt = critical_m.get("top_amount")
        avg_amt = critical_m.get("amount_avg")
        dev     = critical_m.get("deviation")
        sig_labels = ", ".join(critical_m.get("signal_counts", {}).keys()) or "anomalous activity"

        s1 = (
            f"This audit reviewed {len(tables_list)} financial register(s) "
            f"({', '.join(tables_list)}) and identified {total_flagged_all} flagged "
            f"transactions requiring management attention."
        )
        if top_amt and avg_amt and dev:
            s2 = (
                f"The most critical control failure was identified in {critical_tbl}, "
                f"where a transaction of {top_amt:,.2f} represents {dev}x the register "
                f"average of {avg_amt:,.4f}, indicating {sig_labels} that requires "
                f"immediate escalation."
            )
        else:
            s2 = (
                f"The most critical findings were identified in {critical_tbl}, "
                f"where {critical_m.get('total_flagged', 0)} transactions were flagged "
                f"for {sig_labels}."
            )

        other_tbls = [(t, m) for t, m in per_table_metrics.items() if t != critical_tbl]
        if other_tbls:
            parts = []
            for t, m in other_tbls[:3]:
                o_sigs = ", ".join(m.get("signal_counts", {}).keys()) or "anomalies detected"
                parts.append(f"{t} ({m.get('total_flagged', 0)} flagged, {o_sigs})")
            s3 = f"Secondary risk patterns were also detected in {', '.join(parts)}."
        else:
            s3 = (
                f"All flagged transactions are concentrated in {critical_tbl}, "
                f"warranting focused remediation."
            )

        s4 = (
            "Management is required to initiate immediate review of all flagged "
            "transactions, strengthen approval controls, and provide written "
            "remediation responses within the audit response period."
        )

        parsed["overall_narrative"] = " ".join([s1, s2, s3, s4])

    return parsed


# ── Summarisers ───────────────────────────────────────────────────────────────

def _parse_signal(sig: str) -> tuple[str, str, str]:
    """Parse a raw signal string into (prefix, column_name, raw_value_part).

    Signal formats emitted by universal_row_agent:
      num:col=value(mean=X)
      cat:col=value(rare<3%)
      dt:col=HH:MM(off-hours)
      txt:col='text'(kw:word)
    Returns ("", "", "") when the signal cannot be parsed.
    """
    if ":" not in sig or "=" not in sig:
        return "", "", ""
    prefix, rest = sig.split(":", 1)
    col_name, val_part = rest.split("=", 1)
    return prefix.strip(), col_name.strip(), val_part


def _summarise_universal(results: list, max_chars: int = 2_000) -> str:
    """Build a row-by-row data block from actual AgentResult all_values.

    Each row shows EVERY column value locked together as stored in the DB,
    with anomalous columns marked (!) and their deviation ratio pre-computed
    using the accurate col_means from UniversalRowAgent (full-table means).
    No signal string parsing is used for any numeric value.

    Grouped per table.  Rows sorted by signal_count descending so the most
    anomalous transactions appear first.
    """
    if not results:
        return "No universal anomaly analysis available."

    by_table: dict[str, list] = {}
    for r in results:
        by_table.setdefault(r.get("table", "unknown"), []).append(r)

    lines: list[str] = []

    for table, rows in by_table.items():
        # ── col_means: from UniversalRowAgent, computed on the full table ─────
        # All rows in the same table carry the same col_means dict — merge once.
        col_means: dict[str, float] = {}
        for r in rows:
            cm = r.get("analysis", {}).get("col_means", {})
            if cm:
                col_means.update(cm)

        # ── Column hotspots (how often each col is anomalous) ─────────────────
        col_hit: dict[str, int] = {}
        for r in rows:
            for col in r.get("analysis", {}).get("anomalous_columns", []):
                col_hit[col] = col_hit.get(col, 0) + 1
            for sig in r.get("analysis", {}).get("all_signals", []):
                pfx, col_name, _ = _parse_signal(sig)
                if pfx in ("cat", "dt", "txt") and col_name:
                    col_hit[col_name] = col_hit.get(col_name, 0) + 1

        unique_idx = {r.get("analysis", {}).get("row_index") for r in rows if r.get("analysis", {}).get("row_index") is not None}
        unique_count = len(unique_idx) if unique_idx else len(rows)
        lines.append(f"\nTable '{table}' — {unique_count} flagged rows")
        if col_means:
            means_str = ", ".join(f"{c}={v}" for c, v in col_means.items())
            lines.append(f"  Full-table column means: {means_str}")
        if col_hit:
            hotspots = sorted(col_hit.items(), key=lambda x: -x[1])
            lines.append(
                "  Anomaly hotspot columns: "
                + ", ".join(f"{c}({n}x)" for c, n in hotspots)
            )

        lines.append(
            "  Flagged rows — ALL column values per row "
            "(! = anomalous, ratio=value/full-table-mean shown — "
            "only pair values that appear on the SAME row):"
        )

        sorted_rows = sorted(
            rows,
            key=lambda r: r.get("analysis", {}).get("signal_count", 0),
            reverse=True,
        )

        # Show as many rows as the char budget allows (checked after each row)
        current_len = sum(len(l) for l in lines)
        for r in sorted_rows:
            if current_len >= max_chars:
                break

            analysis       = r.get("analysis", {})
            row_idx        = analysis.get("row_index", "?")
            all_values     = analysis.get("all_values", {})
            anomalous_cols = set(analysis.get("anomalous_columns", []))
            signals        = analysis.get("all_signals", [])

            # Include cat / dt / txt anomalous columns from signals
            for sig in signals:
                pfx, col_name, _ = _parse_signal(sig)
                if pfx in ("cat", "dt", "txt") and col_name:
                    anomalous_cols.add(col_name)

            if not all_values:
                line = f"    row_{row_idx}: " + " | ".join(signals)
                lines.append(line)
                current_len += len(line)
                continue

            parts: list[str] = []
            for col, val in all_values.items():
                if col in anomalous_cols:
                    mean = col_means.get(col)
                    if mean and mean > 0:
                        try:
                            ratio = float(val) / mean
                            parts.append(f"{col}={val}(!) [{ratio:.1f}x avg={mean:.4f}]")
                        except (ValueError, TypeError):
                            parts.append(f"{col}={val}(!)")
                    else:
                        parts.append(f"{col}={val}(!)")
                else:
                    parts.append(f"{col}={val}")

            line = f"    row_{row_idx}: " + " | ".join(parts)
            lines.append(line)
            current_len += len(line)

    return "\n".join(lines)[:max_chars]


def _summarise_enterprise_row(results: list, max_chars: int = 1_200) -> str:
    """Summarise enterprise row data-quality / fraud results dynamically.

    Groups by table, counts signal types per table only, and shows every
    flagged row as a complete locked record (all columns together).
    """
    if not results:
        return "No enterprise data quality analysis available."

    by_table: dict[str, list] = {}
    for r in results:
        by_table.setdefault(r.get("table", "unknown"), []).append(r)

    dup_count = sum(1 for r in results if r.get("analysis", {}).get("is_duplicate"))
    unique_global = {(r.get("table"), r.get("analysis", {}).get("row_index")) for r in results if r.get("analysis", {}).get("row_index") is not None}
    lines: list[str] = [f"Enterprise flagged rows: {len(unique_global) or len(results)} | Duplicates: {dup_count}"]

    for table, rows in by_table.items():
        unique_tbl = {r.get("analysis", {}).get("row_index") for r in rows if r.get("analysis", {}).get("row_index") is not None}
        unique_tbl_count = len(unique_tbl) if unique_tbl else len(rows)
        # Per-table signal counts (per-row, not per-signal-instance)
        tbl_sig_counts: dict[str, int] = {}
        for r in rows:
            row_prefixes: set[str] = set()
            for sig in r.get("analysis", {}).get("signals", []):
                prefix = sig.split(":")[0] if ":" in sig else sig[:10]
                row_prefixes.add(prefix)
            for prefix in row_prefixes:
                tbl_sig_counts[prefix] = tbl_sig_counts.get(prefix, 0) + 1

        lines.append(f"\nTable '{table}' — {unique_tbl_count} flagged rows:")
        if tbl_sig_counts:
            lines.append(
                "  Per-table exception counts (USE ONLY these for this table): "
                + ", ".join(f"{k}={v}" for k, v in sorted(tbl_sig_counts.items(), key=lambda x: -x[1]))
            )

        lines.append(
            "  Flagged rows — ALL columns per row "
            "(! = anomalous — only link values on the SAME row):"
        )
        for r in sorted(rows, key=lambda r: r.get("analysis", {}).get("signal_count", 0), reverse=True)[:8]:
            all_values  = r.get("analysis", {}).get("all_values", {})
            signals     = r.get("analysis", {}).get("signals", [])
            null_cols   = set(r.get("analysis", {}).get("null_columns", []))
            row_idx     = r.get("analysis", {}).get("row_index", "?")
            is_dup      = r.get("analysis", {}).get("is_duplicate", False)

            # Identify flagged columns from signals
            sig_flagged: set[str] = set(null_cols)
            for sig in signals:
                if ":" in sig:
                    part = sig.split(":", 1)[1]
                    col_name = part.split("=")[0].split("(")[0]
                    if col_name and col_name != "EXACT_DUPLICATE":
                        sig_flagged.add(col_name)

            if not all_values:
                lines.append(f"    row_{row_idx}: {' | '.join(signals[:6])}")
                continue

            prefix = "DUP! " if is_dup else ""
            parts = [
                f"{col}={val}(!)" if col in sig_flagged else f"{col}={val}"
                for col, val in all_values.items()
            ]
            lines.append(f"    row_{row_idx}: {prefix}" + " | ".join(parts))

    return "\n".join(lines)[:max_chars]


def _summarise_wow_row(results: list, max_chars: int = 1_200) -> str:
    """Summarise WoW row business-pattern results dynamically.

    Groups by table, counts pattern types per table only, and shows every
    flagged row as a complete locked record (all columns together).
    """
    if not results:
        return "No WoW business pattern findings."

    by_table: dict[str, list] = {}
    for r in results:
        by_table.setdefault(r.get("table", "unknown"), []).append(r)

    unique_wow_global = {(r.get("table"), r.get("analysis", {}).get("row_index")) for r in results if r.get("analysis", {}).get("row_index") is not None}
    lines: list[str] = [f"WoW flagged rows: {len(unique_wow_global) or len(results)}"]

    for table, rows in by_table.items():
        unique_wow_tbl = {r.get("analysis", {}).get("row_index") for r in rows if r.get("analysis", {}).get("row_index") is not None}
        unique_wow_count = len(unique_wow_tbl) if unique_wow_tbl else len(rows)
        # Per-table pattern type counts (per-row, not per-signal-instance)
        tbl_type_counts: dict[str, int] = {}
        for r in rows:
            row_types: set[str] = set()
            for sig_type in r.get("analysis", {}).get("signal_types", []):
                row_types.add(sig_type)
            for sig_type in row_types:
                tbl_type_counts[sig_type] = tbl_type_counts.get(sig_type, 0) + 1

        lines.append(f"\nTable '{table}' — {unique_wow_count} WoW flagged rows:")
        if tbl_type_counts:
            lines.append(
                "  Per-table pattern counts (USE ONLY these for this table): "
                + ", ".join(f"{k}={v}" for k, v in sorted(tbl_type_counts.items(), key=lambda x: -x[1]))
            )

        lines.append(
            "  Flagged rows — ALL columns per row "
            "(! = anomalous — only link values on the SAME row):"
        )
        for r in sorted(rows, key=lambda r: r.get("analysis", {}).get("signal_count", 0), reverse=True)[:8]:
            all_values = r.get("analysis", {}).get("all_values", {})
            signals    = r.get("analysis", {}).get("wow_signals", [])
            row_idx    = r.get("analysis", {}).get("row_index", "?")

            # Identify flagged columns from wow signal strings
            sig_flagged: set[str] = set()
            for sig in signals:
                if ":" in sig:
                    rest = sig.split(":", 1)[1]
                    for token in rest.split(","):
                        col_name = token.split("=")[0].split("(")[0].strip()
                        if col_name:
                            sig_flagged.add(col_name)

            if not all_values:
                lines.append(f"    row_{row_idx}: {' | '.join(signals[:6])}")
                continue

            parts = [
                f"{col}={val}(!)" if col in sig_flagged else f"{col}={val}"
                for col, val in all_values.items()
            ]
            lines.append(f"    row_{row_idx}: " + " | ".join(parts))

    return "\n".join(lines)[:max_chars]


# ── Agent ─────────────────────────────────────────────────────────────────────

# Mapping from raw signal prefix codes to human-readable audit terminology.
# Built as a function so callers always get a resolved label regardless of
# the code casing or value produced by different pipeline agents.
_SIGNAL_LABEL_MAP: dict[str, str] = {
    "dt":      "off-hours / temporal anomaly",
    "num":     "statistical value outlier",
    "cat":     "categorical irregularity",
    "txt":     "keyword or pattern anomaly",
    "fraud":   "fraud structuring indicator",
    "null":    "missing data / completeness gap",
    "dup":     "duplicate transaction",
    "other":   "unclassified anomaly",
}


def _sig_label(code: str) -> str:
    """Return the human-readable audit label for a raw signal type code."""
    return _SIGNAL_LABEL_MAP.get(code.lower(), f"{code} anomaly")


def _build_system_prompt(
    tables: list[str],
    agent_names: list[str],
    total_flagged: int,
    signal_breakdown: dict[str, int],
) -> str:
    """Build a fully dynamic audit system prompt derived entirely from pipeline results."""
    table_str = ", ".join(tables) if tables else "unknown financial dataset"
    agent_str = ", ".join(agent_names) if agent_names else "no agents"
    n_tables  = len(tables)

    # Derive dominant risk types dynamically — use human labels throughout
    top_signals = sorted(signal_breakdown.items(), key=lambda x: -x[1])
    dominant_risks = [k for k, _ in top_signals[:3]] if top_signals else []
    risk_focus = (
        "The dominant risk categories in this dataset are: "
        + ", ".join(_sig_label(k) for k in dominant_risks)
        + " — weight your findings accordingly."
        if dominant_risks else ""
    )

    # Regulatory exposure — inferred by the LLM from the signal labels present.
    # No hardcoded signal-to-regulation mapping here: any new signal type the
    # pipeline emits will automatically be included in the legend and the LLM
    # will determine its regulatory implication from context.
    reg_str = (
        "infer applicable regulatory and compliance frameworks from the signal types "
        "listed in the legend above — e.g. off-hours anomalies suggest payment window "
        "controls, statistical outliers suggest financial threshold controls, fraud indicators "
        "suggest AML/CFT obligations — but derive this from the actual signals present, "
        "not from a fixed list"
    )

    # Build a signal legend — code → human label — for use inside the prompt
    sig_legend = "; ".join(
        f'"{k}" means {_sig_label(k)}' for k in signal_breakdown
    ) or "no signals detected"

    # Signal breakdown with human labels for readability
    sig_str = (
        ", ".join(f"{_sig_label(k)} ({v} occurrences)" for k, v in top_signals)
        if top_signals else "no signals detected"
    )

    return (
        "You are a senior financial audit partner producing a board-level audit executive summary.\n\n"
        f"Audit scope: {n_tables} financial table(s) — {table_str}.\n"
        f"Audit agents executed: {agent_str}.\n"
        f"Total flagged transactions: {total_flagged} across all tables.\n"
        f"Risk signal breakdown: {sig_str}.\n"
        f"Signal type legend: {sig_legend}.\n"
        f"Regulatory exposure areas: {reg_str}.\n"
        f"{risk_focus}\n\n"
        "WRITING RULES — apply to every description, insight, narrative, and action field:\n"
        "  1. Write in professional financial audit language a CFO or Audit Committee would understand.\n"
        "  2. NEVER write raw row IDs (e.g. row_14), database column/field names (e.g. po_amount, cheque_date, "
        "inv_amt_sum, cperiod, price_var_pct), or signal codes (dt, num, cat) anywhere in description or "
        "narrative text — use plain business language instead (e.g. 'payment amount' not 'po_amount', "
        "'transaction date' not 'cheque_date', 'invoice amount' not 'inv_amt_sum').\n"
        "  3. NEVER write any actual number in description, narrative, condition, or financial_consequence. "
        "Use ONLY these placeholder tokens — Python will replace them with verified pre-computed values:\n"
        "     [FLAGGED]   = exception/flagged-row count for the current table\n"
        "     [TOP_AMT]   = top anomalous transaction amount for the current table\n"
        "     [DEV]       = deviation multiple (write '[DEV]x', not '8.6x')\n"
        "     [AVG_AMT]   = register average amount for the current table\n"
        "     [SIG_COUNT] = total exception count for the current table\n"
        "     In overall_narrative ONLY: [TOTAL] = global total flagged; [FLAGGED:table_name] = per-table count.\n"
        "     Example: 'A payment of [TOP_AMT] represents [DEV]x the register average of [AVG_AMT], "
        "with [FLAGGED] transactions flagged across this register.'\n"
        "     NEVER write raw digits in any of these fields — always use the token. Do NOT add any currency symbol.\n"
        "  4. Frame every finding as a financial control risk or compliance exposure — not a data observation.\n"
        "  5. Recommendations must name the responsible role/team, state the specific action, "
        "and include a verification step or timeframe.\n"
        "  6. Every token placeholder must come from the list in rule 3 — never invent numbers, percentages, or ratios.\n\n"
        "Produce valid JSON only. Do NOT include markdown fences or any text outside the JSON object."
    )


class InsightAgent:
    # The executive summary JSON is large — use a higher token ceiling than
    # the row agents so the response is never truncated mid-string.
    _MIN_MAX_TOKENS = 3500

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.output_llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=max(self.config.output_llm_max_tokens, self._MIN_MAX_TOKENS),
            max_retries=0,
        )
        # Fallback to 8b when the 70b quota window is exhausted (retry-after > 60s).
        # groq_invoke switches to this automatically — no pipeline stall.
        self._fallback_llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=max(self.config.output_llm_max_tokens, self._MIN_MAX_TOKENS),
            max_retries=0,
        )

    # Conservative token ceiling — keeps every request safely under Groq's
    # per-request / TPM limit regardless of dataset size.
    _TOKEN_HARD_LIMIT = 5_000

    def run(
        self,
        universal_results:      list[AgentResult],
        enterprise_row_results: list[AgentResult] | None = None,
        wow_row_results:        list[AgentResult] | None = None,
        table_row_counts:       dict[str, int] | None = None,
    ) -> dict:
        """
        table_row_counts: optional dict of {table_name: actual_row_count} read
        directly from the DB by the orchestrator.  When provided, these counts
        are used as the authoritative total_flagged ceiling for every table,
        overriding the AgentResult-dedup heuristic.  This is the permanent
        ground truth anchor — no AgentResult metadata is needed for counts.
        """
        universal = universal_results or []
        ent_results = enterprise_row_results or []
        wow_results = wow_row_results or []

        # ── Step 1: compute metadata (needed before building system/schema) ────
        all_results  = universal + ent_results + wow_results
        tables       = sorted({r.get("table", "unknown") for r in all_results if r.get("table")})
        agent_names  = sorted({r.get("agent_name", "unknown") for r in all_results if r.get("agent_name")})
        # Total flagged — permanent ground-truth hierarchy (same as per-table logic):
        #   1. Sum of DB row counts from table_row_counts — exact, schema-independent.
        #   2. Deduplicated (table, row_index) set — reliable when row_index is set.
        #   3. len(all_results) — last resort; may overcount multi-agent results.
        if table_row_counts:
            total_flagged = sum(table_row_counts.values())
        else:
            unique_flagged_global = {
                (r.get("table"), r.get("analysis", {}).get("row_index"))
                for r in all_results
                if r.get("table") and r.get("analysis", {}).get("row_index") is not None
            }
            total_flagged = len(unique_flagged_global) if unique_flagged_global else len(all_results)

        # ── Pre-compute ALL per-table metrics in Python — never delegated to LLM ──
        # This is the single source of truth for all numeric fields in the output.
        # The LLM receives these pre-computed values and is instructed to use them
        # exactly — it only writes narrative text, never recomputes numbers.
        per_table_metrics = _compute_per_table_metrics(universal, ent_results, wow_results, table_row_counts or {})
        per_table_metrics_json = json.dumps(per_table_metrics, indent=2)

        # Aggregate signal breakdown across all agents dynamically (used for
        # system-prompt context only — NOT exposed as per-table counts in user prompt)
        signal_breakdown: dict[str, int] = {}
        for r in universal:
            for sig in r.get("analysis", {}).get("all_signals", []):
                prefix = sig.split(":")[0] if ":" in sig else "other"
                signal_breakdown[prefix] = signal_breakdown.get(prefix, 0) + 1
        for r in ent_results:
            for sig in r.get("analysis", {}).get("signals", []):
                prefix = sig.split(":")[0] if ":" in sig else sig[:10]
                signal_breakdown[prefix] = signal_breakdown.get(prefix, 0) + 1
        for r in wow_results:
            for sig_type in r.get("analysis", {}).get("signal_types", []):
                signal_breakdown[sig_type] = signal_breakdown.get(sig_type, 0) + 1

        # High-risk rows for extra prompt context
        high_risk_rows = sorted(
            [r for r in all_results if "HIGH RISK" in (r.get("insights_text") or "").upper()],
            key=lambda r: r.get("analysis", {}).get("signal_count", 0),
            reverse=True,
        )[:5]
        high_risk_str = "\n".join(
            f"  [{r.get('table')}/{r.get('column')}] {(r.get('insights_text') or '')[:200]}"
            for r in high_risk_rows
        ) or "None detected."

        system = _build_system_prompt(tables, agent_names, total_flagged, signal_breakdown)

        # Build dynamic examples from actual pipeline data — no hardcoded values
        top_row       = high_risk_rows[0] if high_risk_rows else None
        top_sig_type  = next(iter(signal_breakdown), "unknown")
        top_sig_label = _sig_label(top_sig_type)
        top_sig_count = signal_breakdown.get(top_sig_type, 0)
        top_table     = tables[0] if tables else "unknown_table"

        # Derive the dominant signal label for the top table dynamically
        top_table_sigs = [
            sig.split(":")[0] for r in all_results
            if r.get("table") == top_table
            for sig in r.get("analysis", {}).get("all_signals",
                r.get("analysis", {}).get("signals", []))
        ]
        top_table_sig_label = _sig_label(top_table_sigs[0]) if top_table_sigs else top_sig_label
        top_table_count = sum(1 for r in all_results if r.get("table") == top_table)

        # priority_hint — derived from signal composition and high-risk row ratio,
        # not from hardcoded count thresholds, so it scales to any dataset size
        high_risk_ratio = len(high_risk_rows) / max(total_flagged, 1)
        priority_hint = (
            "IMMEDIATE" if signal_breakdown.get("fraud", 0) > 0
            else "HIGH"   if high_risk_ratio >= 0.5 or len(high_risk_rows) >= 3
            else "MEDIUM" if high_risk_rows
            else "LOW"
        )

        # ── Dynamic examples — derived entirely from actual pipeline data ────────
        # These show the LLM the required FORMAT and DEPTH only.
        # All content (risk type, role, action, timeframe) must be inferred by
        # the LLM from the data provided — nothing is hardcoded here.

        kf_table = top_row.get("table", top_table) if top_row else top_table

        kf_ex = (
            f"'A payment of [TOP_AMT] in the {kf_table} register represents [DEV]x the register "
            f"average of [AVG_AMT] — a [fill: domain-specific control weakness derived from "
            f"the {kf_table} context and {top_sig_label} pattern, e.g. approval-limit bypass or "
            f"segregation-of-duties failure] — creating [fill: financial or compliance consequence, "
            f"e.g. unapproved expenditure or regulatory reporting obligation].'"
        )

        rs_ex = (
            f"'[SIG_COUNT] {top_sig_label} exceptions across the {top_table} register "
            f"represent a [fill: domain-specific control gap derived from {top_table} context], "
            f"exposing [TOP_AMT] to [fill: specific risk consequence such as unauthorized payment, "
            f"regulatory breach, or audit qualification] if not immediately remediated.'"
        )

        rec_ex = (
            f"'[Infer from the {top_table} domain and {top_sig_label} risk type which team or role "
            f"is accountable] must [infer the specific control action needed to close the "
            f"{top_sig_label} gap in {top_table}], verify by [infer the appropriate sign-off or "
            f"reconciliation step for this control type], and complete remediation within "
            f"[timeframe appropriate to {priority_hint} audit priority].'"
        )

        ins_ex = (
            f'{{"{top_table}": {{'
            f'"condition": "[FLAGGED] {top_table_sig_label} exceptions detected in [FLAGGED] '
            f'flagged transactions across the {top_table} register, with [TOP_AMT] being the '
            f'top anomalous value.", '
            f'"narrative": "[FLAGGED] transactions in the {top_table} register were flagged for '
            f'{top_table_sig_label}, with the highest-value transaction reaching [TOP_AMT] — '
            f'[DEV]x the register average of [AVG_AMT]. [Describe which named_entities '
            f'(from GROUND TRUTH METRICS) appear on the same rows as the anomalous amounts — '
            f'only pair entity + amount if they appear on the SAME ROW in FLAGGED ROW DATA.] '
            f'This pattern indicates [infer domain-specific control weakness from {top_table} '
            f'context — e.g. three-way match breakdown, approval-limit bypass]. '
            f'If left unaddressed, [state the financial or compliance consequence]. '
            f'[Describe the concentration or spread pattern from the row data.] '
            f'Management should [infer the specific action required] as a {priority_hint} priority.", '
            f'"financial_consequence": "Unaddressed exceptions with a top payment of [TOP_AMT] '
            f'expose the organization to [infer domain-specific financial or regulatory risk]."}}}}'
        )

        # Allowed signal type labels — derived from this run's signal_breakdown only
        allowed_sig_labels = " | ".join(f'"{_sig_label(k)}"' for k in signal_breakdown) or '"unknown"'

        # Build per-table ground-truth string for schema instructions.
        # Each table is on its own block so the LLM cannot cross-contaminate tables.
        # top_amount and amount_avg always refer to the SAME column (primary_col)
        # so deviation_multiple = top_amount / amount_avg is always correct.
        # col_stats lists every anomalous numeric column with its own pre-computed
        # deviation — the LLM reads these directly and never divides anything itself.
        gt_lines = []
        for tbl, m in per_table_metrics.items():
            sig_str_tbl  = ", ".join(f"{lbl}={cnt}" for lbl, cnt in m["signal_counts"].items())
            col_stats_str = " | ".join(
                f"{col}(max_anomalous={cs['max_anomalous']}, "
                f"full_table_mean={cs['mean']}, "
                f"deviation={cs['deviation']}x)"
                for col, cs in m.get("col_stats", {}).items()
            ) or "none"
            deviation = (
                round(m["top_amount"] / m["amount_avg"], 1)
                if m["top_amount"] and m["amount_avg"] and m["amount_avg"] > 0
                else None
            )
            top_row_ref = (
                f"row_{m['top_row_index']}" if m.get("top_row_index") is not None else "unknown"
            )
            gt_lines.append(
                f"  {tbl}:\n"
                f"    total_flagged={m['total_flagged']}\n"
                f"    primary_col={m.get('primary_col', 'none')} "
                f"(top_amount={m['top_amount']}, top_amount_row={top_row_ref}, "
                f"full_table_mean={m['amount_avg']}, deviation={deviation}x)\n"
                f"    all_anomalous_columns: {col_stats_str}\n"
                f"    signal_counts={{{sig_str_tbl}}}\n"
                f"    named_entities={m['named_entities']}"
            )
        gt_block = "\n".join(gt_lines)

        schema_keys: list[str] = []
        schema_keys.append(
            f"key_findings: list of top 5 audit findings — each item is a flat JSON object with "
            f"exactly these scalar fields (NO nested objects, NO arrays for any field): "
            f"table (string — exact name from [{', '.join(tables)}]), "
            f"column (string — use primary_col from GROUND TRUTH METRICS for this table), "
            f"row (string — use top_amount_row from GROUND TRUTH METRICS for this table — "
            f"this is the exact row_N identifier of the row carrying top_amount — "
            f"NEVER use any other row_N for this field), "
            f"anomalous_value (number — use top_amount from GROUND TRUTH METRICS for this table), "
            f"statistical_context (number — use full_table_mean of primary_col from GROUND TRUTH METRICS), "
            f"description (string — EXACTLY ONE sentence for a CFO/Audit Committee audience that: "
            f"(a) states anomalous_value and uses the pre-computed deviation from col_stats — "
            f"do NOT divide, use the deviation value already in GROUND TRUTH METRICS, "
            f"(b) infers the domain-specific control risk from the table name and signal type, "
            f"(c) states the financial or compliance consequence — "
            f"NEVER put column names, row IDs, or signal codes inside description) — "
            f"e.g. {kf_ex}"
        )
        schema_keys.append(
            f"risk_signals: list of top detected risks — each item is a flat JSON object with "
            f"exactly these scalar fields (NO nested objects): "
            f"table (string — exact name from [{', '.join(tables)}]), "
            f"column (string — single affected field name from UNIVERSAL ANOMALY ANALYSIS rows), "
            f"signal_type (string — use ONLY one of these exact labels: "
            f"{allowed_sig_labels} — FORBIDDEN to invent any other label), "
            f"value (number — use the signal_counts value for this signal_type from GROUND TRUTH METRICS for this table), "
            f"statistical_context (number — use amount_avg from GROUND TRUTH METRICS for this table, or 0 if null), "
            f"description (string — EXACTLY ONE sentence using occurrence count and top_amount from "
            f"GROUND TRUTH METRICS for this table — NEVER use global totals — "
            f"NEVER use column names, row IDs, or signal codes) — "
            f"e.g. {rs_ex}"
        )
        schema_keys.append(
            f"insights: JSON object — for each of the {len(tables)} table(s) [{', '.join(tables)}] "
            f"produce a structured object with exactly these fields: "
            f"'condition' (string — one sentence stating anomaly type, the total_flagged count from "
            f"GROUND TRUTH METRICS for THIS table, and the table domain — use signal label words, never codes), "
            f"'named_entities' (array of strings — use named_entities from GROUND TRUTH METRICS for THIS table — "
            f"set to [] if empty — NEVER invent names), "
            f"'top_amount' (number or null — use top_amount from GROUND TRUTH METRICS for THIS table), "
            f"'deviation_multiple' (number or null — use the pre-computed deviation value from "
            f"GROUND TRUTH METRICS primary_col col_stats for THIS table — "
            f"do NOT compute this yourself, read it directly from col_stats deviation field — "
            f"set to null if top_amount is null), "
            f"'control_weakness' (string — one phrase: specific control failure inferred from the table domain), "
            f"'financial_consequence' (string or null — one sentence using top_amount from GROUND TRUTH METRICS — "
            f"set to null if top_amount is null), "
            f"'narrative' (string — 5-7 sentences of board-level analysis: "
            f"sentence 1: anomaly type and total_flagged from GROUND TRUTH METRICS for this table; "
            f"sentence 2: name the specific vendors/entities from named_entities in GROUND TRUTH METRICS "
            f"paired with the row-level amounts from UNIVERSAL ANOMALY ANALYSIS — "
            f"ONLY pair entity + amount if they appear on the SAME ROW in the analysis block; "
            f"sentence 3: specific control weakness inferred from the table domain; "
            f"sentence 4: financial or compliance consequence if left unaddressed; "
            f"sentence 5: concentration or spread pattern from the row data; "
            f"sentence 6: urgency and recommended management action; "
            f"sentence 7: secondary risk indicators — "
            f"NEVER invent any number not present in GROUND TRUTH METRICS) — "
            f"e.g. {ins_ex}"
        )
        schema_keys.append(
            f"recommendations: list of SMART remediation actions — each item is a flat JSON object "
            f"(NO nested objects for column): "
            f"table (string — exact name from [{', '.join(tables)}]), "
            f"column (string — single affected field name), "
            f"responsible_team (string — specific team or role inferred from the table domain), "
            f"action (string — COMPLETE sentence stating: "
            f"(a) the specific control action required, "
            f"(b) the verification or sign-off step, "
            f"(c) a timeframe proportional to {priority_hint} severity — "
            f"NEVER mention column or field names in action text) — e.g. {rec_ex}"
        )
        schema_keys.append(
            "data_quality_score: integer 0-100 — deduct points proportionally using total signal counts "
            "from GROUND TRUTH METRICS across all tables; "
            "100=fully clean, 0=severely compromised"
        )
        schema_keys.append(
            f"overall_narrative: REQUIRED — non-empty string — 3-4 sentence professional audit executive opinion "
            f"written for a board or Audit Committee audience — "
            f"sentence 1: state the audit scope covering the {len(tables)} table(s) [{', '.join(tables)}] "
            f"and the total of {total_flagged} flagged transactions; "
            f"sentence 2: describe the most critical control failure using amounts and counts from GROUND TRUTH METRICS; "
            f"sentence 3: identify secondary risk patterns using GROUND TRUTH METRICS for the remaining tables; "
            f"sentence 4: state the required management action and urgency — "
            f"write in professional financial language: no signal codes, no field names, no row identifiers — "
            f"this field MUST NOT be empty or null"
        )
        if wow_results:
            schema_keys.append(
                "wow_highlights: list of top 3 high-impact business pattern findings — "
                "each must describe in plain business language what the pattern is, why it is "
                "unexpected relative to the dataset baseline, and what financial risk it represents — "
                "do NOT use signal codes or field identifiers — cite actual values and table context"
            )
        else:
            schema_keys.append(
                'wow_highlights: empty list [] — no WoW business pattern data was available for this run'
            )
        schema_keys.append(
            f"audit_priority: one of IMMEDIATE | HIGH | MEDIUM | LOW — "
            f"based on the signals above, the expected priority is '{priority_hint}' "
            f"but set this from the actual severity of findings in the data"
        )

        schema_str = "\n".join(f"- {k}" for k in schema_keys)

        # ── Dynamic token budget for summaries ────────────────────────────────
        # Estimate tokens for all fixed parts (1 token ≈ 4 chars).
        # Subtract from the hard limit so summaries never push the prompt over.
        # The row data block is the largest section — give it 60 % of the budget
        # so the LLM sees as many complete, paired row records as possible.
        _user_frame = (
            f"Audit tables: {', '.join(tables) or 'unknown'}\n"
            f"Total flagged rows: {total_flagged} | High-risk rows: {len(high_risk_rows)}\n\n"
            f"=== GROUND TRUTH METRICS ===\n{gt_block}\n\n"
            f"TOP HIGH-RISK TRANSACTIONS:\n{high_risk_str}\n\n"
            "=== FLAGGED ROW DATA ===\n\n"
            "=== ENTERPRISE DATA QUALITY ===\n\n"
            "=== WOW BUSINESS PATTERNS ===\n\n"
            f"Generate a JSON audit executive summary with exactly these keys:\n{schema_str}\n"
        )
        _CPT = 3.5  # chars per token — matches groq_rate_limiter._CHARS_PER_TOKEN
        fixed_tokens = int(len(system) / _CPT) + int(len(_user_frame) / _CPT) + 50
        summary_token_budget = max(200, self._TOKEN_HARD_LIMIT - fixed_tokens)
        summary_char_budget  = int(summary_token_budget * _CPT)

        # Distribute budget: row data 60 %, enterprise 25 %, wow 15 %
        univ_summary = _summarise_universal(universal,        max_chars=int(summary_char_budget * 0.60))
        ent_summary  = _summarise_enterprise_row(ent_results, max_chars=int(summary_char_budget * 0.25))
        wow_summary  = _summarise_wow_row(wow_results,        max_chars=int(summary_char_budget * 0.15))

        user = f"""\
Audit tables: {', '.join(tables) or 'unknown'}
Total flagged rows: {total_flagged} | High-risk rows: {len(high_risk_rows)}

=== GROUND TRUTH METRICS ===
RULES — apply to every numeric field you write:
  1. top_amount, amount_avg, deviation_multiple — read from primary_col col_stats ONLY.
     DO NOT divide top_amount by amount_avg yourself — use the pre-computed deviation value.
  2. signal_counts — read per-table values ONLY; never use global totals.
  3. named_entities — copy exactly from this block; NEVER invent names.
  4. row identifier in key_findings — MUST match an actual row_N visible in FLAGGED ROW DATA.
  5. entity + amount pairings in narratives — ONLY link values that appear on the SAME row
     in the FLAGGED ROW DATA block below.
{gt_block}

TOP HIGH-RISK TRANSACTIONS:
{high_risk_str}

=== FLAGGED ROW DATA (actual DB values per row — use for row identification, entity-amount pairing, and narrative context) ===
{univ_summary}

=== ENTERPRISE DATA QUALITY (use for narrative context only) ===
{ent_summary}

=== WOW BUSINESS PATTERNS (use for narrative context only) ===
{wow_summary}

Generate a JSON audit executive summary with exactly these keys:
{schema_str}
"""

        try:
            response = groq_invoke(self.llm, [SystemMessage(content=system), HumanMessage(content=user)], agent_name="insight", fallback_llm=self._fallback_llm)
            content = response.content.strip()
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            # Try direct parse first; use json_repair for any LLM-generated
            # malformation (missing commas, unclosed strings, truncation, etc.)
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                from json_repair import repair_json
                parsed = json.loads(repair_json(content))
            # Normalise insights: fix narrative list → string if LLM returned a list.
            if isinstance(parsed.get("insights"), dict):
                for tbl, val in parsed["insights"].items():
                    if not isinstance(val, dict):
                        continue
                    if isinstance(val.get("narrative"), list):
                        val["narrative"] = " ".join(
                            s.strip() for s in val["narrative"]
                            if isinstance(s, str) and s.strip()
                        )
            # Override every numeric field with pre-computed ground-truth values.
            # This is the permanent validation layer — LLM text is kept,
            # but no number from the LLM is ever trusted.
            parsed = _merge_ground_truth(parsed, per_table_metrics)
            return parsed
        except Exception as exc:
            log.error("InsightAgent failed: %s", exc)
            top_risk = high_risk_rows[0].get("insights_text", "") if high_risk_rows else ""
            sig_str  = ", ".join(f"{k}={v}" for k, v in signal_breakdown.items()) or "none"
            fallback_priority = (
                "IMMEDIATE" if signal_breakdown.get("fraud", 0) > 0
                else "HIGH" if len(high_risk_rows) >= 3
                else "MEDIUM" if high_risk_rows
                else "LOW"
            )
            # Build per-table structured insights directly from pre-computed metrics.
            # No re-extraction from raw signals — per_table_metrics is the single source of truth.
            def _fallback_insight(tbl: str) -> dict:
                m          = per_table_metrics.get(tbl, {})
                top_amount = m.get("top_amount")
                avg_amount = m.get("amount_avg")
                sig_labels = sorted(m.get("signal_counts", {}).keys())
                flagged    = m.get("total_flagged", 0)
                deviation  = (
                    round(top_amount / avg_amount, 1)
                    if top_amount and avg_amount and avg_amount > 0
                    else None
                )
                return {
                    "condition": f"{flagged} flagged transactions detected in {tbl}",
                    "named_entities": m.get("named_entities", []),
                    "top_amount": top_amount,
                    "deviation_multiple": deviation,
                    "control_weakness": sig_labels[0] if sig_labels else "review required",
                    "financial_consequence": (
                        f"{top_amount:,.2f} at risk — manual review required" if top_amount else None
                    ),
                    "narrative": (
                        f"{flagged} transactions in {tbl} were flagged for "
                        f"{', '.join(sig_labels) or 'anomalous signals'}. "
                        "LLM synthesis failed — review individual agent results for full details."
                    ),
                }
            table_insights_structured = {
                tbl: _fallback_insight(tbl) for tbl in tables
                if tbl in per_table_metrics
            }
            return {
                "key_findings": [
                    {
                        "table": r.get("table", "unknown"),
                        "column": r.get("column", "unknown"),
                        "row": r.get("row", "unknown"),
                        "anomalous_value": None,
                        "statistical_context": sig_str,
                        "description": (r.get("insights_text") or "")[:200],
                    }
                    for r in high_risk_rows[:5]
                ] or [
                    {
                        "table": tables[0] if tables else "unknown",
                        "column": "multiple",
                        "row": "multiple",
                        "anomalous_value": None,
                        "statistical_context": sig_str,
                        "description": f"{total_flagged} flagged rows detected with signals: {sig_str}. Review individual agent results for row-level details.",
                    }
                ],
                "risk_signals": [
                    {
                        "table": tables[0] if tables else "unknown",
                        "column": "multiple",
                        "signal_type": list(signal_breakdown.keys())[0] if signal_breakdown else "unknown",
                        "value": None,
                        "statistical_context": sig_str,
                        "description": top_risk[:200] if top_risk else f"Pipeline detected signals: {sig_str}. LLM synthesis failed — review agent results.",
                    }
                ],
                "insights": table_insights_structured or {
                    "summary": {"condition": f"Audit pipeline flagged {total_flagged} rows.",
                                "named_entities": [], "top_amount": None,
                                "deviation_multiple": None, "control_weakness": "review required",
                                "financial_consequence": None,
                                "narrative": "Review individual agent results for details."}
                },
                "recommendations": [
                    {
                        "table": tables[0] if tables else "unknown",
                        "column": "multiple",
                        "responsible_team": "Audit team",
                        "action": f"Escalate all HIGH RISK flagged rows in {', '.join(tables)} for immediate manual review.",
                    },
                    {
                        "table": "all",
                        "column": "multiple",
                        "responsible_team": "Finance / IT Audit",
                        "action": f"Investigate dominant signal types ({sig_str}) with the responsible finance or IT audit team.",
                    },
                ],
                "data_quality_score": max(0, 100 - len(high_risk_rows) * 5 - sum(signal_breakdown.values())),
                "overall_narrative": (
                    f"The audit pipeline analyzed {len(tables)} table(s) ({', '.join(tables)}) "
                    f"and flagged {total_flagged} transactions with signal breakdown: {sig_str}. "
                    f"{len(high_risk_rows)} HIGH RISK rows require immediate attention. "
                    "LLM synthesis failed — review individual agent results for full row-level details."
                ),
                "wow_highlights": [
                    f"[{r.get('table')}/{r.get('column')}] {(r.get('insights_text') or '')[:150]}"
                    for r in wow_results[:3]
                ],
                "audit_priority": fallback_priority,
            }
