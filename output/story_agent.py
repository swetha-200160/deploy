from __future__ import annotations

import json
import logging
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import Config, get_config
from core.groq_rate_limiter import groq_invoke

log = logging.getLogger(__name__)


def _compute_story_ground_truth(state: dict) -> dict:
    """Compute all numeric facts from pipeline state — dynamic, schema-independent.

    Works for any dataset: reads column types from column_routes, row counts from
    raw_data, and agent result lists — no hardcoded column names or table names.
    """
    raw_data    = state.get("raw_data") or {}
    routes      = state.get("column_routes", [])
    insights    = state.get("insights", {})

    type_counts: dict[str, int] = {}
    for r in routes:
        t = r.get("detected_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    universal   = state.get("universal_results", []) or state.get("numerical_results", [])
    ent_results = state.get("enterprise_row_results", []) or []
    wow_results = state.get("wow_row_results", []) or []

    # Insights block — contains only real audit tables (tables the agents actually analyzed).
    # This is the most reliable source for table list: it is built from agent results,
    # so it never includes system tables (e.g. foresight_query_sessions) that were
    # loaded from DB but had no anomalies and were not processed by any agent.
    insights_block = insights.get("insights") or {}
    audit_tables = [t for t in insights_block if isinstance(insights_block[t], dict)]

    # Tables — use audit_tables from insights as primary source (excludes system tables).
    # Fallback to raw_data / tables_analyzed only if insights is not yet available.
    if audit_tables:
        tables = audit_tables
    elif raw_data:
        tables = list(raw_data.keys())
    elif state.get("tables_analyzed"):
        tables = state["tables_analyzed"]
    elif state.get("schema_info"):
        tables = list(state["schema_info"].keys())
    else:
        tables = sorted({
            r.get("table") for r in universal + ent_results + wow_results
            if r.get("table")
        })

    total_rows = (
        sum(len(df) for df in raw_data.values()) if raw_data
        else state.get("total_rows_analyzed", 0)
    )

    # total_flagged — InsightAgent now stores total_flagged in each table's insights entry.
    # This is the permanent authoritative source — no fallback guessing needed.
    total_flagged = sum(
        m.get("total_flagged", 0)
        for m in insights_block.values()
        if isinstance(m, dict)
    )
    if not total_flagged:
        # Fallback: dedup from universal results (only if insights not yet available)
        unique_pairs = {
            (r.get("table"), r.get("analysis", {}).get("row_index"))
            for r in universal if r.get("table") and r.get("analysis", {}).get("row_index") is not None
        }
        total_flagged = len(unique_pairs) if unique_pairs else len(universal)

    return {
        "tables":             tables,
        "n_tables":           len(tables),
        "total_rows":         total_rows,
        "type_counts":        type_counts,       # {detected_type: count}
        "n_universal":        len(universal),
        "n_enterprise":       len(ent_results),
        "n_wow":              len(wow_results),
        "total_flagged":      total_flagged,
        "data_quality_score": insights.get("data_quality_score"),
        "overall_narrative":  insights.get("overall_narrative", ""),
    }


def _merge_story_ground_truth(story: dict, gt: dict) -> dict:
    """Override chapter_1 and chapter_2 opening with Python-computed facts.

    Permanent principle: chapters 1 and 2 contain only facts (table names, row
    counts, column type counts, agent row counts) — Python owns all of these.
    The LLM's qualitative wording in the remainder of each chapter is kept.

    Works for any dataset because gt is built dynamically from pipeline state.
    """
    tables      = gt["tables"]
    type_counts = gt["type_counts"]
    n_tables    = gt["n_tables"]
    total_flagged = gt["total_flagged"]

    # Build column type description from whatever types the pipeline detected
    type_parts = [f"{cnt} {typ}" for typ, cnt in sorted(type_counts.items()) if cnt]
    col_desc   = ", ".join(type_parts) if type_parts else "multiple column types"

    # Rebuild chapter_1 opening sentence — fully Python, then append LLM tail
    ch1_prefix = (
        f"The audit covered {n_tables} financial table(s) — "
        f"{', '.join(tables)} — with {col_desc} examined across all tables, "
        f"and a total of {total_flagged} flagged transactions identified."
    )
    old_ch1 = story.get("chapter_1_data_overview", "")
    # Keep qualitative LLM tail (everything after first ". ") if it adds context
    tail_idx = old_ch1.find(". ", len(ch1_prefix) // 2)
    ch1_tail = (" " + old_ch1[tail_idx + 2:].strip()) if tail_idx > 0 else ""
    story["chapter_1_data_overview"] = ch1_prefix + ch1_tail

    # Fix any invented agent row counts in chapter_2
    ch2 = story.get("chapter_2_what_we_checked", "")
    if ch2 and gt["n_universal"]:
        ch2 = re.sub(r'\b\d[\d,]*\s+rows?\s+(?:flagged|analysed|analyzed|examined)',
                     f"{gt['n_universal']} rows flagged", ch2, count=1)
    story["chapter_2_what_we_checked"] = ch2

    # Strip fabricated total counts from chapter_3 — dynamic, no hardcoded values.
    # Python builds the set of verified numbers; any sentence that contains an
    # unknown 3+ digit integer AND talks about totals/counts is dropped.
    ch3 = story.get("chapter_3_wow_moments", "")
    if ch3:
        allowed_numbers = {
            str(total_flagged),
            str(gt["n_universal"]),
            str(gt["n_enterprise"]),
            str(gt["n_wow"]),
        }
        count_keywords = {"identified", "transactions", "flagged", "rows"}
        clean = []
        for sentence in ch3.split(". "):
            words = sentence.replace(",", "").split()
            invented = [
                w for w in words
                if w.isdigit() and len(w) >= 3 and w not in allowed_numbers
            ]
            is_count_sentence = bool(count_keywords & set(sentence.lower().split()))
            if invented and is_count_sentence:
                continue  # drop sentence with fabricated total
            clean.append(sentence)
        story["chapter_3_wow_moments"] = ". ".join(clean).strip()

    return story


class StoryAgent:
    """Turns the full pipeline execution into a chapter-by-chapter narrative.

    Reads every finding in order and writes a flowing story like a consultant
    walking a business executive through what the data revealed — step by step.

    Output chapters:
      chapter_1_data_overview   — What data did we have?
      chapter_2_what_we_checked — What did we check, step by step?
      chapter_3_wow_moments     — What surprising things did we find?
      chapter_4_what_to_do      — What should be done about it?
      one_liner                 — One sentence for the executive summary card.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.output_llm_model,
            temperature=0.3,
            max_tokens=max(self.config.output_llm_max_tokens, 1500),
            max_retries=0,
        )
        # Fallback to 8b when the 70b quota window is exhausted (retry-after > 60s).
        self._fallback_llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=0.3,
            max_tokens=max(self.config.output_llm_max_tokens, 1500),
            max_retries=0,
        )

    def run(self, state: dict) -> dict:
        """Generate a narrative story from pipeline state. Never raises."""
        try:
            # Step 1: compute all numeric facts from pipeline state — Python owns these
            gt = _compute_story_ground_truth(state)
            # Step 2: LLM writes only qualitative chapters (3, 4, one_liner)
            context = self._build_context(state, gt)
            story   = self._call_llm(context, state)
            # Step 3: override chapter_1 and chapter_2 counts with Python-built text
            story   = _merge_story_ground_truth(story, gt)
            log.info("StoryAgent complete.")
            return story
        except Exception as exc:
            log.error("StoryAgent failed: %s", exc)
            return self._fallback(state)

    # ── Context builder ────────────────────────────────────────────────────────

    def _build_context(self, state: dict, gt: dict) -> str:
        """Summarise pipeline state into a concise text block for the LLM."""
        lines: list[str] = []

        # --- Data overview (Python ground truth — LLM must NOT restate these numbers) ---
        tables = gt["tables"]
        lines.append(f"TABLES: {', '.join(tables) if tables else 'unknown'}")
        if gt["total_rows"]:
            lines.append(f"TOTAL ROWS: {gt['total_rows']:,}")
        if gt["type_counts"]:
            lines.append(f"COLUMNS BY TYPE (exact — do not restate in chapter_1/2): {gt['type_counts']}")
        lines.append(
            f"FLAGGED ROWS: universal={gt['n_universal']}, "
            f"enterprise={gt['n_enterprise']}, wow={gt['n_wow']}, "
            f"total_flagged={gt['total_flagged']} "
            f"(exact — do not restate in chapter_1/2)"
        )

        # --- Universal agent findings (replaces old 4-type loop) ---
        # Prefer explicit universal_results; fall back to numerical_results alias.
        all_standard = (
            state.get("universal_results", [])
            or state.get("numerical_results", [])
        )
        universal = (
            [r for r in all_standard if r.get("agent_name") == "universal"]
            or all_standard
        )
        if universal:
            by_table: dict[str, list] = {}
            for r in universal:
                by_table.setdefault(r.get("table", "unknown"), []).append(r)

            # Signal type breakdown (dynamic — parses prefix before ':')
            type_counts_u: dict[str, int] = {}
            for r in universal:
                for sig in r.get("analysis", {}).get("all_signals", []):
                    prefix = sig.split(":")[0] if ":" in sig else "other"
                    type_counts_u[prefix] = type_counts_u.get(prefix, 0) + 1

            lines.append(
                f"\nUNIVERSAL AGENT — {len(universal)} flagged rows "
                f"across {len(by_table)} table(s) | "
                f"signal types: {dict(sorted(type_counts_u.items(), key=lambda x: -x[1]))}"
            )
            # Top 2 rows per table (highest signal count)
            for table, rows in list(by_table.items())[:3]:
                top = sorted(
                    rows,
                    key=lambda r: r.get("analysis", {}).get("signal_count", 0),
                    reverse=True,
                )[:2]
                for r in top:
                    it = (r.get("insights_text") or "").strip()
                    if it:
                        lines.append(
                            f"  [{table}.{r.get('column','')}]: {it[:180]}"
                        )

        # --- Enterprise row results ---
        ent_results = state.get("enterprise_row_results") or []
        if ent_results:
            # Count signal types dynamically
            ent_type_counts: dict[str, int] = {}
            for r in ent_results:
                for sig in r.get("analysis", {}).get("signals", []):
                    prefix = sig.split(":")[0] if ":" in sig else sig[:10]
                    ent_type_counts[prefix] = ent_type_counts.get(prefix, 0) + 1

            dup_count = sum(1 for r in ent_results if r.get("analysis", {}).get("is_duplicate"))
            lines.append(
                f"\nENTERPRISE AGENT — {len(ent_results)} flagged rows | "
                f"duplicates={dup_count} | "
                f"signals: {dict(sorted(ent_type_counts.items(), key=lambda x: -x[1]))}"
            )
            top_ent = sorted(
                ent_results,
                key=lambda r: r.get("analysis", {}).get("signal_count", 0),
                reverse=True,
            )[:3]
            for r in top_ent:
                it = (r.get("insights_text") or "").strip()
                if it:
                    lines.append(f"  [{r.get('table','')}.{r.get('column','')}]: {it[:150]}")

        # --- WoW row results ---
        wow_results = state.get("wow_row_results") or []
        if wow_results:
            # Count pattern types dynamically from signal_types field
            wow_type_counts: dict[str, int] = {}
            for r in wow_results:
                for sig_type in r.get("analysis", {}).get("signal_types", []):
                    wow_type_counts[sig_type] = wow_type_counts.get(sig_type, 0) + 1

            lines.append(
                f"\nWOW AGENT — {len(wow_results)} flagged rows | "
                f"pattern types: {dict(sorted(wow_type_counts.items(), key=lambda x: -x[1]))}"
            )
            top_wow = sorted(
                wow_results,
                key=lambda r: r.get("analysis", {}).get("signal_count", 0),
                reverse=True,
            )[:3]
            for r in top_wow:
                it = (r.get("insights_text") or "").strip()
                if it:
                    lines.append(f"  [{r.get('table','')}.{r.get('column','')}]: {it[:150]}")

        # --- Insights from InsightAgent ---
        insights = state.get("insights", {})
        if insights:
            lines.append(f"\nKEY FINDINGS (InsightAgent):")
            for f in insights.get("key_findings", [])[:3]:
                lines.append(f"  - {f}")
            lines.append(
                f"DATA QUALITY SCORE: {insights.get('data_quality_score', 'N/A')}/100"
            )
            narrative = insights.get("overall_narrative", "")
            if narrative:
                lines.append(f"NARRATIVE: {narrative[:300]}")
            wow_highlights = insights.get("wow_highlights", [])
            if wow_highlights:
                lines.append("WOW HIGHLIGHTS:")
                for h in wow_highlights[:3]:
                    lines.append(f"  - {h}")

        return "\n".join(lines)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _build_system_prompt(self, state: dict) -> str:
        """Build a dynamic audit-narrative system prompt from actual pipeline context."""
        tables     = list((state.get("raw_data") or state.get("schema_info", {})).keys())
        universal  = state.get("universal_results", []) or state.get("numerical_results", [])
        ent        = state.get("enterprise_row_results", []) or []
        wow        = state.get("wow_row_results", []) or []
        insights   = state.get("insights", {})

        high_risk_count = sum(
            1 for r in universal + ent + wow
            if "HIGH RISK" in (r.get("insights_text") or "").upper()
        )
        audit_priority = insights.get("audit_priority", "MEDIUM")
        table_str = ", ".join(tables) if tables else "financial tables"

        return (
            f"You are a senior audit partner presenting findings to the audit committee.\n\n"
            f"Engagement context: {len(tables)} financial table(s) audited — {table_str}. "
            f"Overall audit priority: {audit_priority}. "
            f"High-risk transactions identified: {high_risk_count}.\n\n"
            "Your job is to turn the audit pipeline's results into an authoritative, "
            "chapter-by-chapter narrative for the audit committee — evidence-based, specific, "
            "and action-oriented. Cite actual table names, column names, anomalous values, "
            "and risk levels (HIGH/MEDIUM/LOW) from the context. Never be vague.\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            "{\n"
            '  "chapter_1_data_overview": "The audit covered... '
            "(write a placeholder sentence only — do NOT write any numbers, table names, or column counts here "
            "— Python will replace this entire field with verified facts)\",\n"
            '  "chapter_2_what_we_checked": "Our audit procedures included... '
            "(write exactly 3 sentences — one per agent: "
            "(1) Universal agent: describe that it scored every row for statistical anomalies across all numerical, categorical, datetime, and text columns using multi-signal scoring; "
            "(2) Enterprise agent: describe that it checked for data quality issues — NULL values, duplicate records, type mismatches, and fraud signals such as stop payments and account closures; "
            "(3) WoW agent: describe that it detected unexpected business patterns including off-hours transactions, weekend activity, period-end clustering, and sudden value spikes. "
            "Do NOT write any row counts or numbers — Python will correct them)\",\n"
            '  "chapter_3_wow_moments": "The most material finding was... '
            "(lead with the highest-risk anomaly — cite exact table name, anomalous amount, multiplier vs average, and risk level "
            "from the KEY FINDINGS block — these numbers ARE correct and must be used exactly. "
            "Then describe the second-highest finding. Then state what these findings collectively indicate about control quality. "
            "3–4 sentences total)\",\n"
            '  "chapter_4_what_to_do": "We recommend... '
            "(specific, prioritised audit actions — name the responsible process or system owner for each action. "
            "First address the highest-risk finding, then data quality, then broader controls. "
            "3–4 sentences total)\",\n"
            '  "one_liner": "One sentence — must name the single most critical table and anomalous amount found, and state the business risk in plain English. Under 30 words."\n'
            "}\n\n"
            "Rules:\n"
            "- Write in past tense (audit has been performed)\n"
            f"- Chapter 3 must mention the {high_risk_count} HIGH RISK rows if any exist; "
            "otherwise lead with the highest-signal anomaly\n"
            "- If fraud signals exist (Account Closed, Stop Payment, structuring), they must dominate chapter 3\n"
            "- Chapter 3 and 4: cite actual table names and amounts from KEY FINDINGS — never invent numbers\n"
            "- Chapter 3 must NOT restate the total flagged count — chapter_1 owns that number; focus only on the top 2–3 individual findings\n"
            "- Chapter 2: describe each agent's function in plain English — no jargon, no numbers\n"
            "- one_liner: must reference a specific table name or amount — never be generic\n"
            "- Do NOT wrap output in markdown code fences"
        )

    def _call_llm(self, context: str, state: dict | None = None) -> dict:
        system = self._build_system_prompt(state or {})

        user = (
            f"Here is the audit pipeline execution summary:\n\n{context}\n\n"
            "Write the audit committee narrative as JSON."
        )

        response = groq_invoke(self.llm, [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ], agent_name="story", fallback_llm=self._fallback_llm)

        text = response.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

        return json.loads(text)

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback(self, state: dict) -> dict:
        raw_data    = state.get("raw_data") or {}
        schema_info = state.get("schema_info", {})
        tables      = list(raw_data.keys()) if raw_data else list(schema_info.keys())
        routes      = state.get("column_routes", [])

        # Best single wow finding (highest signal_count)
        wow_results = state.get("wow_row_results") or []
        top_wow = (
            sorted(wow_results, key=lambda r: r.get("analysis", {}).get("signal_count", 0), reverse=True)
        )
        top_wow_text = (
            (top_wow[0].get("insights_text") or "").strip()[:150]
            if top_wow else "Several business patterns were detected across the dataset."
        )

        # Universal flagged row count
        universal = (
            state.get("universal_results", []) or state.get("numerical_results", [])
        )

        return {
            "chapter_1_data_overview": (
                f"We analyzed {len(tables)} table(s) — {', '.join(tables)} — "
                f"covering {len(routes)} columns in total."
            ),
            "chapter_2_what_we_checked": (
                f"The pipeline ran the universal row agent (flagging {len(universal)} anomalous rows "
                "across all column types), the enterprise data-quality agent (detecting NULL values, "
                "duplicates, type mismatches, and fraud signals), and the WoW pattern agent "
                "(scanning for hidden business patterns and leading indicators)."
            ),
            "chapter_3_wow_moments": top_wow_text,
            "chapter_4_what_to_do": (
                "Review the key findings and recommendations in the report below "
                "for specific, actionable next steps."
            ),
            "one_liner": "Analysis complete — see the report for insights and recommendations.",
        }
