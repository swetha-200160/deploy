from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from config import Config, get_config
from core.state import AgentResult, PipelineState

log = logging.getLogger(__name__)


class ReportingAgent:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()

    def build(self, state: dict) -> dict:
        """Build final JSON + HTML report from pipeline state."""
        run_id = state.get("run_id", "report")
        run_dir = os.path.join(self.config.report_output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Prefer explicit universal_results; fall back to numerical_results alias.
        universal_results = (
            state.get("universal_results", [])
            or [r for r in state.get("numerical_results", []) if r.get("agent_name") == "universal"]
            or state.get("numerical_results", [])
        )

        report = {
            "run_id":                 state.get("run_id", ""),
            "generated_at":           datetime.now().isoformat(),
            "tables_analyzed":        state.get("tables_analyzed", list(state.get("raw_data", {}).keys())),
            "total_columns_analyzed": len(state.get("column_routes", [])),
            "agents": {
                "universal":      universal_results,
                "enterprise_row": state.get("enterprise_row_results", []),
                "wow_row":        state.get("wow_row_results", []),
            },
            "insights": state.get("insights", {}),
            "story":    state.get("story", {}),
            "charts":   {k: "<!-- embedded chart -->" for k in state.get("charts", {}).keys()},
            "status":   state.get("status", "done"),
        }

        # Save JSON report
        json_path = os.path.join(run_dir, f"{run_id}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            log.info("JSON report saved: %s", json_path)
        except Exception as exc:
            log.error("Failed to save JSON report: %s", exc)

        # Build and save HTML report — stream directly to file to avoid MemoryError
        html_path = os.path.join(run_dir, f"{run_id}.html")
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                self._write_html(f, report, state.get("charts", {}), state.get("story", {}))
            log.info("HTML report saved: %s", html_path)
        except Exception as exc:
            log.error("Failed to save HTML report: %s", exc)

        report["_html_path"] = html_path
        report["_json_path"] = json_path
        return report

    # ── HTML builder ──────────────────────────────────────────────────────────

    def _write_html(self, f, report: dict, charts: dict[str, str], story: dict | None = None) -> None:
        """Write HTML report directly to file object, streaming each section to avoid MemoryError."""
        insights = report.get("insights", {})
        story    = story or {}

        findings_html = "".join(f"<li>{fi}</li>" for fi in insights.get("key_findings", []))
        risks_html    = "".join(f"<li>{r}</li>" for r in insights.get("risk_signals", []))
        recs_html     = "".join(f"<li>{r}</li>" for r in insights.get("recommendations", []))

        # ── Head + header ──────────────────────────────────────────────────
        f.write(f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>4sight v3 — Audit Report — {report.get('run_id', 'Report')}</title>
  <style>
    body {{font-family: Arial, sans-serif; margin: 0; background: #f5f6fa;}}
    .header {{background: #1a237e; color: white; padding: 32px 40px;}}
    .section {{margin: 32px 40px;}}
    h2 {{color: #1a237e; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px;}}
    h3 {{color: #333;}}
    .card {{background: white; border-radius: 8px; padding: 24px; margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);}}
    .score {{font-size: 48px; font-weight: bold; color: #4caf50;}}
    .result-card {{background: white; padding: 14px 18px; margin-bottom: 10px;
                   border-radius: 0 8px 8px 0; font-size: 13px; line-height: 1.5;}}
    .result-card.universal    {{border-left: 4px solid #1a237e;}}
    .result-card.enterprise   {{border-left: 4px solid #e53935;}}
    .result-card.wow          {{border-left: 4px solid #00897b;}}
    .result-card h4 {{margin: 0 0 4px; font-size: 13px; color: #333;}}
    .result-card .signals {{font-size: 11px; color: #888; margin-bottom: 4px;}}
    .result-card p {{margin: 0; color: #555;}}
    .chart-wrap {{background: white; padding: 8px; border-radius: 8px; margin-bottom: 16px;
                  overflow-x: auto; box-shadow: 0 1px 4px rgba(0,0,0,0.06);}}
    .agent-section {{margin-bottom: 28px;}}
    .agent-section h3 {{font-size: 15px; text-transform: uppercase; letter-spacing: 0.05em;
                        padding: 8px 12px; border-radius: 4px; margin-bottom: 8px;}}
    .agent-section.universal h3  {{background: #e8eaf6; color: #1a237e;}}
    .agent-section.enterprise h3 {{background: #ffebee; color: #c62828;}}
    .agent-section.wow h3        {{background: #e0f2f1; color: #00695c;}}
    ul {{padding-left: 24px;}} li {{margin-bottom: 6px;}}
    .story-oneliner {{font-size: 20px; font-style: italic; color: #1a237e; font-weight: 600;
                      background: #e8eaf6; border-left: 5px solid #3949ab; padding: 16px 24px;
                      border-radius: 0 8px 8px 0; margin-bottom: 20px; line-height: 1.5;}}
    .story-chapter {{background: white; border-radius: 8px; padding: 20px 24px;
                     margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                     border-top: 3px solid #7986cb;}}
    .story-chapter h3 {{color: #3949ab; margin-top: 0; font-size: 15px; text-transform: uppercase;
                        letter-spacing: 0.05em;}}
    .story-chapter p  {{color: #333; line-height: 1.7; margin: 0;}}
  </style>
</head>
<body>
<div class="header">
  <h1>4sight v3 — Financial Audit Analysis Report</h1>
  <p>Run ID: {report.get('run_id', '')} | Generated: {report.get('generated_at', '')} |
     Tables Audited: {len(report.get('tables_analyzed', []))} | Columns Examined: {report.get('total_columns_analyzed', 0)} |
     Audit Priority: <strong>{insights.get('audit_priority', 'N/A')}</strong></p>
</div>

<div class="section">
  <h2>Audit Executive Summary</h2>
  <div class="card">
    <p>{insights.get('overall_narrative', '')}</p>
    <div class="score">{insights.get('data_quality_score', 0)}<span style="font-size:18px">/100</span></div>
    <p style="color:#888">Audit Risk Score (100 = clean, 0 = severely compromised)</p>
  </div>
  <div class="card">
    <h3>Key Audit Findings</h3><ul>{findings_html}</ul>
    <h3>Fraud &amp; Control Risk Signals</h3><ul>{risks_html}</ul>
    <h3>Audit Recommendations</h3><ul>{recs_html}</ul>
  </div>
</div>
""")

        # ── Story section ──────────────────────────────────────────────────
        if story:
            chapters = [
                ("Audit Scope",           story.get("chapter_1_data_overview", "")),
                ("Procedures Performed",  story.get("chapter_2_what_we_checked", "")),
                ("Material Findings",     story.get("chapter_3_wow_moments", "")),
                ("Audit Recommendations", story.get("chapter_4_what_to_do", "")),
            ]
            one_liner = story.get("one_liner", "")
            has_story = one_liner or any(body for _, body in chapters)
            if has_story:
                f.write('<div class="section"><h2>Data Story</h2>\n')
                if one_liner:
                    f.write(f"<div class='story-oneliner'>\"{one_liner}\"</div>\n")
                for title, body in chapters:
                    if body:
                        f.write(
                            f"<div class='story-chapter'>"
                            f"<h3>{title}</h3><p>{body}</p>"
                            f"</div>\n"
                        )
                f.write("</div>\n")

        # ── Charts section — written one chart at a time ───────────────────
        f.write('<div class="section">\n  <h2>Charts</h2>\n')
        if charts:
            for name, chart_html in charts.items():
                f.write(f"<h3>{name}</h3><div class='chart-wrap'>{chart_html}</div>\n")
                del chart_html  # release immediately after writing
        else:
            f.write("  <p>No charts generated.</p>\n")
        f.write("</div>\n")

        # ── Agent results ──────────────────────────────────────────────────
        f.write('<div class="section">\n  <h2>Agent Analysis Results</h2>\n')
        agents_html = self._build_agents_html(report.get("agents", {}))
        f.write(agents_html or "  <p>No agent results.</p>\n")
        f.write("</div>\n</body>\n</html>")

    # kept for backward-compat (not called internally)
    def _build_html(self, report: dict, charts: dict[str, str], story: dict | None = None) -> str:
        import io
        buf = io.StringIO()
        self._write_html(buf, report, charts, story)
        return buf.getvalue()

    def _build_agents_html(self, agents: dict) -> str:
        """Render HTML cards for each agent type dynamically."""
        html = ""

        # ── Universal ──────────────────────────────────────────────────────
        universal = agents.get("universal", [])
        if universal:
            # Sort by signal count descending; cap display at 50 cards
            top = sorted(
                (r for r in universal if not r.get("error")),
                key=lambda r: r.get("analysis", {}).get("signal_count", 0),
                reverse=True,
            )[:50]
            cards = ""
            for r in top:
                sigs = r.get("analysis", {}).get("all_signals", [])
                sc   = r.get("analysis", {}).get("signal_count", 0)
                sig_text = " &nbsp;·&nbsp; ".join(sigs[:4]) if sigs else ""
                cards += f"""
<div class='result-card universal'>
  <h4>Universal — {r.get('table','')} / {r.get('column','')} &nbsp;
      <span style='color:#e53935'>({sc} signal{'s' if sc!=1 else ''})</span></h4>
  {f"<div class='signals'>{sig_text}</div>" if sig_text else ""}
  <p><em>{r.get('insights_text','')}</em></p>
</div>"""
            html += f"""
<div class='agent-section universal'>
  <h3>Anomaly &amp; Fraud Detection Agent — Universal ({len(universal)} flagged rows — top {len(top)} shown)</h3>
  {cards}
</div>"""

        # ── Enterprise ─────────────────────────────────────────────────────
        enterprise = agents.get("enterprise_row", [])
        if enterprise:
            top_ent = sorted(
                (r for r in enterprise if not r.get("error")),
                key=lambda r: r.get("analysis", {}).get("signal_count", 0),
                reverse=True,
            )[:50]
            cards = ""
            for r in top_ent:
                sigs     = r.get("analysis", {}).get("signals", [])
                is_dup   = r.get("analysis", {}).get("is_duplicate", False)
                sig_text = " &nbsp;·&nbsp; ".join(sigs[:4]) if sigs else ""
                dup_badge = " <span style='color:#c62828;font-size:11px'>[DUP]</span>" if is_dup else ""
                cards += f"""
<div class='result-card enterprise'>
  <h4>Enterprise — {r.get('table','')} / {r.get('column','')}{dup_badge}</h4>
  {f"<div class='signals'>{sig_text}</div>" if sig_text else ""}
  <p><em>{r.get('insights_text','')}</em></p>
</div>"""
            html += f"""
<div class='agent-section enterprise'>
  <h3>Data Quality &amp; Fraud Control Agent — Enterprise ({len(enterprise)} flagged rows — top {len(top_ent)} shown)</h3>
  {cards}
</div>"""

        # ── WoW ────────────────────────────────────────────────────────────
        wow = agents.get("wow_row", [])
        if wow:
            top_wow = sorted(
                (r for r in wow if not r.get("error")),
                key=lambda r: r.get("analysis", {}).get("signal_count", 0),
                reverse=True,
            )[:50]
            cards = ""
            for r in top_wow:
                sig_types = r.get("analysis", {}).get("signal_types", [])
                sig_text  = " &nbsp;·&nbsp; ".join(sig_types) if sig_types else ""
                cards += f"""
<div class='result-card wow'>
  <h4>WoW — {r.get('table','')} / {r.get('column','')}</h4>
  {f"<div class='signals'>{sig_text}</div>" if sig_text else ""}
  <p><em>{r.get('insights_text','')}</em></p>
</div>"""
            html += f"""
<div class='agent-section wow'>
  <h3>Advanced Pattern Detection Agent — WoW ({len(wow)} flagged rows — top {len(top_wow)} shown)</h3>
  {cards}
</div>"""

        return html
