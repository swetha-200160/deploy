# 4sight v3 — Report Frontend UI Guide

## Primary Endpoint

```
GET /api/report/json?run_id={run_id}
```

This returns the full saved report JSON file directly from disk.  
Response is `application/json` — parse it and render the 7 sections below.

---

## Step 1 — Fetch the Report (JavaScript)

```js
const BASE_URL = "http://localhost:8000";

// 1. Get list of available runs
async function listRuns() {
  const res = await fetch(`${BASE_URL}/api/reports`);
  const data = await res.json();
  return data.saved_reports; // [{ run_id, json_path, size_kb }, ...]
}

// 2. Load a specific report
async function loadReport(run_id) {
  const res = await fetch(`${BASE_URL}/api/report/json?run_id=${run_id}`);
  const report = await res.json();
  return report;
}

// Usage
const report = await loadReport("run_20260601_095828");
```

---

## Step 2 — Response Structure

```json
{
  "run_id": "run_20260601_095828",
  "generated_at": "2026-06-01T10:02:14",
  "tables_analyzed": [
    "foresight_cheque_clearing_67318",
    "foresight_missing_grn_67222",
    "foresight_price_variance_67227",
    "foresight_quantity_variance_67226",
    "foresight_query_sessions",
    "foresight_report_summary"
  ],
  "total_columns_analyzed": 207,

  "agents": {
    "universal":      [ ...511 rows ],
    "enterprise_row": [ ...336 rows ],
    "wow_row":        [ ...481 rows ]
  },

  "insights": {
    "audit_priority":      "IMMEDIATE",
    "data_quality_score":  40,
    "overall_narrative":   "...",
    "key_findings":        [ ...5 items ],
    "risk_signals":        [ ...items ],
    "recommendations":     [ ...items ],
    "wow_highlights":      [ ...items ],
    "insights": {
      "foresight_cheque_clearing_67318": { ... },
      "foresight_missing_grn_67222":     { ... },
      ...
    }
  },

  "story": {
    "one_liner":                  "...",
    "chapter_1_data_overview":    "...",
    "chapter_2_what_we_checked":  "...",
    "chapter_3_wow_moments":      "...",
    "chapter_4_what_to_do":       "..."
  },

  "status": "done"
}
```

---

## Step 3 — Map Response to UI Sections

---

### Section 1 — Executive Summary

| UI Element | JSON Field | Example Value |
|---|---|---|
| Risk priority badge | `report.insights.audit_priority` | `"IMMEDIATE"` |
| Headline sentence | `report.story.one_liner` | `"Audit identified control failures..."` |
| Summary paragraph | `report.insights.overall_narrative` | `"302 transactions..."` |
| Total tables | `report.tables_analyzed.length` | `6` |
| Total flagged rows | `report.agents.universal.length + report.agents.enterprise_row.length + report.agents.wow_row.length` | `1328` |
| Data quality score | `report.insights.data_quality_score` | `40` |
| Report date | `report.generated_at` | `"2026-06-01T10:02:14"` |

```js
const summary = {
  priority:     report.insights.audit_priority,
  one_liner:    report.story.one_liner,
  narrative:    report.insights.overall_narrative,
  tables:       report.tables_analyzed.length,
  flagged:      report.agents.universal.length
                + report.agents.enterprise_row.length
                + report.agents.wow_row.length,
  score:        report.insights.data_quality_score,
  generated_at: report.generated_at,
};
```

**UI:**
```
[ IMMEDIATE ]  "The audit identified potential control failures in 5 tables..."

  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Tables:6 │  │Flagged:  │  │Quality:  │  │Date:     │
  │          │  │  1,328   │  │ 40/100   │  │2026-06-01│
  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

### Section 2 — Story (Audit Narrative)

| UI Tab | JSON Field |
|---|---|
| Data Overview | `report.story.chapter_1_data_overview` |
| What We Checked | `report.story.chapter_2_what_we_checked` |
| Key Moments | `report.story.chapter_3_wow_moments` |
| What To Do | `report.story.chapter_4_what_to_do` |

```js
const chapters = [
  { title: "Data Overview",    text: report.story.chapter_1_data_overview },
  { title: "What We Checked",  text: report.story.chapter_2_what_we_checked },
  { title: "Key Moments",      text: report.story.chapter_3_wow_moments },
  { title: "What To Do",       text: report.story.chapter_4_what_to_do },
];
```

> Render as 4 tabs or 4 accordion panels — plain text, no parsing needed.

---

### Section 3 — Risk Signals

Each item in `report.insights.risk_signals[]`:

```json
{
  "table":               "foresight_cheque_clearing_67318",
  "column":              "amount",
  "signal_type":         "off-hours / temporal anomaly",
  "value":               200,
  "statistical_context": 6800.68,
  "description":         "200 off-hours exceptions represent a potential payment window control failure..."
}
```

```js
const riskSignals = report.insights.risk_signals;
// Sort: IMMEDIATE first, then by value (count) descending
```

**UI card per signal:**
```
┌─────────────────────────────────────────────────────────┐
│  [HIGH]  off-hours / temporal anomaly                   │
│  Table: foresight_cheque_clearing_67318 → amount        │
│  Exceptions: 200  |  Avg: 6,800.68                      │
│  ───────────────────────────────────────────────────    │
│  200 off-hours exceptions represent a potential         │
│  payment window control failure, exposing 172,600...    │
└─────────────────────────────────────────────────────────┘
```

---

### Section 4 — Key Findings

Each item in `report.insights.key_findings[]`:

```json
{
  "table":               "foresight_price_variance_67227",
  "column":              "po_amount",
  "row":                 "row_33",
  "anomalous_value":     241181.0,
  "statistical_context": 28062.52,
  "description":         "A transaction of 241,181 represents 8.6x the register average..."
}
```

```js
const findings = report.insights.key_findings;

// Compute deviation multiple for display
findings.forEach(f => {
  f.deviation = (f.anomalous_value / f.statistical_context).toFixed(1); // "8.6x"
});
```

**UI card per finding:**
```
┌─────────────────────────────────────────────────────────┐
│  Finding #1                              [HIGH RISK]    │
│  foresight_price_variance_67227 → po_amount             │
│  Value: 241,181.00   (8.6x avg of 28,062.52)            │
│  ───────────────────────────────────────────────────    │
│  Potential approval-limit bypass or segregation-of-     │
│  duties failure, creating unapproved expenditure...     │
└─────────────────────────────────────────────────────────┘
```

---

### Section 5 — Per-Table Insights

Each key in `report.insights.insights` is a table name:

```js
const tableInsights = report.insights.insights;
// { "foresight_cheque_clearing_67318": {...}, "foresight_missing_grn_67222": {...}, ... }

Object.entries(tableInsights).forEach(([tableName, data]) => {
  // data.condition          → "Off-hours / temporal anomaly detected..."
  // data.top_amount         → 172600
  // data.deviation_multiple → 25.4
  // data.control_weakness   → "payment window control failure"
  // data.financial_consequence → "Exposes 172,600 to unauthorized payment..."
  // data.named_entities[]   → ["Arshitha", "Al Batinah South", ...]
  // data.narrative          → full paragraph
});
```

**UI — accordion, one panel per table:**
```
▼  foresight_cheque_clearing_67318
   ─────────────────────────────────────────────
   Condition:         Off-hours / temporal anomaly
   Top Amount:        172,600
   Deviation:         25.4x
   Control Weakness:  Payment window control failure
   Financial Impact:  Exposes 172,600 to unauthorized payment
   Entities:          Arshitha, Al Batinah South, Mon...
   ─────────────────────────────────────────────
   "The off-hours pattern is driven by entities such as
    Aaliyah and Arshitha, with amounts exceeding..."

▶  foresight_missing_grn_67222
▶  foresight_price_variance_67227
```

---

### Section 6 — Flagged Transactions Table

Combine all 3 agent arrays:

```js
const allFlagged = [
  ...report.agents.universal.map(r => ({ ...r, agent: "Universal" })),
  ...report.agents.enterprise_row.map(r => ({ ...r, agent: "Enterprise" })),
  ...report.agents.wow_row.map(r => ({ ...r, agent: "WoW" })),
];
// Total: 1,328 rows
```

Each row object:

```json
{
  "agent_name":    "universal",
  "table":         "foresight_cheque_clearing_67318",
  "column":        "row_0",
  "insights_text": "HIGH RISK\nA suspicious cheque_date=00:00 outside business hours...",
  "analysis": {
    "row_index":        2,
    "signal_count":     2,
    "all_signals":      ["dt:cheque_date=00:00(off-hours)", "dt:cperiod=00:00(off-hours)"],
    "all_values": {
      "tran_ref_no":   "CM2026000065",
      "amount":        "4600.0",
      "cheque_date":   "2024-12-07 00:00:00",
      "created_by":    "Arshitha",
      "cheque_status": "Cleared",
      ...
    }
  }
}
```

```js
// Extract risk level from insights_text
function getRiskLevel(insights_text) {
  if (insights_text?.includes("HIGH RISK"))   return "HIGH";
  if (insights_text?.includes("MEDIUM RISK")) return "MEDIUM";
  return "LOW";
}

// Build table rows
const tableRows = allFlagged.map(r => ({
  agent:       r.agent_name,
  table:       r.table,
  row:         r.analysis?.row_index,
  amount:      r.analysis?.all_values?.amount,
  signal_count: r.analysis?.signal_count,
  signals:     r.analysis?.all_signals,
  risk:        getRiskLevel(r.insights_text),
  detail:      r.analysis?.all_values,   // for expanded row view
  notes:       r.insights_text,
}));
```

**UI table:**
```
Filter: [ All Agents ▼ ] [ All Tables ▼ ] [ Risk ▼ ] [ Search... ]

Showing 1,328 rows — Universal: 511 | Enterprise: 336 | WoW: 481

┌────────────┬───────────────────────────┬──────┬──────────┬─────────┬─────────────┐
│ Agent      │ Table                     │ Row  │ Amount   │ Signals │ Risk        │
├────────────┼───────────────────────────┼──────┼──────────┼─────────┼─────────────┤
│ Universal  │ foresight_cheque_clear... │  2   │  4,600   │   2     │ [HIGH]      │
│ Enterprise │ foresight_missing_grn...  │  5   │     99   │   3     │ [MEDIUM]    │
│ WoW        │ foresight_price_var...    │ 33   │241,181   │   5     │ [HIGH]      │
└────────────┴───────────────────────────┴──────┴──────────┴─────────┴─────────────┘

▶ Click row → expand full transaction (all_values) + all signals + insights_text
```

**Row background colors:**
```js
const rowBg = { HIGH: "#FEF2F2", MEDIUM: "#FFFBEB", LOW: "#F0FDF4" };
```

---

### Section 7 — Recommendations

Each item in `report.insights.recommendations[]`:

```json
{
  "table":            "foresight_cheque_clearing_67318",
  "column":           "amount",
  "responsible_team": "Finance and Accounting",
  "action":           "Finance and Accounting must implement payment window controls..."
}
```

```js
const recommendations = report.insights.recommendations;
```

**UI — numbered list:**
```
#1  Finance and Accounting                          [IMMEDIATE]
    foresight_cheque_clearing_67318 → amount
    ────────────────────────────────────────────────────────
    Finance and Accounting must implement payment window
    controls to prevent off-hours exceptions, verify by
    reconciling payment records within 2 weeks.
    ────────────────────────────────────────────────────────
    Status: [ Open ▼ ]

#2  Procurement and Supply Chain                    [HIGH]
    foresight_price_variance_67227 → po_amount
    ...
```

---

## Risk Badge Color Reference

```js
const RISK_COLORS = {
  IMMEDIATE: { bg: "#FEF2F2", text: "#991B1B", border: "#FCA5A5" },
  HIGH:      { bg: "#FFF7ED", text: "#9A3412", border: "#FDBA74" },
  MEDIUM:    { bg: "#FFFBEB", text: "#92400E", border: "#FCD34D" },
  LOW:       { bg: "#F0FDF4", text: "#166534", border: "#86EFAC" },
};

// Tailwind classes
const RISK_TAILWIND = {
  IMMEDIATE: "bg-red-100 text-red-800 border border-red-300",
  HIGH:      "bg-orange-100 text-orange-800 border border-orange-300",
  MEDIUM:    "bg-amber-100 text-amber-800 border border-amber-300",
  LOW:       "bg-green-100 text-green-800 border border-green-300",
};
```

---

## Data Quality Score

```js
const score = report.insights.data_quality_score; // 40

const scoreLabel = score >= 80 ? "GOOD"
                 : score >= 60 ? "FAIR"
                 : score >= 40 ? "POOR"
                 : "CRITICAL";

const scoreColor = score >= 80 ? "#16A34A"
                 : score >= 60 ? "#D97706"
                 : "#DC2626";
```

```
Quality Score:  40 / 100  [ POOR ]
████░░░░░░  → Red — requires immediate attention
```

---

## Complete Fetch + Render Flow

```js
async function renderReport(run_id) {
  // 1. Fetch
  const res    = await fetch(`/api/report/json?run_id=${run_id}`);
  const report = await res.json();

  // 2. Parse
  const summary      = buildSummary(report);       // Section 1
  const chapters     = buildChapters(report);      // Section 2
  const riskSignals  = report.insights.risk_signals;           // Section 3
  const keyFindings  = report.insights.key_findings;           // Section 4
  const tableInsights = report.insights.insights;              // Section 5
  const allFlagged   = buildFlaggedRows(report);   // Section 6
  const recommendations = report.insights.recommendations;     // Section 7

  // 3. Render each section
  renderSummary(summary);
  renderChapters(chapters);
  renderRiskSignals(riskSignals);
  renderKeyFindings(keyFindings);
  renderTableInsights(tableInsights);
  renderFlaggedTable(allFlagged);
  renderRecommendations(recommendations);
}

function buildFlaggedRows(report) {
  return [
    ...report.agents.universal.map(r => ({ ...r, agent: "Universal" })),
    ...report.agents.enterprise_row.map(r => ({ ...r, agent: "Enterprise" })),
    ...report.agents.wow_row.map(r => ({ ...r, agent: "WoW" })),
  ];
}
```

---

## Available API Endpoints

| Endpoint | Use |
|---|---|
| `GET /api/reports` | List all saved runs (for dropdown selector) |
| `GET /api/report/json?run_id={run_id}` | **Primary — full report JSON** |
| `GET /api/report?run_id={run_id}` | Pre-built HTML report (browser / iframe) |
| `GET /api/report/file/{run_id}` | Same as JSON but run_id in URL path |
| `GET /api/report/download?run_id={run_id}` | Download HTML as file |

---

## Reference Run IDs

| Run ID | Date | Flagged Rows |
|---|---|---|
| `run_20260601_095828` | 2026-06-01 | ~1,328 |
| `run_20260530_115542` | 2026-05-30 | ~1,328 |
| `run_20260528_175358` | 2026-05-28 | 1,328 |

**Test URL:**
```
http://localhost:8000/api/report/json?run_id=run_20260601_095828
```
