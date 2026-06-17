from __future__ import annotations

import os
import shutil
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from config import get_config
from api.schemas import AvailableRunsResponse, DeleteResponse

router = APIRouter(prefix="/api/delete", tags=["Delete"])


def _scan_runs(base_dir: str) -> list[str]:
    if not os.path.exists(base_dir):
        return []
    return sorted(e for e in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, e)))


def _scan_files(base_dir: str) -> list[str]:
    if not os.path.exists(base_dir):
        return []
    return sorted(e for e in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, e)))


def _build_html(charts: list[str], faiss: list[str], reports: list[str]) -> str:
    def _buttons(items: list[str], endpoint: str, color: str) -> str:
        if not items:
            return '<p style="color:#9e9e9e;font-style:italic">No items available.</p>'
        return "".join(
            f'<button class="btn" style="border-color:{color}" '
            f'onclick="confirmDelete(\'{endpoint}\',\'{item}\',this)">{item}</button>\n'
            for item in items
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>4sight — Delete Manager</title>
  <style>
    body{{font-family:Arial,sans-serif;background:#f5f5f5;padding:32px}}
    h1{{color:#212121;margin-bottom:4px}}
    .sub{{color:#757575;margin-bottom:28px}}
    .section{{background:#fff;border-radius:8px;padding:24px;margin-bottom:20px;
              box-shadow:0 1px 4px rgba(0,0,0,.12)}}
    h2{{margin-top:0;font-size:1rem}}
    .btns{{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}}
    .btn{{padding:8px 18px;border:2px solid #ccc;border-radius:6px;background:#fff;
          cursor:pointer;font-size:.9rem;transition:background .15s}}
    .btn:hover{{background:#f0f0f0}}
    .btn.done{{background:#ffebee;border-color:#e57373;color:#b71c1c;
               text-decoration:line-through;cursor:default}}
    #toast{{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);
            background:#323232;color:#fff;padding:12px 24px;border-radius:6px;
            display:none;font-size:.9rem;z-index:999}}
  </style>
</head>
<body>
  <h1>Delete Manager</h1>
  <p class="sub">Select a run ID to delete. Action cannot be undone.</p>

  <div class="section">
    <h2 style="color:#1565c0">Charts</h2>
    <div class="btns">{_buttons(charts, "/api/delete/charts", "#1565c0")}</div>
  </div>

  <div class="section">
    <h2 style="color:#6a1b9a">FAISS Indexes</h2>
    <div class="btns">{_buttons(faiss, "/api/delete/faiss", "#6a1b9a")}</div>
  </div>

  <div class="section">
    <h2 style="color:#b71c1c">Reports</h2>
    <div class="btns">{_buttons(reports, "/api/delete/reports", "#b71c1c")}</div>
  </div>

  <div id="toast"></div>
  <script>
    function showToast(msg, ok) {{
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.style.background = ok ? '#2e7d32' : '#c62828';
      t.style.display = 'block';
      setTimeout(() => t.style.display = 'none', 3000);
    }}
    async function confirmDelete(endpoint, item, btn) {{
      if (!confirm('Delete "' + item + '"?')) return;
      const res = await fetch(endpoint + '?item=' + encodeURIComponent(item), {{method:'DELETE'}});
      const data = await res.json();
      if (res.ok) {{
        btn.classList.add('done');
        btn.disabled = true;
        showToast('"' + item + '" deleted.', true);
      }} else {{
        showToast(data.detail || 'Error', false);
      }}
    }}
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# GET /api/delete  — show available run IDs (json | html | both)
# ---------------------------------------------------------------------------

@router.get("")
async def list_available(format: Literal["json", "html", "both"] = "html"):
    """
    Show available run IDs for charts, FAISS, and reports.

    - **json** → AvailableRunsResponse
    - **html** → interactive page with clickable delete buttons
    - **both** → JSON + rendered HTML string
    """
    config = get_config()
    charts  = _scan_runs(config.chart_output_dir)
    faiss   = _scan_runs(config.faiss_index_path)
    reports = _scan_files(config.report_output_dir)

    if format == "json":
        return AvailableRunsResponse(charts=charts, faiss=faiss, reports=reports)

    html = _build_html(charts, faiss, reports)

    if format == "both":
        return JSONResponse(content={
            "available": {"charts": charts, "faiss": faiss, "reports": reports},
            "html": html,
        })

    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# DELETE endpoints (called by the UI buttons or directly)
# ---------------------------------------------------------------------------

@router.delete("/charts", response_model=DeleteResponse)
async def delete_charts(item: str):
    """Delete a chart run directory. Pass the run_id shown in GET /api/delete."""
    config = get_config()
    target = os.path.join(config.chart_output_dir, item)
    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail=f"Chart run '{item}' not found.")
    shutil.rmtree(target)
    return DeleteResponse(deleted=True, item=item, type="charts")


@router.delete("/faiss", response_model=DeleteResponse)
async def delete_faiss(item: str):
    """Delete a FAISS index directory. Pass the run_id shown in GET /api/delete."""
    config = get_config()
    target = os.path.join(config.faiss_index_path, item)
    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail=f"FAISS index '{item}' not found.")
    shutil.rmtree(target)
    return DeleteResponse(deleted=True, item=item, type="faiss")


@router.delete("/reports", response_model=DeleteResponse)
async def delete_reports(item: str):
    """Delete a report file. Pass the filename shown in GET /api/delete."""
    config = get_config()
    target = os.path.join(config.report_output_dir, item)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail=f"Report '{item}' not found.")
    if os.path.isdir(target):
        shutil.rmtree(target)
    else:
        os.remove(target)
    return DeleteResponse(deleted=True, item=item, type="reports")
