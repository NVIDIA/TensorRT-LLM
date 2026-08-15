"""Render a promptfoo JSON eval result into a self-contained, styled HTML report.

Usage:
    python render_report.py <results.json> <out.html>
"""

import html
import json
import sys
from datetime import datetime
from pathlib import Path

CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  margin: 0;
  padding: 24px;
  line-height: 1.5;
}
.container { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px 0; color: #f8fafc; }
.subtitle { color: #94a3b8; font-size: 13px; margin-bottom: 24px; }
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 16px;
}
.card .label { color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
.card .value { font-size: 28px; font-weight: 600; margin-top: 4px; color: #f8fafc; }
.card.pass .value { color: #4ade80; }
.card.fail .value { color: #f87171; }
.card.rate .value { color: #60a5fa; }
.filters {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.filter-btn {
  background: #1e293b;
  border: 1px solid #334155;
  color: #e2e8f0;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-family: inherit;
}
.filter-btn:hover { background: #334155; }
.filter-btn.active { background: #3b82f6; border-color: #3b82f6; }
.test {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 8px;
  margin-bottom: 12px;
  overflow: hidden;
}
.test.pass { border-left: 4px solid #4ade80; }
.test.fail { border-left: 4px solid #f87171; }
.test-header {
  padding: 14px 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
  user-select: none;
}
.test-header:hover { background: #273449; }
.badge {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 4px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  flex-shrink: 0;
}
.badge.pass { background: #14532d; color: #4ade80; }
.badge.fail { background: #7f1d1d; color: #fca5a5; }
.test-title {
  flex: 1;
  font-size: 14px;
  color: #f1f5f9;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.test-meta { color: #64748b; font-size: 12px; flex-shrink: 0; }
.test-body {
  display: none;
  padding: 16px;
  border-top: 1px solid #334155;
  background: #0f172a;
}
.test.open .test-body { display: block; }
.test.open .test-header .arrow { transform: rotate(90deg); }
.arrow {
  color: #64748b;
  transition: transform 0.15s;
  font-family: monospace;
  flex-shrink: 0;
}
.section { margin-bottom: 16px; }
.section-label {
  color: #94a3b8;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
pre {
  background: #020617;
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  font-family: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, monospace;
  font-size: 12.5px;
  line-height: 1.6;
  color: #cbd5e1;
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-height: 400px;
  overflow-y: auto;
}
.assert {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 10px 12px;
  margin-bottom: 8px;
  font-size: 13px;
}
.assert .assert-head {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 4px;
}
.assert .assert-type {
  background: #334155;
  color: #cbd5e1;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: monospace;
}
.assert .assert-value { color: #94a3b8; font-size: 12px; }
.assert .reason { color: #cbd5e1; font-size: 12px; margin-top: 6px; font-style: italic; }
.footer { margin-top: 32px; text-align: center; color: #64748b; font-size: 12px; }
"""


def extract_text(output) -> str:
    """Promptfoo output may be a string or a structured object."""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return output.get("output") or output.get("text") or json.dumps(output, indent=2)
    return str(output)


def render_assert(a: dict) -> str:
    passed = a.get("pass", False)
    badge = "pass" if passed else "fail"
    badge_text = "PASS" if passed else "FAIL"
    a_type = html.escape(str(a.get("type", "?")))
    a_value = a.get("value", "")
    if isinstance(a_value, list):
        a_value = ", ".join(str(v) for v in a_value)
    a_value = html.escape(str(a_value))[:300]
    reason = html.escape(str(a.get("reason", "") or ""))
    reason_html = f'<div class="reason">{reason}</div>' if reason else ""
    return f"""
      <div class="assert">
        <div class="assert-head">
          <span class="badge {badge}">{badge_text}</span>
          <span class="assert-type">{a_type}</span>
          <span class="assert-value">{a_value}</span>
        </div>
        {reason_html}
      </div>
    """


def render_test(idx: int, result: dict) -> str:
    passed = result.get("success", False)
    status = "pass" if passed else "fail"
    badge_text = "PASS" if passed else "FAIL"

    vars_dict = result.get("vars") or result.get("testCase", {}).get("vars", {}) or {}
    user_prompt = vars_dict.get("prompt", "(no prompt)")
    title = html.escape(str(user_prompt))

    response = extract_text(result.get("response", {}).get("output", result.get("response", "")))
    response_html = html.escape(response)

    grading = result.get("gradingResult") or {}
    component_results = grading.get("componentResults") or []
    asserts_html = (
        "".join(render_assert(a) for a in component_results)
        or '<div class="assert"><span class="assert-value">(no assertion details)</span></div>'
    )

    tokens = result.get("response", {}).get("tokenUsage", {}) or {}
    total_tokens = tokens.get("total", 0)
    latency = result.get("latencyMs", 0)
    meta = f"{total_tokens} tok · {latency} ms" if total_tokens or latency else ""

    return f"""
    <div class="test {status}" data-status="{status}">
      <div class="test-header" onclick="this.parentElement.classList.toggle('open')">
        <span class="arrow">▶</span>
        <span class="badge {status}">{badge_text}</span>
        <span class="test-title">#{idx} · {title}</span>
        <span class="test-meta">{meta}</span>
      </div>
      <div class="test-body">
        <div class="section">
          <div class="section-label">User prompt</div>
          <pre>{html.escape(str(user_prompt))}</pre>
        </div>
        <div class="section">
          <div class="section-label">Response</div>
          <pre>{response_html}</pre>
        </div>
        <div class="section">
          <div class="section-label">Assertions</div>
          {asserts_html}
        </div>
      </div>
    </div>
    """


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 1

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    data = json.loads(in_path.read_text(encoding="utf-8"))
    results_root = data.get("results") or data
    results = results_root.get("results") or results_root.get("table", {}).get("body") or []

    if not results and "results" in data and isinstance(data["results"], dict):
        results = data["results"].get("results", [])

    total = len(results)
    passed = sum(1 for r in results if r.get("success"))
    failed = total - passed
    rate = f"{(passed / total * 100):.0f}%" if total else "—"

    provider_ids = sorted({(r.get("provider") or {}).get("id", "?") for r in results})
    provider_label = ", ".join(provider_ids) if provider_ids else "—"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tests_html = "\n".join(render_test(i + 1, r) for i, r in enumerate(results))

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>perf-test-sync · promptfoo report</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>perf-test-sync · promptfoo eval</h1>
  <div class="subtitle">Generated {ts} · Provider: {html.escape(provider_label)}</div>

  <div class="summary">
    <div class="card"><div class="label">Total</div><div class="value">{total}</div></div>
    <div class="card pass"><div class="label">Passed</div><div class="value">{passed}</div></div>
    <div class="card fail"><div class="label">Failed</div><div class="value">{failed}</div></div>
    <div class="card rate"><div class="label">Pass rate</div><div class="value">{rate}</div></div>
  </div>

  <div class="filters">
    <button class="filter-btn active" data-filter="all">All ({total})</button>
    <button class="filter-btn" data-filter="fail">Failed ({failed})</button>
    <button class="filter-btn" data-filter="pass">Passed ({passed})</button>
  </div>

  <div id="tests">
    {tests_html}
  </div>

  <div class="footer">Rendered by render_report.py</div>
</div>

<script>
document.querySelectorAll('.filter-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const f = btn.dataset.filter;
    document.querySelectorAll('.test').forEach(t => {{
      t.style.display = (f === 'all' || t.dataset.status === f) ? '' : 'none';
    }});
  }});
}});
document.querySelectorAll('.test.fail').forEach(t => t.classList.add('open'));
</script>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Report written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
