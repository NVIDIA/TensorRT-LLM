#!/usr/bin/env python3
"""Test Report Generator for TensorRT-LLM test runs.

Parses pytest output and generates a formatted report.

Usage:
    generate_report.py --output-file <path_to_captured_output> \
                       --exit-code <int> \
                       [--format <markdown|html|json>] \
                       [--report-file <output_path>]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_pytest_output(text: str) -> dict:
    """Extract structured data from raw pytest output."""
    result = {
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "warnings": 0,
        "duration_s": None,
        "failed_tests": [],
        "error_details": [],
        "coverage_pct": None,
        "summary_line": "",
    }

    # Summary line: "5 passed, 2 failed, 1 error in 3.45s"
    summary_match = re.search(
        r"(\d+) passed(?:,\s*(\d+) failed)?(?:,\s*(\d+) error(?:s)?)?"
        r"(?:,\s*(\d+) warning(?:s)?)?(?:,\s*(\d+) skipped)?.*?in\s+([\d.]+)s",
        text,
    )
    if summary_match:
        result["passed"] = int(summary_match.group(1) or 0)
        result["failed"] = int(summary_match.group(2) or 0)
        result["error"] = int(summary_match.group(3) or 0)
        result["warnings"] = int(summary_match.group(4) or 0)
        result["skipped"] = int(summary_match.group(5) or 0)
        result["duration_s"] = float(summary_match.group(6))
        result["summary_line"] = summary_match.group(0)

    # Failed test names from short summary section: "FAILED tests/foo/test_bar.py::TestClass::test_method"
    raw_matches = re.findall(r"FAILED\s+(tests/[\w/.:_-]+)", text)
    result["failed_tests"] = list(dict.fromkeys(raw_matches))  # deduplicate, preserve order

    # Error details: capture FAILED sections
    error_sections = re.findall(r"_{5,}\s+([\w/.:_-]+)\s+_{5,}(.*?)(?=_{5,}|\Z)", text, re.DOTALL)
    for name, detail in error_sections:
        snippet = detail.strip()[:500]
        result["error_details"].append({"test": name.strip(), "detail": snippet})

    # Coverage: "TOTAL  1234  56  95%"
    cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", text)
    if cov_match:
        result["coverage_pct"] = int(cov_match.group(1))

    return result


def parse_benchmark_metrics(text: str) -> list[dict]:
    """Extract performance metrics from benchmark output."""
    metrics = []

    # Throughput patterns: "Throughput: 1234.5 samples/sec"
    for match in re.finditer(r"(?i)(throughput|samples/sec|tokens/s|it/s)[:\s]+([\d.]+)", text):
        metrics.append({"metric": match.group(1), "value": float(match.group(2))})

    # Latency patterns: "Latency: 15.3 ms"
    for match in re.finditer(r"(?i)(latency)[:\s]+([\d.]+)\s*(ms|s)?", text):
        unit = match.group(3) or "ms"
        metrics.append({"metric": f"Latency ({unit})", "value": float(match.group(2))})

    # GPU utilization: "GPU Util: 85%"
    for match in re.finditer(r"(?i)(gpu[\s_]?util(?:ization)?)[:\s]+([\d.]+)\s*%?", text):
        metrics.append({"metric": "GPU Utilization (%)", "value": float(match.group(2))})

    # SOL / MFU
    for match in re.finditer(r"(?i)(SOL|MFU)[:\s]+([\d.]+)\s*%?", text):
        metrics.append({"metric": f"{match.group(1)} (%)", "value": float(match.group(2))})

    return metrics


def parse_feature_matrix(text: str) -> list[dict]:
    """Extract feature support results from test output."""
    features = []

    # Pattern: "PASSED tests/tools/test_trtllm_modeling.py::TestClass::test_feature_name"
    for match in re.finditer(
        r"(PASSED|FAILED|ERROR)\s+(tests/tools/test_trtllm_modeling\.py::[:\w]+)", text
    ):
        status = match.group(1)
        test_id = match.group(2)
        # Convert test ID to readable feature name
        parts = test_id.split("::")
        feature = parts[-1].replace("test_", "").replace("_", " ").title()
        features.append({"feature": feature, "test_id": test_id, "status": status})

    return features


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_markdown(test_type: str, parsed: dict, exit_code: int, timestamp: str) -> str:
    total = parsed["passed"] + parsed["failed"] + parsed["error"]
    status = "PASS" if exit_code == 0 else "FAIL"
    status_icon = "✅" if exit_code == 0 else "❌"

    lines = [
        f"# Test Report — {test_type.replace('-', ' ').title()}",
        "",
        f"**Status:** {status_icon} {status}  ",
        f"**Generated:** {timestamp}  ",
        f"**Duration:** {parsed['duration_s']}s" if parsed["duration_s"] else "",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total  | {total} |",
        f"| Passed | {parsed['passed']} |",
        f"| Failed | {parsed['failed']} |",
        f"| Errors | {parsed['error']} |",
        f"| Skipped | {parsed['skipped']} |",
    ]

    if parsed.get("coverage_pct") is not None:
        lines += [f"| Coverage | {parsed['coverage_pct']}% |"]

    if parsed["failed_tests"]:
        lines += [
            "",
            "## Failed Tests",
            "",
        ]
        for t in parsed["failed_tests"]:
            lines.append(f"- `{t}`")

    if parsed["error_details"]:
        lines += ["", "## Error Details", ""]
        for err in parsed["error_details"][:5]:  # cap at 5
            lines += [
                f"### `{err['test']}`",
                "",
                "```",
                err["detail"],
                "```",
                "",
            ]

    if test_type == "benchmark" and parsed.get("benchmark_metrics"):
        lines += ["", "## Performance Metrics", "", "| Metric | Value |", "|--------|-------|"]
        for m in parsed["benchmark_metrics"]:
            lines.append(f"| {m['metric']} | {m['value']} |")

    if test_type == "feature-matrix" and parsed.get("features"):
        lines += [
            "",
            "## Feature Support Matrix",
            "",
            "| Feature | Status |",
            "|---------|--------|",
        ]
        for f in parsed["features"]:
            icon = "✅" if f["status"] == "PASSED" else "❌"
            lines.append(f"| {f['feature']} | {icon} {f['status']} |")

    return "\n".join(filter(lambda x: x is not None, lines))


def format_html(test_type: str, parsed: dict, exit_code: int, timestamp: str) -> str:
    md = format_markdown(test_type, parsed, exit_code, timestamp)
    # Simple HTML wrap — convert markdown tables and code blocks minimally
    md_escaped = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Convert ``` blocks
    md_escaped = re.sub(
        r"```\n(.*?)```", r"<pre><code>\1</code></pre>", md_escaped, flags=re.DOTALL
    )
    # Convert headers
    for i in range(3, 0, -1):
        md_escaped = re.sub(
            r"^#{" + str(i) + r"}\s+(.+)$", rf"<h{i}>\1</h{i}>", md_escaped, flags=re.MULTILINE
        )
    md_escaped = md_escaped.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")

    status_color = "#2e7d32" if exit_code == 0 else "#c62828"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Test Report — {test_type}</title>
  <style>
    body {{ font-family: monospace; padding: 2em; max-width: 900px; margin: auto; }}
    h1 {{ color: {status_color}; }}
    pre {{ background: #f5f5f5; padding: 1em; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #ccc; padding: 0.5em; text-align: left; }}
    th {{ background: #eee; }}
  </style>
</head>
<body>
{md_escaped}
</body>
</html>"""


def format_json(test_type: str, parsed: dict, exit_code: int, timestamp: str) -> str:
    output = {
        "test_type": test_type,
        "status": "pass" if exit_code == 0 else "fail",
        "exit_code": exit_code,
        "timestamp": timestamp,
        "summary": {
            "passed": parsed["passed"],
            "failed": parsed["failed"],
            "error": parsed["error"],
            "skipped": parsed["skipped"],
            "duration_s": parsed["duration_s"],
            "coverage_pct": parsed.get("coverage_pct"),
        },
        "failed_tests": parsed["failed_tests"],
    }
    if test_type == "benchmark":
        output["benchmark_metrics"] = parsed.get("benchmark_metrics", [])
    if test_type == "feature-matrix":
        output["features"] = parsed.get("features", [])
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate test report from pytest output")
    parser.add_argument(
        "--test-type", required=True, choices=["functional", "benchmark", "feature-matrix"]
    )
    parser.add_argument("--output-file", required=True, help="Path to captured test output file")
    parser.add_argument(
        "--exit-code", required=True, type=int, help="Exit code from test run (0=pass)"
    )
    parser.add_argument("--format", default="markdown", choices=["markdown", "html", "json"])
    parser.add_argument("--report-file", default=None, help="Optional path to write report to")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    if not output_path.exists():
        print(f"Error: output file not found: {output_path}", file=sys.stderr)
        sys.exit(1)

    raw = output_path.read_text(errors="replace")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parsed = parse_pytest_output(raw)

    # Enrich with type-specific parsing
    if args.test_type == "benchmark":
        parsed["benchmark_metrics"] = parse_benchmark_metrics(raw)
    elif args.test_type == "feature-matrix":
        parsed["features"] = parse_feature_matrix(raw)

    # Format
    if args.format == "markdown":
        report = format_markdown(args.test_type, parsed, args.exit_code, timestamp)
    elif args.format == "html":
        report = format_html(args.test_type, parsed, args.exit_code, timestamp)
    else:
        report = format_json(args.test_type, parsed, args.exit_code, timestamp)

    # Output
    if args.report_file:
        Path(args.report_file).write_text(report)
        print(f"Report written to: {args.report_file}")
    else:
        print(report)


if __name__ == "__main__":
    main()
