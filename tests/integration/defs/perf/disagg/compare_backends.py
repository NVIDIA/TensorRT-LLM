#!/usr/bin/env python3
"""Compare performance test results between different backends (UCX vs NIXL)."""

import argparse
import os
import re
import sys

import pandas as pd


def extract_backend(test_name):
    """Extract backend type from test_name.

    New format: ccb-NIXL or ccb-UCX or ccb-DEFAULT
    Example: disagg_perf_deepseek-r1-fp4_1k1k_ctx2_gen1_dep16_bs128_eplb288_mtp3_ccb-NIXL

    Note: "DEFAULT" is a special marker that represents the default backend
    """
    match = re.search(r"ccb-(\w+)", test_name)
    return match.group(1) if match else None


def extract_base_case_name(test_name):
    """Extract standardized case name (remove backend information).

    Replace ccb-XXX with ccb-BACKEND to create a common base name for grouping.
    Example: disagg_perf_deepseek-r1-fp4_1k1k_..._ccb-NIXL -> disagg_perf_deepseek-r1-fp4_1k1k_..._ccb-BACKEND
    """
    # Replace ccb-XXX with ccb-BACKEND to normalize
    pattern = r"ccb-\w+"
    base_case = re.sub(pattern, "ccb-BACKEND", test_name)

    return base_case


def compare_backends(csv_path, threshold=5.0, default_backend="NIXL"):
    """Compare performance metrics between DEFAULT backend and UCX.

    Only focus on cases where DEFAULT is slower than UCX.

    Args:
        csv_path: CSV file path
        threshold: Performance difference threshold (percentage)
        default_backend: DEFAULT backend name (currently NIXL, may switch in the future)
                        Cases marked as "ccb-DEFAULT" will be treated as this backend

    Returns:
        DataFrame: Comparison results
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit(0)

    # Read CSV file
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit(0)

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print(f"No data found in CSV file: {csv_path}")
        sys.exit(0)

    # Filter only keep tests related to disagg_perf
    # Determine from test_name field (new format: disagg_perf_{model_name}_...)
    df = df[df["test_name"].str.contains("disagg_perf_", na=False)]
    if len(df) == 0:
        print(f"No disagg_perf tests found in CSV file: {csv_path}")
        sys.exit(0)

    # Extract backend and standardized case name
    df["backend"] = df["test_name"].apply(extract_backend)
    df["base_case_name"] = df["test_name"].apply(extract_base_case_name)

    # Normalize "DEFAULT" backend to the actual default_backend value
    # This allows cases marked as "ccb-DEFAULT" to be treated as the default backend
    df["backend"] = df["backend"].apply(
        lambda x: default_backend if x and x.upper() == "DEFAULT" else x
    )

    # Group by base_case_name and metric_type
    grouped = df.groupby(["base_case_name", "metric_type"])

    results = []
    comparison_pairs = 0
    single_backend_skipped = 0

    for (base_case, metric_type), group in grouped:
        # Get DEFAULT backend and UCX data
        default_data = group[group["backend"] == default_backend]
        ucx_data = group[group["backend"] == "UCX"]

        # Skip if both have no data (this case may not exist)
        if len(default_data) == 0 and len(ucx_data) == 0:
            continue

        # Skip single-backend cases (only has one backend, not a comparison pair)
        # This happens when a test case only runs on one backend
        if len(default_data) == 0 or len(ucx_data) == 0:
            single_backend_skipped += 1
            continue

        # This is a valid comparison pair
        comparison_pairs += 1

        # Extract values and original test names
        default_value = default_data["perf_metric"].values[0] if len(default_data) > 0 else None
        default_original_name = (
            default_data["network_name"].values[0] if len(default_data) > 0 else None
        )

        ucx_value = ucx_data["perf_metric"].values[0] if len(ucx_data) > 0 else None
        ucx_original_name = ucx_data["network_name"].values[0] if len(ucx_data) > 0 else None

        # Determine status
        status = "Pass"
        diff_pct = None
        regression_pct = None

        # If one has value and the other has no value, mark as Fail (test run failed)
        if default_value is None or ucx_value is None:
            status = "Fail"
        elif ucx_value != 0:
            # Calculate performance difference percentage
            # For TTFT and E2EL metrics, smaller is better
            # regression_pct > 0 means DEFAULT is slower than UCX (performance degradation)
            # regression_pct < 0 means DEFAULT is faster than UCX (performance improvement)
            regression_pct = ((default_value - ucx_value) / ucx_value) * 100
            diff_pct = abs(regression_pct)

            # Only fail if DEFAULT is slower than UCX and exceeds threshold
            if regression_pct > threshold:
                status = "Fail"
            else:
                status = "Pass"
        else:
            # UCX value is 0 is an abnormal case
            if default_value != 0:
                status = "Fail"

        # Use original network names, or "N/A" if data doesn't exist
        test_case_name_default = default_original_name if default_original_name else "N/A"
        test_case_name_ucx = ucx_original_name if ucx_original_name else "N/A"

        results.append(
            {
                "test_case_name_default": test_case_name_default,
                "test_case_name_ucx": test_case_name_ucx,
                "metric_type": metric_type,
                "default_value": default_value,
                "ucx_value": ucx_value,
                "diff_pct": diff_pct,
                "regression_pct": regression_pct,
                "status": status,
            }
        )

    # Print statistics
    print("\n=== Backend Comparison Statistics ===")
    print(f"Default backend: {default_backend}")
    print(f"Comparison pairs: {comparison_pairs}")
    print(f"Single-backend cases (skipped): {single_backend_skipped}")
    print("=" * 37)

    # If no comparison pairs found, exit with success
    if comparison_pairs == 0:
        print("\nInfo: No backend comparison pairs found in disagg_perf tests")
        print("All cases are single-backend only, no comparison needed")
        sys.exit(0)

    # Convert to DataFrame
    result_df = pd.DataFrame(results)

    return result_df


def generate_html_report(result_df, threshold, default_backend, output_path):
    """Generate HTML format comparison report."""
    # Statistics
    total = len(result_df)
    failed = len(result_df[result_df["status"] == "Fail"])
    passed = total - failed

    # HTML template
    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backend Comparison Report - DEFAULT vs UCX</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
        }}
        .summary-box {{
            flex: 1;
            margin: 0 10px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            color: white;
        }}
        .summary-box.total {{
            background-color: #2196F3;
        }}
        .summary-box.pass {{
            background-color: #4CAF50;
        }}
        .summary-box.fail {{
            background-color: #f44336;
        }}
        .summary-box h2 {{
            margin: 0;
            font-size: 36px;
        }}
        .summary-box p {{
            margin: 5px 0 0 0;
            font-size: 14px;
        }}
        .info {{
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .warning-box {{
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 14px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-pass {{
            background-color: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .status-fail {{
            background-color: #f44336;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .metric-type {{
            background-color: #2196F3;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .regression {{
            color: #f44336;
            font-weight: bold;
        }}
        .improvement {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .neutral {{
            color: #666;
        }}
        .test-name {{
            font-family: monospace;
            font-size: 12px;
            word-break: break-all;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Backend Comparison Report: DEFAULT ({default_backend}) vs UCX</h1>

        <div class="info">
            <strong>DEFAULT Backend:</strong> {default_backend}
            <br>
            <strong>Comparison Backend:</strong> UCX
            <br>
            <strong>Threshold:</strong> {threshold}%
            <br>
            <strong>Description:</strong> Only focus on cases where DEFAULT is slower than UCX.
            Mark as Fail if performance degradation exceeds threshold
        </div>

        <div class="warning-box">
            <strong>⚠️ Attention:</strong>
            <ul style="margin: 5px 0;">
                <li>✅ <strong>Pass</strong>: DEFAULT is similar to or better than UCX</li>
                <li>❌ <strong>Fail</strong>: DEFAULT is slower than UCX{threshold}%（Performance degradation）</li>
                <li>📊 Positive value means DEFAULT is slower than UCX, negative value means
                DEFAULT is faster than UCX</li>
            </ul>
        </div>

        <div class="summary">
            <div class="summary-box total">
                <h2>{total}</h2>
                <p>Total tests</p>
            </div>
            <div class="summary-box pass">
                <h2>{passed}</h2>
                <p>Pass</p>
            </div>
            <div class="summary-box fail">
                <h2>{failed}</h2>
                <p>Performance degradation</p>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th style="width: 22%;">DEFAULT ({default_backend})</th>
                    <th style="width: 22%;">UCX</th>
                    <th style="width: 10%;">Metric type</th>
                    <th style="width: 10%;">DEFAULT value</th>
                    <th style="width: 10%;">UCX value</th>
                    <th style="width: 8%;">Difference (%)</th>
                    <th style="width: 10%;">Regression/Improvement (%)</th>
                    <th style="width: 8%;">Status</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>

        <div class="footer">
            <p>Generated time: {timestamp}</p>
        </div>
    </div>
</body>
</html>
"""

    # Generate table rows
    table_rows = []
    for _, row in result_df.iterrows():
        status_class = "status-pass" if row["status"] == "Pass" else "status-fail"

        # Format difference percentage
        if pd.notna(row["diff_pct"]):
            diff_str = f"{row['diff_pct']:.2f}%"
        else:
            diff_str = "N/A"

        # Format regression/improvement percentage
        if pd.notna(row["regression_pct"]):
            if row["regression_pct"] > 0:
                # Positive value: DEFAULT is slower than UCX (regression)
                regression_str = f"+{row['regression_pct']:.2f}%"
                regression_class = "regression"
            else:
                # Negative value: DEFAULT is faster than UCX (improvement)
                regression_str = f"{row['regression_pct']:.2f}%"
                regression_class = "improvement"
        else:
            regression_str = "N/A"
            regression_class = "neutral"

        # Format values
        default_val = f"{row['default_value']:.2f}" if pd.notna(row["default_value"]) else "N/A"
        ucx_val = f"{row['ucx_value']:.2f}" if pd.notna(row["ucx_value"]) else "N/A"

        row_html = f"""
                <tr>
                    <td class="test-name">{row["test_case_name_default"]}</td>
                    <td class="test-name">{row["test_case_name_ucx"]}</td>
                    <td><span class="metric-type">{row["metric_type"]}</span></td>
                    <td>{default_val}</td>
                    <td>{ucx_val}</td>
                    <td>{diff_str}</td>
                    <td class="{regression_class}">{regression_str}</td>
                    <td><span class="{status_class}">{row["status"]}</span></td>
                </tr>
        """
        table_rows.append(row_html)

    # Fill template
    from datetime import datetime

    html_content = html_template.format(
        default_backend=default_backend,
        threshold=threshold,
        total=total,
        passed=passed,
        failed=failed,
        table_rows="".join(table_rows),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare performance test results between DEFAULT backend and UCX, "
            "only focus on cases where DEFAULT is slower than UCX"
        )
    )
    parser.add_argument(
        "--csv-path", type=str, required=True, help="Performance test results CSV file path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help=(
            "Performance difference threshold (percentage), default 5.0%. "
            "Only mark as Fail if DEFAULT is slower than UCX exceeds this threshold"
        ),
    )
    parser.add_argument(
        "--default-backend",
        type=str,
        default="NIXL",
        help="DEFAULT backend name (default NIXL, may switch to other backend in the future)",
    )
    parser.add_argument(
        "--output", type=str, help="Output CSV file path (optional, default print to stdout)"
    )
    parser.add_argument("--html", type=str, help="Output HTML report file path (optional)")

    args = parser.parse_args()

    # Execute comparison
    result_df = compare_backends(args.csv_path, args.threshold, args.default_backend)

    # Output CSV results
    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"CSV results saved to: {args.output}")
    else:
        print(result_df.to_string(index=False))

    # Output HTML report
    if args.html:
        generate_html_report(result_df, args.threshold, args.default_backend, args.html)
        print(f"HTML report saved to: {args.html}")

    # Statistics
    total = len(result_df)
    failed = len(result_df[result_df["status"] == "Fail"])
    passed = total - failed

    print("\n============= Statistics =============")
    print(f"DEFAULT Backend: {args.default_backend}")
    print("Comparison Backend: UCX")
    print(f"Threshold: {args.threshold}%")
    print("-----------------------------------")
    print(f"Total: {total}")
    print(f"Pass: {passed} (DEFAULT performance normal)")
    print(f"Fail: {failed} (DEFAULT is slower than UCX exceeds {args.threshold}%)")
    print("===================================\n")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
