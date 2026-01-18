#!/usr/bin/env python3
"""Merge perf regression info from multiple YAML files into an HTML report."""

import argparse
from html import escape as escape_html

import yaml

# Metrics where larger is better
MAXIMIZE_METRICS = [
    "d_seq_throughput",
    "d_token_throughput",
    "d_total_token_throughput",
    "d_user_throughput",
    "d_mean_tpot",
    "d_median_tpot",
    "d_p99_tpot",
]

# Metrics where smaller is better
MINIMIZE_METRICS = [
    "d_mean_ttft",
    "d_median_ttft",
    "d_p99_ttft",
    "d_mean_itl",
    "d_median_itl",
    "d_p99_itl",
    "d_mean_e2el",
    "d_median_e2el",
    "d_p99_e2el",
]


def _get_metric_keys():
    """Get all metric-related keys for filtering config keys."""
    metric_keys = set()
    for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
        metric_suffix = metric[2:]  # Strip "d_" prefix
        metric_keys.add(metric)
        metric_keys.add(f"d_baseline_{metric_suffix}")
        metric_keys.add(f"d_threshold_post_merge_{metric_suffix}")
        metric_keys.add(f"d_threshold_pre_merge_{metric_suffix}")
    return metric_keys


def _get_regression_content(data):
    """Get regression info and config content as a list of lines."""
    lines = []
    if "s_regression_info" in data:
        lines.append("=== Regression Info ===")
        regression_info = data["s_regression_info"]
        for line in regression_info.split(","):
            lines.append(line)

    metric_keys = _get_metric_keys()

    lines.append("")
    lines.append("=== Config ===")
    config_keys = sorted([key for key in data.keys() if key not in metric_keys])
    for key in config_keys:
        if key == "s_regression_info":
            continue
        value = data[key]
        lines.append(f'"{key}": {value}')

    return lines


def merge_regression_data(input_files):
    """Read all yaml file paths and merge regression data."""
    yaml_files = [f.strip() for f in input_files.split(",") if f.strip()]

    regression_dict = {}
    load_failures = 0

    for yaml_file in yaml_files:
        try:
            # Path format: .../{stage_name}/{folder_name}/regression_data.yaml
            path_parts = yaml_file.replace("\\", "/").split("/")
            if len(path_parts) < 3:
                continue

            stage_name = path_parts[-3]
            folder_name = path_parts[-2]

            with open(yaml_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if content is None or not isinstance(content, list):
                    continue

                filtered_data = [
                    d for d in content if isinstance(d, dict) and "s_test_case_name" in d
                ]

                if not filtered_data:
                    continue

                if stage_name not in regression_dict:
                    regression_dict[stage_name] = {}

                if folder_name not in regression_dict[stage_name]:
                    regression_dict[stage_name][folder_name] = []

                regression_dict[stage_name][folder_name].extend(filtered_data)

        except (OSError, yaml.YAMLError, UnicodeDecodeError) as e:
            load_failures += 1
            print(f"Warning: Failed to load {yaml_file}: {e}")
            continue

    # Fail fast if caller provided inputs but none were readable/parseable.
    # (Keeps "no regressions found" working when yaml_files is empty.)
    if yaml_files and not regression_dict and load_failures == len(yaml_files):
        raise RuntimeError("Failed to load any regression YAML inputs; cannot generate report.")

    return regression_dict


def generate_html(regression_dict, output_file):
    """Generate HTML report from regression data."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Perf Regression Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 10px; }}
            .suite-container {{
                margin-bottom: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .suite-header {{
                padding: 10px;
                background: #f8f9fa;
                border-bottom: 1px solid #ddd;
            }}
            .summary {{ margin-bottom: 10px; }}
            .regression {{ color: #d93025; }}
            .testcase {{
                border-left: 4px solid #d93025;
                margin: 5px 0;
                background: white;
            }}
            .test-details {{
                padding: 10px;
                background: #f5f5f5;
                border-radius: 3px;
            }}
            pre {{
                margin: 0;
                white-space: pre-wrap;
                word-wrap: break-word;
                background: #2b2b2b;
                color: #cccccc;
                padding: 10px;
                counter-reset: line;
            }}
            pre + pre {{
                border-top: none;
                padding-top: 0;
            }}
            pre span {{
                display: block;
                position: relative;
                padding-left: 4em;
            }}
            pre span:before {{
                counter-increment: line;
                content: counter(line);
                position: absolute;
                left: 0;
                width: 3em;
                text-align: right;
                color: #666;
                padding-right: 1em;
            }}
            details summary {{
                cursor: pointer;
                outline: none;
            }}
            details[open] summary {{
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h2>Perf Regression Summary</h2>
        {test_suites}
    </body>
    </html>
    """

    all_suites_html = []
    total_tests = 0

    for stage_name in regression_dict:
        folder_dict = regression_dict[stage_name]
        # Count total tests for this stage
        tests_count = sum(len(data_list) for data_list in folder_dict.values())
        total_tests += tests_count

        # Generate summary for the suite
        summary = f"""
            <div class="suite-header">
                <h3>Stage: {escape_html(stage_name)}</h3>
                <p><span class="regression">Regression Tests: {tests_count}</span></p>
            </div>
        """

        # Generate test case details for the suite
        test_cases_html = []

        for folder_name, data_list in folder_dict.items():
            for data in data_list:
                test_case_name = data.get("s_test_case_name", "N/A")
                test_name = f"perf/test_perf_sanity.py::test_e2e[{folder_name}] - {test_case_name}"

                # Get content lines
                content_lines = _get_regression_content(data)
                content_html = "".join(
                    f"<span>{escape_html(line)}</span>" for line in content_lines
                )

                details = f"""
                    <details class="test-details">
                        <summary>{escape_html(test_name)}</summary>
                        <pre>{content_html}</pre>
                    </details>
                """

                test_case_html = f"""
                    <div class="testcase">
                        {details}
                    </div>
                """
                test_cases_html.append(test_case_html)

        # Combine summary and test cases for this suite
        suite_html = f"""
            <div class="suite-container">
                {summary}
                <div class="test-cases">
                    {" ".join(test_cases_html)}
                </div>
            </div>
        """
        all_suites_html.append(suite_html)

    # Generate complete HTML
    html_content = html_template.format(test_suites="\n".join(all_suites_html))

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated HTML report with {total_tests} regression entries: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge perf regression info from YAML files into an HTML report."
    )
    parser.add_argument(
        "--input-files", type=str, required=True, help="Comma-separated list of YAML file paths"
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output HTML file path")
    args = parser.parse_args()

    regression_dict = merge_regression_data(args.input_files)
    generate_html(regression_dict, args.output_file)


if __name__ == "__main__":
    main()
