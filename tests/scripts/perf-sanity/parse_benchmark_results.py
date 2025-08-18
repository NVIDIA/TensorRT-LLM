#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml


def extract_config_from_log_content(log_file):
    """
    Extract configuration from log file content using "Completed benchmark with Configuration:" pattern
    """
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "Completed benchmark with Configuration:" in line:
                    # Extract values using regex patterns
                    model_label_match = re.search(r'model_label=([^,]+)', line)
                    gpus_match = re.search(r'GPUs=(\d+)', line)
                    tp_match = re.search(r'TP=(\d+)', line)
                    ep_match = re.search(r'EP=(\d+)', line)
                    attn_backend_match = re.search(r'attn_backend=([^,]+)',
                                                   line)
                    moe_backend_match = re.search(r'moe_backend=([^,]+)', line)
                    enable_attention_dp_match = re.search(
                        r'enable_attention_dp=([^,]+)', line)
                    free_gpu_mem_fraction_match = re.search(
                        r'free_gpu_mem_fraction=([^,]+)', line)
                    max_batch_size_match = re.search(r'max_batch_size=(\d+)',
                                                     line)
                    isl_match = re.search(r'ISL=(\d+)', line)
                    osl_match = re.search(r'OSL=(\d+)', line)
                    max_num_tokens_match = re.search(r'max_num_tokens=(\d+)',
                                                     line)
                    moe_max_num_tokens_match = re.search(
                        r'moe_max_num_tokens=([^,]+)', line)
                    concurrency_match = re.search(r'Concurrency=(\d+)', line)

                    # Extract values, use empty string if not found
                    model_label = model_label_match.group(
                        1) if model_label_match else ""
                    gpus = int(gpus_match.group(1)) if gpus_match else ""
                    tp = int(tp_match.group(1)) if tp_match else ""
                    ep = int(ep_match.group(1)) if ep_match else ""
                    attn_backend = attn_backend_match.group(
                        1) if attn_backend_match else ""
                    moe_backend = moe_backend_match.group(
                        1) if moe_backend_match else ""
                    enable_attention_dp = enable_attention_dp_match.group(
                        1) if enable_attention_dp_match else ""
                    free_gpu_mem_fraction = float(
                        free_gpu_mem_fraction_match.group(
                            1)) if free_gpu_mem_fraction_match else ""
                    max_batch_size = int(max_batch_size_match.group(
                        1)) if max_batch_size_match else ""
                    isl = int(isl_match.group(1)) if isl_match else ""
                    osl = int(osl_match.group(1)) if osl_match else ""
                    max_num_tokens = int(max_num_tokens_match.group(
                        1)) if max_num_tokens_match else ""
                    moe_max_num_tokens_str = moe_max_num_tokens_match.group(
                        1) if moe_max_num_tokens_match else ""
                    concurrency = int(
                        concurrency_match.group(1)) if concurrency_match else ""

                    # Handle moe_max_num_tokens (could be "N/A", empty, or a number)
                    moe_max_num_tokens = ""
                    if moe_max_num_tokens_str and moe_max_num_tokens_str != "N/A":
                        try:
                            moe_max_num_tokens = int(moe_max_num_tokens_str)
                        except ValueError:
                            moe_max_num_tokens = ""
                    elif not moe_max_num_tokens_str:
                        moe_max_num_tokens = ""

                    # Handle enable_attention_dp (convert string to boolean)
                    enable_attention_dp_bool = ""
                    if enable_attention_dp:
                        enable_attention_dp_bool = enable_attention_dp.lower(
                        ) == "true"

                    # Check if all required fields are present (not empty strings)
                    if (model_label and gpus != "" and tp != "" and ep != ""
                            and attn_backend and free_gpu_mem_fraction != ""
                            and max_batch_size != "" and isl != "" and osl != ""
                            and max_num_tokens != "" and concurrency != ""):
                        return {
                            'model_name': model_label,
                            'gpus': gpus,
                            'tp': tp,
                            'ep': ep,
                            'attn_backend': attn_backend,
                            'moe_backend': moe_backend,
                            'enable_attention_dp': enable_attention_dp_bool,
                            'free_gpu_mem_fraction': free_gpu_mem_fraction,
                            'max_batch_size': max_batch_size,
                            'isl': isl,
                            'osl': osl,
                            'max_num_tokens': max_num_tokens,
                            'moe_max_num_tokens': moe_max_num_tokens,
                            'concurrency': concurrency,
                            'found_in_log': True
                        }
                    else:
                        print(
                            f"Warning: Incomplete configuration in {log_file} - missing required fields"
                        )
                        return None
    except Exception as e:
        print(f"Warning: Could not read {log_file}: {e}")

    return None


def extract_metrics_from_log(log_file):
    """
    Extract Total Token throughput and User throughput from log file
    """
    total_throughput = ""
    user_throughput = ""

    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "Total Token throughput (tok/s):" in line:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        total_throughput = parts[4]
                elif "User throughput (tok/s):" in line:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        user_throughput = parts[3]
    except Exception as e:
        print(f"Warning: Could not read {log_file}: {e}")

    return total_throughput, user_throughput


def generate_all_test_cases(benchmark_config):
    """
    Generate all test cases from benchmark_config.yaml including all concurrency iterations
    """
    all_test_cases = []

    for test_case in benchmark_config['test_cases']:
        base_config = {
            'model_name': test_case['model'],
            'gpus': test_case['gpus'],
            'tp': test_case['tp'],
            'ep': test_case['ep'],
            'attn_backend': test_case['attn_backend'],
            'moe_backend': test_case['moe_backend'],
            'enable_attention_dp': test_case['enable_attention_dp'],
            'free_gpu_mem_fraction': test_case['free_gpu_mem_fraction'],
            'max_batch_size': test_case['max_batch_size'],
            'isl': test_case['isl'],
            'osl': test_case['osl'],
            'max_num_tokens': test_case['max_num_tokens'],
            'moe_max_num_tokens': test_case['moe_max_num_tokens'],
        }

        # Generate a test case for each concurrency iteration
        for concurrency, iterations in test_case['concurrency_iterations']:
            test_case_config = base_config.copy()
            test_case_config['concurrency'] = concurrency
            test_case_config['iterations'] = iterations
            test_case_config['TPS/System'] = ""
            test_case_config['TPS/User'] = ""
            all_test_cases.append(test_case_config)

    return all_test_cases


def match_log_to_test_case(log_config, test_case):
    """
    Check if a log configuration matches a test case configuration
    Returns True if all parameters match exactly
    """
    if not log_config:
        return False

    # Check if all key parameters match exactly
    return (log_config['model_name'] == test_case['model_name']
            and log_config['gpus'] == test_case['gpus']
            and log_config['tp'] == test_case['tp']
            and log_config['ep'] == test_case['ep']
            and log_config['attn_backend'] == test_case['attn_backend']
            and log_config['moe_backend'] == test_case['moe_backend']
            and log_config['enable_attention_dp']
            == test_case['enable_attention_dp']
            and log_config['free_gpu_mem_fraction']
            == test_case['free_gpu_mem_fraction']
            and log_config['max_batch_size'] == test_case['max_batch_size']
            and log_config['isl'] == test_case['isl']
            and log_config['osl'] == test_case['osl']
            and log_config['max_num_tokens'] == test_case['max_num_tokens'] and
            (log_config['moe_max_num_tokens'] == test_case['moe_max_num_tokens']
             or (not log_config['moe_max_num_tokens']
                 and not test_case['moe_max_num_tokens']))
            and log_config['concurrency'] == test_case['concurrency'])


def create_test_case_row(test_case):
    """
    Create a row for a test case with empty performance data
    """
    return {
        'model_name': test_case['model_name'],
        'GPUs': test_case['gpus'],
        'TP': test_case['tp'],
        'EP': test_case['ep'],
        'attn_backend': test_case['attn_backend'],
        'moe_backend': test_case['moe_backend'],
        'enable_attention_dp': test_case['enable_attention_dp'],
        'free_gpu_mem_fraction': test_case['free_gpu_mem_fraction'],
        'max_batch_size': test_case['max_batch_size'],
        'ISL': test_case['isl'],
        'OSL': test_case['osl'],
        'max_num_tokens': test_case['max_num_tokens'],
        'moe_max_num_tokens': test_case['moe_max_num_tokens'],
        'Concurrency': test_case['concurrency'],
        'Iterations': test_case['iterations'],
        'TPS/System': test_case['TPS/System'],
        'TPS/User': test_case['TPS/User'],
    }


def parse_benchmark_results(input_folder, output_csv, config_file):
    """
    Parse benchmark results and generate CSV table
    """
    input_folder = Path(input_folder)
    config_file = Path(config_file)

    # Validate inputs
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return

    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory")
        return

    if not config_file.exists():
        print(f"Error: Config file '{config_file}' does not exist")
        return

    # Load benchmark configuration
    try:
        with open(config_file, 'r') as f:
            benchmark_config = yaml.safe_load(f)
        print(f"Loaded benchmark configuration from: {config_file}")
    except Exception as e:
        print(f"Error: Could not load {config_file}: {e}")
        return

    # Generate all test cases from config
    all_test_cases = generate_all_test_cases(benchmark_config)
    print(f"Generated {len(all_test_cases)} test cases from configuration")

    # Find all serve.*.log files
    log_files = list(input_folder.glob("serve.*.log"))
    print(f"Found {len(log_files)} log files to process")

    # Process each log file
    matched_count = 0
    for log_file in log_files:
        print(f"Processing: {log_file.name}")

        # Extract configuration from log
        log_config = extract_config_from_log_content(log_file)
        if not log_config:
            print(f"  Skipped - could not parse configuration")
            continue

        # Extract performance metrics
        total_throughput, user_throughput = extract_metrics_from_log(log_file)

        # Find matching test case in table
        matched = False
        for test_case in all_test_cases:
            if match_log_to_test_case(log_config, test_case):
                # Update performance data
                test_case['TPS/System'] = total_throughput
                test_case['TPS/User'] = user_throughput
                matched = True
                matched_count += 1
                break

        if not matched:
            print(
                f"  Skipped - no matching test case found for test case {test_case}"
            )

    print(f"Successfully matched {matched_count} log files to test cases")

    table_rows = []
    for test_case in all_test_cases:
        row = create_test_case_row(test_case)
        table_rows.append(row)

    # Add empty rows between different test configurations
    final_table = []
    for i, row in enumerate(table_rows):
        if i > 0:
            prev_row = table_rows[i - 1]
            # Check if any key parameters changed
            if (row['model_name'] != prev_row['model_name']
                    or row['TP'] != prev_row['TP']
                    or row['EP'] != prev_row['EP']
                    or row['moe_backend'] != prev_row['moe_backend']
                    or row['ISL'] != prev_row['ISL']
                    or row['OSL'] != prev_row['OSL']):
                # Add empty row
                empty_row = {key: '' for key in row.keys()}
                final_table.append(empty_row)

        final_table.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(final_table)

    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"\nCSV table saved to: {output_path}")
    print(
        f"Total rows: {len(final_table)} (including {len(final_table) - len(table_rows)} empty separator rows)"
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description=
        "Script to parse benchmark metrics from a specified folder and generate CSV table",
        epilog=
        "Example: python parse_benchmark_results.py ./benchmark_logs results.csv ./benchmark_config.yaml"
    )
    parser.add_argument(
        "--input_folder",
        help="Folder containing benchmark log files (serve.*.log)")
    parser.add_argument("--output_csv",
                        help="Output CSV filename for the results table")
    parser.add_argument("--config_file",
                        help="Path to benchmark_config.yaml file")

    args = parser.parse_args()

    # Validate inputs
    input_folder_path = Path(args.input_folder)
    config_file_path = Path(args.config_file)

    if not input_folder_path.exists():
        print(f"Error: Input folder '{args.input_folder}' not found.")
        sys.exit(1)
    if not input_folder_path.is_dir():
        print(f"Error: '{args.input_folder}' is not a directory.")
        sys.exit(1)
    if not config_file_path.exists():
        print(f"Error: Config file '{args.config_file}' not found.")
        sys.exit(1)

    print(f"Using input folder: {input_folder_path}")
    print(f"Using config file: {config_file_path}")
    print(f"Output will be saved to: {args.output_csv}")
    print()

    parse_benchmark_results(args.input_folder, args.output_csv,
                            args.config_file)


if __name__ == "__main__":
    main()
