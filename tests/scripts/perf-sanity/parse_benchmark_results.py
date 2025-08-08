#!/usr/bin/env python3

"""
Script to parse benchmark metrics from a specified folder and generate CSV table
Usage: python parse_benchmark_results.py <folder_name> [output_csv]
  folder_name: Folder containing benchmark log files
  output_csv: Output CSV filename (optional, default: folder_name.csv)
"""

import os
import sys
import re
import glob
from datetime import datetime
from pathlib import Path
import pandas as pd


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
                    attn_backend_match = re.search(r'attn_backend=([^,]+)', line)
                    moe_backend_match = re.search(r'moe_backend=([^,]+)', line)
                    enable_attention_dp_match = re.search(r'enable_attention_dp=([^,]+)', line)
                    free_gpu_mem_fraction_match = re.search(r'free_gpu_mem_fraction=([^,]+)', line)
                    max_batch_size_match = re.search(r'max_batch_size=(\d+)', line)
                    isl_match = re.search(r'ISL=(\d+)', line)
                    osl_match = re.search(r'OSL=(\d+)', line)
                    max_num_tokens_match = re.search(r'max_num_tokens=(\d+)', line)
                    moe_max_num_tokens_match = re.search(r'moe_max_num_tokens=([^,]+)', line)
                    concurrency_match = re.search(r'Concurrency=(\d+)', line)
                    
                    # Extract values, return None if not found
                    model_label = model_label_match.group(1) if model_label_match else None
                    gpus = int(gpus_match.group(1)) if gpus_match else None
                    tp = int(tp_match.group(1)) if tp_match else None
                    ep = int(ep_match.group(1)) if ep_match else None
                    attn_backend = attn_backend_match.group(1) if attn_backend_match else None
                    moe_backend = moe_backend_match.group(1) if moe_backend_match else None
                    enable_attention_dp = enable_attention_dp_match.group(1) if enable_attention_dp_match else None
                    free_gpu_mem_fraction = float(free_gpu_mem_fraction_match.group(1)) if free_gpu_mem_fraction_match else None
                    max_batch_size = int(max_batch_size_match.group(1)) if max_batch_size_match else None
                    isl = int(isl_match.group(1)) if isl_match else None
                    osl = int(osl_match.group(1)) if osl_match else None
                    max_num_tokens = int(max_num_tokens_match.group(1)) if max_num_tokens_match else None
                    moe_max_num_tokens_str = moe_max_num_tokens_match.group(1) if moe_max_num_tokens_match else None
                    concurrency = int(concurrency_match.group(1)) if concurrency_match else None
                    
                    # Handle moe_max_num_tokens (could be "N/A" or a number)
                    moe_max_num_tokens = None
                    if moe_max_num_tokens_str and moe_max_num_tokens_str != "N/A":
                        try:
                            moe_max_num_tokens = int(moe_max_num_tokens_str)
                        except ValueError:
                            moe_max_num_tokens = None
                    
                    # Handle enable_attention_dp (convert string to boolean)
                    enable_attention_dp_bool = None
                    if enable_attention_dp:
                        enable_attention_dp_bool = enable_attention_dp.lower() == "true"
                    
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
    except Exception as e:
        print(f"Warning: Could not read {log_file}: {e}")
    
    return None


def get_default_config():
    """
    Return default configuration values when log content parsing fails
    """
    return {
        'attn_backend': 'TRTLLM',
        'moe_backend': '',
        'enable_attention_dp': False,
        'free_gpu_mem_fraction': 0.9,
        'max_num_tokens': 16384,
        'moe_max_num_tokens': '',
        'gpus': 1,
        'max_batch_size': 512
    }


def extract_metrics_from_log(log_file):
    """
    Extract Total Token throughput and User throughput from log file
    """
    total_throughput = None
    user_throughput = None
    
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


def parse_benchmark_results(folder_path):
    """
    Parse benchmark results from the specified folder and generate CSV table
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        return
    
    # Find all serve.*.log files (only files with prefix="serve.")
    log_files = list(folder_path.glob("serve.*.log"))
    
    if not log_files:
        print(f"Error: No serve.*.log files found in folder '{folder_path}'")
        return
    
    # Extract configuration and metrics for each log file
    results = []
    for log_file in log_files:
        # Try to extract configuration from log content first
        log_config = extract_config_from_log_content(log_file)
        
        if log_config and log_config['found_in_log']:
            # Use configuration from log content
            config = log_config
            print(f"Using configuration from log content for {os.path.basename(log_file)}")

        total_throughput, user_throughput = extract_metrics_from_log(log_file)
        
        results.append({
            'model_name': config['model_name'],
            'GPUs': config['gpus'],
            'TP': config['tp'],
            'EP': config['ep'],
            'attn_backend': config['attn_backend'],
            'moe_backend': config['moe_backend'],
            'enable_attention_dp': config['enable_attention_dp'],
            'free_gpu_mem_fraction': config['free_gpu_mem_fraction'],
            'max_batch_size': config['max_batch_size'],
            'ISL': config['isl'],
            'OSL': config['osl'],
            'max_num_tokens': config['max_num_tokens'],
            'moe_max_num_tokens': config['moe_max_num_tokens'],
            'Concurrency': config['concurrency'],
            'TPS/System': total_throughput,
            'TPS/User': user_throughput,
            'filename': str(log_file),
            'config_source': 'log_content' if config['found_in_log'] else 'filename'
        })
    
    # Sort results by model_name, TP, EP, ISL, OSL, Concurrency
    results.sort(key=lambda x: (x['model_name'], x['TP'], x['EP'], x['ISL'], x['OSL'], x['Concurrency']))
    
    # Add empty rows when key parameters change
    results_with_gaps = []
    for i, result in enumerate(results):
        if i > 0:
            prev_result = results[i-1]
            # Check if any of the specified parameters changed
            if (result['model_name'] != prev_result['model_name'] or
                result['TP'] != prev_result['TP'] or
                result['EP'] != prev_result['EP'] or
                result['moe_backend'] != prev_result['moe_backend'] or
                result['ISL'] != prev_result['ISL'] or
                result['OSL'] != prev_result['OSL']):
                # Add empty row
                empty_row = {key: '' for key in result.keys()}
                results_with_gaps.append(empty_row)
        
        results_with_gaps.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results_with_gaps)
    
    # Remove filename and config_source columns for Excel output
    df_excel = df.drop(['filename', 'config_source'], axis=1)
    
    # Generate CSV filename
    if len(sys.argv) > 2:
        # Use provided output CSV filename
        csv_filename = sys.argv[2]
        # If it's a relative path, make it relative to the output folder's parent
        if not os.path.isabs(csv_filename):
            csv_path = folder_path.parent / csv_filename
        else:
            csv_path = Path(csv_filename)
    else:
        # Use default filename based on folder name
        folder_name = folder_path.name
        csv_filename = f"{folder_name}.csv"
        csv_path = folder_path.parent / csv_filename
    
    # Save to CSV
    df_excel.to_csv(csv_path, index=False)
    
    # Print results to console as well
    end_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    
    print("Performance report - trtllm-serve")
    print("===========================================")
    print(f"Report path: {folder_path.absolute()}")
    print(f"Parsed at: {end_time} on {os.uname().nodename}")
    print(f"CSV file generated: {csv_path}")
    print()
    
    for result in results:
        log_name = os.path.basename(result['filename'])
        config_source = result['config_source']
        print(f"Log: {log_name} (config from: {config_source})")
        print(f"  Configuration: model_name={result['model_name']}, GPUs={result['GPUs']}, TP={result['TP']}, EP={result['EP']}, attn_backend={result['attn_backend']}, moe_backend={result['moe_backend']}, enable_attention_dp={result['enable_attention_dp']}, free_gpu_mem_fraction={result['free_gpu_mem_fraction']}, max_batch_size={result['max_batch_size']}, ISL={result['ISL']}, OSL={result['OSL']}, max_num_tokens={result['max_num_tokens']}, moe_max_num_tokens={result['moe_max_num_tokens']}, Concurrency={result['Concurrency']}")
        
        if result['TPS/System']:
            print(f"  Total Token throughput (tok/s): {result['TPS/System']}")
        
        if result['TPS/User']:
            print(f"  User throughput (tok/s): {result['TPS/User']}")
        
        print()
    
    print("===========================================")
    print(f"CSV table saved to: {csv_path}")
    
    return df_excel


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Error: Please provide a folder name as argument")
        print(f"Usage: {sys.argv[0]} <folder_name> [output_csv]")
        sys.exit(1)
    
    folder_name = sys.argv[1]
    parse_benchmark_results(folder_name)


if __name__ == "__main__":
    main() 