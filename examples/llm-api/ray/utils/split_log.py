#!/usr/bin/env python3
"""
Script to analyze ray.log file and count "rank: x" occurrences per PID.
Reads from ray.log, splits by lines, finds "rank: x" patterns, and creates
a dictionary with PID as key and count of each rank occurrence as value.
Can also split the log into separate files by PID based on rank.
"""

import argparse
import re
from collections import defaultdict
from typing import Dict, Optional, Tuple


def extract_pid_and_rank(line: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract PID and rank from a log line.

    Args:
        line: A log line from ray.log

    Returns:
        Tuple of (pid, rank) if both found, otherwise (None, None)
    """
    # Extract PID from the beginning of the line: (RayWorkerWrapper pid=XXXXX) or (pid=XXXXX)
    pid_match = re.search(r'\((?:RayWorkerWrapper\s+)?pid=(\d+)\)', line)
    if not pid_match:
        return None, None

    pid = pid_match.group(1)

    # Extract rank from "rank: X" pattern
    rank_match = re.search(r'rank:\s*(\d+)', line)
    if not rank_match:
        return pid, None

    rank = int(rank_match.group(1))
    return pid, rank


def get_pid_from_line(line: str) -> Optional[str]:
    """
    Extract just the PID from a log line (used for all lines from a PID, not just rank lines).

    Args:
        line: A log line from ray.log

    Returns:
        PID if found, otherwise None
    """
    pid_match = re.search(r'\((?:RayWorkerWrapper\s+)?pid=(\d+)\)', line)
    return pid_match.group(1) if pid_match else None


def analyze_ray_log(log_file_path: str) -> Dict[str, Dict[int, int]]:
    """
    Analyze ray.log file and count "rank: x" occurrences per PID.

    Args:
        log_file_path: Path to the ray.log file

    Returns:
        Dictionary with PID as key and dict of {rank: count} as value
    """
    result = defaultdict(lambda: defaultdict(int))

    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                pid, rank = extract_pid_and_rank(line.strip())

                if pid is not None and rank is not None:
                    result[pid][rank] += 1

    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}

    # Convert defaultdict to regular dict for cleaner output
    return {pid: dict(ranks) for pid, ranks in result.items()}


def get_highest_rank_per_pid(
        rank_counts: Dict[str, Dict[int, int]]) -> Dict[str, int]:
    """
    Get the highest rank (by occurrence count) for each PID.

    Args:
        rank_counts: Dictionary with PID as key and rank counts as value

    Returns:
        Dictionary with PID as key and highest rank as value
    """
    pid_to_highest_rank = {}

    for pid, ranks in rank_counts.items():
        # Find the rank with the highest count
        highest_rank = max(ranks.keys(), key=lambda rank: ranks[rank])
        pid_to_highest_rank[pid] = highest_rank

    return pid_to_highest_rank


def split_log_by_pid(log_file_path: str, output_dir: str = '.') -> None:
    """
    Split the ray.log file by PID into separate files named rank{x}.log.

    Args:
        log_file_path: Path to the ray.log file
        output_dir: Directory to save the split log files
    """
    print("Analyzing log to determine PID-to-rank mapping...")

    # First, analyze the log to get rank counts per PID
    rank_counts = analyze_ray_log(log_file_path)

    if not rank_counts:
        print("No rank data found. Cannot split log files.")
        return

    # Get the highest rank for each PID
    pid_to_rank = get_highest_rank_per_pid(rank_counts)

    print(f"PID to rank mapping:")
    for pid, rank in sorted(pid_to_rank.items()):
        print(f"  PID {pid} -> rank {rank}")

    # Create output file handles
    output_files = {}
    rank_to_filename = {}

    for pid, rank in pid_to_rank.items():
        filename = f"rank{rank}.log"
        rank_to_filename[rank] = filename
        if rank not in output_files:
            output_files[rank] = open(f"{output_dir}/{filename}",
                                      'w',
                                      encoding='utf-8')

    print(f"\nSplitting log into files:")
    for rank in sorted(output_files.keys()):
        print(f"  rank{rank}.log")

    try:
        line_count = 0
        lines_per_file = defaultdict(int)

        with open(log_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
                pid = get_pid_from_line(line.strip())

                if pid and pid in pid_to_rank:
                    rank = pid_to_rank[pid]
                    output_files[rank].write(line)
                    lines_per_file[rank] += 1

        print(f"\nProcessed {line_count} lines:")
        for rank in sorted(lines_per_file.keys()):
            print(f"  rank{rank}.log: {lines_per_file[rank]} lines")

    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")
    finally:
        # Close all output files
        for file_handle in output_files.values():
            file_handle.close()

    print(f"\nLog splitting completed!")


def print_results(results: Dict[str, Dict[int, int]]) -> None:
    """
    Print the results in a formatted way.

    Args:
        results: Dictionary with PID as key and rank counts as value
    """
    if not results:
        print("No results found.")
        return

    print("=" * 50)
    print("RANK ANALYSIS RESULTS")
    print("=" * 50)

    for pid in sorted(results.keys()):
        print(f"\nPID: {pid}")
        print("-" * 20)

        rank_counts = results[pid]
        total_count = sum(rank_counts.values())

        for rank in sorted(rank_counts.keys()):
            count = rank_counts[rank]
            print(f"  rank: {rank} -> {count} occurrences")

        print(f"  Total: {total_count} occurrences")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description=
        "Analyze ray.log file and count 'rank: x' occurrences per PID. "
        "Can also split the log into separate files by PID based on rank.")

    parser.add_argument('log_file', help='Path to the ray.log file to analyze')

    parser.add_argument(
        '--split',
        action='store_true',
        help='Split the log into separate files by PID (named rank{x}.log)')

    parser.add_argument(
        '--output-dir',
        default='.',
        help='Directory to save the split log files (default: current directory)'
    )

    args = parser.parse_args()

    print(f"Analyzing {args.log_file}...")

    # Analyze the log file
    results = analyze_ray_log(args.log_file)

    # Print results
    print_results(results)

    # Split the log by PID if requested
    if args.split:
        print("\n" + "=" * 50)
        print("SPLITTING LOG BY PID")
        print("=" * 50)
        split_log_by_pid(args.log_file, args.output_dir)

    # Also return the results for programmatic use
    return results


if __name__ == "__main__":
    results = main()
