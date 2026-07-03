import os
import shutil
import sys
import time
from typing import List

ERROR_KEYWORDS = [
    "RuntimeError",
    "out of memory",
    "ValueError",
    "FileNotFoundError",
    "ConnectionRefusedError",
    "ClientConnectorError",
    "CancelledError",
    "TimeoutError",
    "PMI2_Init failed to initialize",
    "OSError",
]
SLURM_LOG_TAIL_LINES = 200  # Number of lines to print from slurm job logs
ERROR_CONTEXT_LINES = 100  # Number of lines to print before and after error line


def check_error(file_path: str) -> list[tuple[int, str]]:
    if not os.path.exists(file_path):
        return []

    error_lines = []
    with open(file_path, "r", errors="replace") as f:
        for line_idx, line in enumerate(f, start=1):
            for keyword in ERROR_KEYWORDS:
                if keyword in line:
                    error_lines.append((line_idx, line.strip()))
                    break  # Only add line once even if multiple keywords match

    return error_lines


def report_error(
    error_msg: str | Exception, log_files: list[str], tail_lines: int = SLURM_LOG_TAIL_LINES
) -> None:
    # Convert Exception to string if needed
    if isinstance(error_msg, Exception):
        error_msg_str = str(error_msg)
    else:
        error_msg_str = error_msg

    messages = [error_msg_str]

    for log_file in log_files:
        if not os.path.exists(log_file):
            messages.append(f"Failed to read {log_file}: Path doesn't exist")

        all_lines = None
        error_lines = []
        try:
            with open(log_file, "r", errors="replace") as f:
                all_lines = f.readlines()
                for line_idx, line in enumerate(f, start=1):
                    for keyword in ERROR_KEYWORDS:
                        if keyword in line:
                            error_lines.append((line_idx, line.strip()))
                            break
        except Exception as e:
            all_lines = None
            error_lines = []
            messages.append(f"Failed to read {log_file}: {e}")

        if error_lines:
            error_lines_str = ", ".join(
                [f"Error line {line_idx}: {line_str}" for line_idx, line_str in error_lines]
            )
            messages.append(error_lines_str)
            # Find the first error line number for context
            first_idx = min(line_idx for line_idx, _ in error_lines)
            if all_lines is not None:
                # Use all_lines for error context
                start_idx = max(0, first_idx - ERROR_CONTEXT_LINES - 1)
                end_idx = min(len(all_lines), first_idx + ERROR_CONTEXT_LINES)
                context_lines = all_lines[start_idx:end_idx]
                messages.append("".join(context_lines))
        else:
            tail_content = "".join(all_lines[-tail_lines:]) if all_lines else "(empty)"
            messages.append(f"--- {log_file} [last {tail_lines} lines] ---")
            messages.append(tail_content)

    raise RuntimeError("\n".join(messages))


# --- Disagg BENCHMARK sibling-log dump -------------------------------------
#
# In CI, only the BENCHMARK srun step's stdout is displayed. To make debugging
# convenient, BENCHMARK's test_e2e dumps every sibling log (CTX/GEN/DISAGG_SERVER
# pytest srun outputs and the nested trtllm-serve subprocess logs) into its own
# stdout at teardown.
#
# Coordination is a hybrid of pytest_done marker files (primary) and log-file
# size stability (fallback for crashed workers that never wrote a marker), both
# bounded by a max timeout so BENCHMARK never hangs on a stuck worker. A short
# flush sleep after detection covers the last srun `&> file` redirect
# finalization.

SIBLING_LOG_WAIT_TIMEOUT_S = 120
SIBLING_LOG_POLL_INTERVAL_S = 3
SIBLING_LOG_STABILITY_POLLS = 3
SIBLING_LOG_FLUSH_SLEEP_S = 3


def pytest_done_marker_path(
    test_output_dir: str, disagg_serving_type: str, server_idx: int
) -> str:
    return os.path.join(
        test_output_dir, f"pytest_done.{disagg_serving_type}.{server_idx}.txt"
    )


def expected_sibling_marker_paths(
    test_output_dir: str, num_ctx_servers: int, num_gen_servers: int, server_idx: int
) -> List[str]:
    markers = []
    for i in range(num_ctx_servers):
        markers.append(pytest_done_marker_path(test_output_dir, f"CTX_{i}", server_idx))
    for i in range(num_gen_servers):
        markers.append(pytest_done_marker_path(test_output_dir, f"GEN_{i}", server_idx))
    markers.append(pytest_done_marker_path(test_output_dir, "DISAGG_SERVER", server_idx))
    return markers


def write_pytest_done_marker(
    test_output_dir: str, disagg_serving_type: str, server_idx: int
) -> None:
    """Signal to BENCHMARK that this worker's pytest is finishing.

    Best-effort: any exception is logged and swallowed since a failed marker
    write must not fail the test — BENCHMARK will fall back to size stability.
    """
    try:
        os.makedirs(test_output_dir, exist_ok=True)
        marker_path = pytest_done_marker_path(test_output_dir, disagg_serving_type, server_idx)
        with open(marker_path, "w") as f:
            f.write("done")
    except Exception as e:
        print(f"Failed to write pytest_done marker: {e}")


def wait_for_sibling_completion(
    marker_paths: List[str],
    log_paths: List[str],
    timeout_s: int = SIBLING_LOG_WAIT_TIMEOUT_S,
    poll_interval_s: int = SIBLING_LOG_POLL_INTERVAL_S,
    stability_polls: int = SIBLING_LOG_STABILITY_POLLS,
) -> str:
    """Wait until sibling workers are done, or their logs stop growing, or timeout.

    Returns the reason for exit: "markers", "stable", or "timeout". A crashed
    worker that never wrote a marker still lets us exit via size stability, so
    BENCHMARK never blocks the full timeout on a worker failure.
    """
    start = time.time()
    prev_sizes = {p: -1 for p in log_paths}
    stable_counts = {p: 0 for p in log_paths}

    while time.time() - start < timeout_s:
        if all(os.path.exists(m) for m in marker_paths):
            print(
                f"All worker pytest_done markers found after "
                f"{time.time() - start:.1f}s."
            )
            return "markers"

        for p in log_paths:
            try:
                size = os.path.getsize(p) if os.path.exists(p) else 0
            except OSError:
                size = 0
            if size == prev_sizes[p]:
                stable_counts[p] += 1
            else:
                stable_counts[p] = 0
            prev_sizes[p] = size

        # A log is "stable" once we've seen `stability_polls` consecutive same-size
        # polls. Missing files count as stable so a worker that crashed before
        # writing anything doesn't block. Require at least one non-empty log to
        # avoid returning immediately on the first tick before any worker started.
        all_stable = all(stable_counts[p] >= stability_polls for p in log_paths)
        any_nonempty = any(prev_sizes[p] > 0 for p in log_paths)
        if all_stable and any_nonempty:
            print(
                f"All sibling logs stable after {time.time() - start:.1f}s "
                f"(marker fallback)."
            )
            return "stable"

        time.sleep(poll_interval_s)

    missing = [m for m in marker_paths if not os.path.exists(m)]
    print(
        f"Timed out after {timeout_s}s waiting for sibling completion. "
        f"Missing markers: {missing}. Dumping logs anyway."
    )
    return "timeout"


def dump_sibling_logs(log_paths: List[str]) -> None:
    """Print full content of each sibling log so it lands in BENCHMARK's stdout."""
    separator = "=" * 80
    print(f"\n{separator}")
    print("=== Sibling log dump (from BENCHMARK pytest for CI visibility) ===")
    print(f"{separator}")
    for log_path in log_paths:
        print(f"\n----- BEGIN {log_path} -----")
        if not os.path.exists(log_path):
            print("(file does not exist)")
        else:
            try:
                with open(log_path, "r", errors="replace") as f:
                    shutil.copyfileobj(f, sys.stdout)
                # Ensure the footer starts on its own line even if the log
                # didn't end with a newline.
                sys.stdout.write("\n")
            except Exception as e:
                print(f"(failed to read: {e})")
        print(f"----- END {log_path} -----")
    print(f"{separator}\n")
    sys.stdout.flush()
