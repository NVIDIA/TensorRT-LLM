import os

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
# Autotuner warmup intentionally probes tactics that can OOM and logs e.g.
# "[Autotuner] Single-pair run failed ... CUDA out of memory ...". Only that
# specific marker+OOM combination is benign: an autotuner-prefixed line with
# any other error keyword (e.g. "[Autotuner] RuntimeError: ...") is a real
# failure and must still be reported.
AUTOTUNER_MARKER = "[Autotuner]"
AUTOTUNER_BENIGN_TEXTS = [
    "out of memory",
]


def is_benign_line(line: str) -> bool:
    """True for lines expected during a HEALTHY run despite an error keyword."""
    return AUTOTUNER_MARKER in line and any(text in line for text in AUTOTUNER_BENIGN_TEXTS)


SLURM_LOG_TAIL_LINES = 200  # Number of lines to print from slurm job logs
ERROR_CONTEXT_LINES = 100  # Number of lines to print before and after error line


def check_error(file_path: str) -> list[tuple[int, str]]:
    if not os.path.exists(file_path):
        return []

    error_lines = []
    with open(file_path, "r", errors="replace") as f:
        for line_idx, line in enumerate(f, start=1):
            if is_benign_line(line):
                continue
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
            continue

        all_lines = None
        error_lines = []
        try:
            with open(log_file, "r", errors="replace") as f:
                all_lines = f.readlines()
            # Scan the buffered lines (iterating the exhausted file handle
            # after readlines() would yield nothing).
            for line_idx, line in enumerate(all_lines, start=1):
                if is_benign_line(line):
                    continue
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
        if all_lines is not None:
            # ALWAYS include the end of the log, even when a keyword matched
            # above: the first keyword hit may be benign noise while the
            # actual fatal error sits at the end of the file.
            tail_content = "".join(all_lines[-tail_lines:]) if all_lines else "(empty)"
            messages.append(f"--- {log_file} [last {tail_lines} lines] ---")
            messages.append(tail_content)

    raise RuntimeError("\n".join(messages))
