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
    "PMI2_Init failed to intialize",
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
