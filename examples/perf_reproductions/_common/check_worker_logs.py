# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check worker logs against an entry's ``verify.log_patterns``.

Each log file is matched against the named pattern; a file passes when the pattern occurs at least once. Use it after
the workers have started to confirm that the expected switch profile took effect on every role.

Usage:
    python3 check_worker_logs.py <entry>/source.yaml decode=/logs/decode_*.log prefill_tep=/logs/prefill_*.log
"""

import glob
import re
import sys
from pathlib import Path

import yaml


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    patterns = (yaml.safe_load(Path(sys.argv[1]).read_text()).get("verify") or {}).get(
        "log_patterns"
    ) or {}
    failed = 0
    for spec in sys.argv[2:]:
        name, _, pattern_glob = spec.partition("=")
        if name not in patterns:
            print(f"unknown pattern {name!r}; available: {', '.join(sorted(patterns))}")
            return 2
        regex = re.compile(patterns[name])
        files = sorted(glob.glob(pattern_glob))
        if not files:
            print(f"FAIL {name}: no files match {pattern_glob}")
            failed += 1
        for path in files:
            text = Path(path).read_text(errors="replace")
            ok = regex.search(text) is not None
            failed += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {name}: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
