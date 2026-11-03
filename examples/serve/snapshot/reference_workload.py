# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the unchanged pinned upstream workload with its benchmark-token hook."""

import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    """Bind a source-memory token, then run the original upstream application."""
    config = json.loads(Path("/snapshot-control/experiment.json").read_text())
    sys.argv = ["/reference/app.py", json.dumps({"model": "/model"})]
    runpy.run_path(
        "/reference/app.py",
        run_name="__main__",
        init_globals={"benchmark_token": config["template"]},
    )


if __name__ == "__main__":
    main()
