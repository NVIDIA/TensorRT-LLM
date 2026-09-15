# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assert the reference ladder's independence, statically.

Two properties the whole of Goal 1.1 rests on, neither of which is visible by
reading the files casually once they are a thousand lines long:

  * Neither file imports ``tensorrt_llm``. A reference that reached through the
    package under test would not be an independent reference.
  * ``refmods.py`` imports nothing from the checkpoint's own ``inference/``
    tree. A reference sharing a helper with the implementation it checks cannot
    separate "my reference is wrong" from "my port is wrong", which is the one
    thing this rung exists to do. ``refcapture.py`` *does* import it -- that is
    the harness, and driving the native forward is its job.

Exits nonzero on violation. Reads the source with ``ast``; imports nothing.
"""

from __future__ import annotations

import ast
import pathlib
import sys

#: The checkpoint's own modules. ``refmods`` must reach none of them.
REFERENCE_TREE = {"model", "kernel", "engram", "vision", "generate", "convert", "image_processor"}


def imports_of(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    names |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    return names


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    bad: list[str] = []
    for name in ("refmods.py", "refcapture.py"):
        path = root / name
        if not path.exists():
            bad.append(f"{name}: missing")
            continue
        reached = imports_of(path)
        print(f"{name} imports: {' '.join(sorted(reached))}", flush=True)
        if "tensorrt_llm" in reached:
            bad.append(f"{name}: imports tensorrt_llm")
        if name == "refmods.py":
            shared = reached & REFERENCE_TREE
            if shared:
                bad.append(f"refmods.py: shares the implementation's own modules {sorted(shared)}")
    if bad:
        print("INDEPENDENCE VIOLATED: " + "; ".join(bad), flush=True)
        return 1
    print("independence holds: no tensorrt_llm anywhere, no reference-tree helper in refmods", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
