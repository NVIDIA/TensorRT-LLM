#!/usr/bin/env python3
"""Check `serve_check.py --find` against a store built for the purpose.

  test_discovery.py

A real store cannot assert anything: what it holds changes under us, and the
layouts that matter -- a checkpoint nested a level down, a checkpoint holding
another -- are the ones a given store may happen not to have.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SERVE_CHECK = Path(__file__).resolve().parent.parent / "scripts" / "serve_check.py"

# Every layout --find has to handle, and what each is here to prove.
LAYOUT = {
    "Flat": "the level the walk used to stop at",
    "Nested-FP8/release-0000": "a manifest one level down, under a bare parent",
    "Both": "a checkpoint that also holds one",
    "Both/child": "...and the one it holds",
    "Multi": "per-precision weight files beside the manifest",
}
WEIGHT_FILES = {"Multi": ["bf16.safetensors", "quant-fp4.safetensors"]}


def build(root: Path) -> None:
    for rel in LAYOUT:
        directory = root / rel
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline"}))
        for name in WEIGHT_FILES.get(rel, []):
            # Over LFS_STUB, so status reads "ok" rather than "LFS-STUB".
            (directory / name).write_bytes(b"\0" * 20_480)
    plain = root / "NotVisualGen"
    plain.mkdir(parents=True, exist_ok=True)
    (plain / "config.json").write_text("{}")
    deep = root / "TooDeep/a/b/c"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline"}))


def run(root: Path, *words) -> tuple[int, str]:
    done = subprocess.run(
        [sys.executable, str(SERVE_CHECK), "--find", *words],
        capture_output=True,
        text=True,
        env={**os.environ, "LLM_MODELS_ROOT": str(root)},
    )
    return done.returncode, done.stdout


def main() -> int:
    problems = []
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        build(root)

        code, out = run(root)
        for rel, why in LAYOUT.items():
            if f"{root / rel} " not in out and not out.rstrip().endswith(str(root / rel)):
                problems.append(f"--find lists neither {rel} nor what it stands for: {why}")
        if "NotVisualGen" in out:
            problems.append("--find listed a directory carrying no model_index.json")
        if "TooDeep" in out:
            problems.append("--find descended past MAX_DEPTH")
        if code != 0:
            problems.append(f"--find with no words exited {code}")

        # 'nested' appears only in the parent segment, so a leaf-name match misses it.
        code, out = run(root, "nested", "fp8")
        if "Nested-FP8/release-0000" not in out:
            problems.append("--find nested fp8 misses a match carried by the parent segment")

        code, out = run(root, "multi", "fp4")
        if "quant-fp4.safetensors" not in out:
            problems.append("--find multi fp4 does not address the per-precision file")

        code, out = run(root, "notvisualgen")
        if code == 0:
            problems.append("--find exited 0 with nothing to report")

    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"{len(LAYOUT)} layouts, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
