#!/usr/bin/env python3
"""Resolve a VisualGen model to a checkpoint path, and derive its warmup config.

  serve_check.py --find "wan 14b fp4"
      Match a loose description against every checkpoint under LLM_MODELS_ROOT and
      print the paths to pass to `trtllm-serve`, the pipeline each dispatches to,
      and whether the weights are actually materialized. No args lists everything.

  serve_check.py --fields [BLOCK]
      Every field of VisualGenArgs -- type, default, description -- read from the
      pydantic model itself. Optionally narrow to one block (attention, parallel,
      compilation, ...). Needs an environment where `import tensorrt_llm` works.

  serve_check.py --warmup-from <workload.yaml>
      Read the shape a bench workload will request (width/height/num_frames) and
      print the server-side `compilation_config` block that warms exactly it.

Warmup defaults are Python properties that compute their value, so a static reader
guesses wrong on exactly the models that need one -- derive, do not look up.
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(os.environ["TRTLLM_REPO"]) if os.environ.get("TRTLLM_REPO") else None
MODELS = REPO / "tensorrt_llm/_torch/visual_gen/models" if REPO else None
_WEIGHTS_ENV = os.environ.get("LLM_MODELS_ROOT")
WEIGHTS = Path(_WEIGHTS_ENV) if _WEIGHTS_ENV else None
LFS_STUB = 10_240  # a git-lfs pointer is ~134 bytes; real shards are GB-scale
# Checkpoints sit one or two levels into a store. Each level past that descends
# into component directories holding every weight shard, which on a network
# store costs 40s at depth 4 and two minutes unbounded, for no further hit.
MAX_DEPTH = 3


# AutoPipeline._detect_from_checkpoint's substring fallback, in its order, used when
# a checkpoint's _class_name is not itself a registry key.
FALLBACK = [
    ("ImageToVideo|I2V", "Wan", "WanImageToVideoPipeline"),
    (None, "Wan", "WanPipeline"),
    (None, "Flux2", "Flux2Pipeline"),
    (None, "Flux", "FluxPipeline"),
    (None, "QwenImageLayered", "QwenImageLayeredPipeline"),
    (None, "QwenImage", "QwenImagePipeline"),
    (None, "Cosmos3", "Cosmos3OmniMoTPipeline"),
    (None, "HunyuanVideo15", "HunyuanVideo15Pipeline"),
]


def registry_keys():
    """The names register_pipeline() is called with -- the keys AutoPipeline looks up."""
    if MODELS is None or not MODELS.is_dir():
        return None
    out = set()
    for f in sorted(MODELS.rglob("pipeline_*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            for d in getattr(node, "decorator_list", []):
                if (
                    isinstance(d, ast.Call)
                    and getattr(d.func, "id", "") == "register_pipeline"
                    and d.args
                    and isinstance(d.args[0], ast.Constant)
                ):
                    out.add(d.args[0].value)
    return out


def dispatch(class_name, keys):
    """Which pipeline AutoPipeline will instantiate for this _class_name."""
    if keys is None:
        return f"{class_name} (?)"
    if class_name in keys:
        return class_name
    for extra, needle, target in FALLBACK:
        if needle in class_name and (extra is None or re.search(extra, class_name)):
            return target
    return f"{class_name} (UNKNOWN)"


def materialized(d):
    """'ok', 'LFS-STUB' (cloned without lfs pull), or 'no-weights'."""
    sh = list(d.rglob("*.safetensors")) or list(d.rglob("*.bin"))
    if not sh:
        return "no-weights"
    return "ok" if max(f.stat().st_size for f in sh) > LFS_STUB else "LFS-STUB"


def reference_keyed_pipelines():
    """Pipelines whose warmup key carries the reference's own pixel size.

    Such a pipeline conditions on the reference at its own resolution rather than
    resizing it to the requested shape, so the key gains dimensions
    ``compilation_config`` has no field for and no config warms that request.
    Read from the source: a hardcoded list here would age past the next model.
    """
    if MODELS is None:
        return []
    found = set()
    for path in sorted(MODELS.glob("*/pipeline_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
                if fn.name == "request_warmup_cache_key" and re.search(
                    r"reference|condition_image", ast.dump(fn)
                ):
                    found.add(cls.name)
    return sorted(found)


def manifests(root):
    """Every ``model_index.json`` within ``MAX_DEPTH`` of the store, in path order."""
    base = str(root).rstrip(os.sep).count(os.sep)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "model_index.json" in filenames:
            found.append(Path(dirpath) / "model_index.json")
        if dirpath.count(os.sep) - base >= MAX_DEPTH:
            dirnames[:] = []
    return sorted(found)


def find(tokens):
    """[(path, pipeline, status)] for VisualGen checkpoints matching every token.

    Discovery walks the store for ``model_index.json`` and takes the directory
    holding it as the checkpoint, so a store that nests one a level down
    (``Cosmos3-Nano-FP8/<release>/``) is reachable. A directory that holds one
    and also contains others is itself a checkpoint, and all of them are listed.

    Matching is against the path relative to the store, since a nested leaf need
    not repeat what its parent already says. A weight file inside a checkpoint
    matches on that path plus its own name, which is how per-precision files
    (LTX-2 ships fp4/fp8/bf16 side by side in one dir) become addressable.
    """
    if WEIGHTS is None:
        sys.exit("set LLM_MODELS_ROOT to the checkpoint store to search")
    if not WEIGHTS.is_dir():
        sys.exit(f"{WEIGHTS} not readable -- check LLM_MODELS_ROOT")
    keys = registry_keys()
    if keys is None:
        print(
            "warning: TRTLLM_REPO unset or wrong -- the pipeline column shows the "
            "checkpoint's raw _class_name marked '(?)', not the class that will "
            "actually handle it.",
            file=sys.stderr,
        )
    hits = []
    # A diffusers checkpoint carries model_index.json; LLM checkpoints do not.
    # That is both the VisualGen filter and, via _class_name, the dispatch key
    # for checkpoints whose id was never registered.
    for mi in manifests(WEIGHTS):
        d = mi.parent
        try:
            cls = json.loads(mi.read_text()).get("_class_name", "?")
        except (ValueError, OSError):
            cls = "?"
        pipe = dispatch(cls, keys)
        rel = d.relative_to(WEIGHTS).as_posix().lower()
        if all(t in rel for t in tokens):
            hits.append((d, pipe, materialized(d)))
            continue
        for f in sorted(d.glob("*.safetensors")):
            if all(t in f"{rel} {f.name}".lower() for t in tokens):
                hits.append((f, pipe, "ok" if f.stat().st_size > LFS_STUB else "LFS-STUB"))
    return hits


def dump_pipeline_config(checkpoint):
    """Print the ``pipeline_config`` keys one checkpoint accepts, with their defaults.

    The block is a ``Dict[str, Any]`` on ``VisualGenArgs``, so the pydantic walk that
    prints every other block has nothing to descend into: the keys belong to the
    architecture, and only the model resolves them. It is strict at load, so a key
    that is not here raises.
    """
    from tensorrt_llm import VisualGen

    keys = VisualGen.pipeline_config(str(checkpoint))
    print(f"pipeline_config for {checkpoint}")
    if not keys:
        print("    (this architecture declares none)")
        return
    width = max(len(k) for k in keys)
    for key, default in keys.items():
        print(f"    {key:{width}s}  {default!r}")


def dump_fields(only):
    """Print VisualGenArgs' fields from the live pydantic model, one block at a time."""
    from pydantic import BaseModel
    from tensorrt_llm.visual_gen.args import VisualGenArgs  # isort: skip

    def rows(model, prefix=""):
        for name, f in model.model_fields.items():
            ann = f.annotation
            inner = next(
                (
                    x
                    for x in ([ann] + list(getattr(ann, "__args__", [])))
                    if isinstance(x, type) and issubclass(x, BaseModel)
                ),
                None,
            )
            default = f.get_default(call_default_factory=False)
            yield prefix + name, ann, default, (f.description or "").replace("\n", " ")
            if inner is not None:
                yield from rows(inner, prefix + name + ".")

    def short(x):
        s = str(x).replace("typing.", "").replace("tensorrt_llm.visual_gen.args.", "")
        return s if len(s) <= 46 else s[:43] + "..."

    for name, ann, default, doc in rows(VisualGenArgs):
        if only and not name.startswith(only):
            continue
        print(f"{name:44s} {short(ann):48s} {short(default)}")
        if doc:
            print(f"    {doc[:150]}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--find", nargs="*", metavar="WORD", help="loose model description, e.g. --find wan 14b fp4"
    )
    g.add_argument(
        "--fields",
        nargs="?",
        const="",
        metavar="BLOCK",
        help="dump VisualGenArgs fields, optionally one block",
    )
    ap.add_argument(
        "--model",
        metavar="CHECKPOINT",
        help="checkpoint --fields pipeline_config reads its per-architecture keys from",
    )
    g.add_argument(
        "--warmup-from",
        type=Path,
        metavar="WORKLOAD",
        dest="warmup_from",
        help="bench workload to derive the server compilation_config from",
    )
    a = ap.parse_args()

    if a.fields is not None:
        if a.fields == "pipeline_config":
            if not a.model:
                sys.exit(
                    "pipeline_config keys are per-architecture, so this block needs a "
                    "checkpoint: --fields pipeline_config --model <path from --find>"
                )
            dump_pipeline_config(a.model)
            return
        dump_fields(a.fields)
        return

    if a.find is not None:
        toks = [t for t in re.split(r"[\s_,-]+", " ".join(a.find).lower()) if t]
        hits = find(toks)
        if not hits:
            sys.exit(
                f"no checkpoint under {WEIGHTS} matches {toks}. "
                f"Run --find with no words to see everything."
            )
        for p, pipe, st in hits:
            print(f"{str(p):72s} {pipe:30s} {st}")
        print(
            f"\n{len(hits)} VisualGen checkpoint(s) under {WEIGHTS}. Pass a path "
            f"above to trtllm-serve.\nLFS-STUB means the clone holds pointer files, "
            f"not weights."
        )
        return

    rec = yaml.safe_load(a.warmup_from.read_text(encoding="utf-8"))
    common = rec.get("common_params") or {}
    res, frames = set(), set()
    for item in rec.get("requests") or [{}]:
        m = {**common, **(item or {})}
        if m.get("height") is None or m.get("width") is None:
            sys.exit(
                f"{a.warmup_from.name}: width/height unset -- the size is derived "
                f"from the reference at inference time. Pin both to the shape it "
                f"resolves to, or serve with no compilation_config and let the "
                f"pipeline warm its own default."
            )
        res.add((int(m["height"]), int(m["width"])))
        frames.add(int(m["num_frames"]) if m.get("num_frames") else 1)

    # On stderr: stdout is redirected into the config, where a warning would sit
    # unread in the file it is warning about.
    if any(k.endswith("_reference") for item in rec.get("requests") or [] for k in (item or {})):
        keyed = reference_keyed_pipelines()
        if keyed:
            print(
                f"note: {a.warmup_from.name} names a reference. On "
                f"{', '.join(keyed)} the warmup key carries the reference's own pixel "
                f"size, which this config cannot state, so every request there reports "
                f"'was not warmed up' and the first one compiles inside the measured "
                f"latency.",
                file=sys.stderr,
            )

    print(
        f"# from {a.warmup_from.name}. resolutions is (height, width) -- reversed from "
        f"the workload.\n# Both keys are required: config replaces the pipeline default "
        f"per axis, so\n# resolutions alone leaves num_frames at the default and the "
        f"product misses you."
    )
    print(
        yaml.safe_dump(
            {
                "compilation_config": {
                    "resolutions": [list(r) for r in sorted(res)],
                    "num_frames": sorted(frames),
                }
            },
            default_flow_style=None,
            sort_keys=False,
        ).rstrip()
    )
    if len(res) * len(frames) > max(len(res), len(frames)):
        print(
            f"# NOTE {len(res)}x{len(frames)} shapes: the Cartesian product over-covers "
            f"a mixed-shape workload."
        )


if __name__ == "__main__":
    main()
