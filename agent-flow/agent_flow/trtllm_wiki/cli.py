# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-wiki``: build a pack of a TensorRT-LLM checkout and ask it questions.

Example::

    trtllm-wiki --repo ~/TensorRT-LLM feature reuse

``--pack DIR`` reads the pack in DIR and refreshes it from the checkout it
was built from (or ``--repo``) when that checkout moved; with only ``--repo``
the pack is kept under ``$XDG_CACHE_HOME/trtllm-wiki`` per commit. A lookup
never blocks or polls: it builds the pack once (seconds) when the checkout
moved, then reads files. Output is text, or JSON with ``--json``. Exit codes:
0 an answer, 1 no pack to read, 2 usage error, 4 nothing matched (for ``ci``:
no list runs the test; for ``validate``: a field is not declared).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from . import pack as pack_mod
from . import render
from .query import Wiki

NOT_FOUND = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtllm-wiki",
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=render.USAGE,
    )
    parser.add_argument("--repo", type=Path, help="TensorRT-LLM checkout to read")
    parser.add_argument("--pack", type=Path, help="pack directory (built from --repo when stale)")
    parser.add_argument("--json", action="store_true", help="print JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="(re)build the pack and print its manifest").add_argument(
        "--force", action="store_true"
    )
    sub.add_parser("index", help="print the pack's index.md")
    for name, helptext in (
        ("field", "a field path, or text in field paths"),
        ("feature", "a feature name (see index)"),
        ("family", "a family directory or a routed architecture"),
        ("op", "text in an operation's name, module or wrapped calls"),
        ("ci", "a test path (repo-relative or as in test lists) or node id"),
        ("accuracy", "an accuracy task (gsm8k) or text in a model key"),
        ("env", "text in a TLLM_/TRTLLM_ variable name"),
    ):
        sub.add_parser(name, help=helptext).add_argument(
            "query", nargs="?" if name == "env" else None, default=""
        )
    recipe = sub.add_parser("recipe", help="nearest shipped serving configs")
    recipe.add_argument("text", nargs="?", default="", help="words in the model, path or name")
    recipe.add_argument("--gpu", default="")
    for flag in ("gpus", "isl", "osl", "concurrency", "limit"):
        recipe.add_argument(f"--{flag}", type=int, default=5 if flag == "limit" else None)
    sub.add_parser(
        "validate", help="fields an options YAML names that the schema does not"
    ).add_argument("options", type=Path)
    return parser


def open_wiki(repo: Path | None, pack: Path | None, *, force: bool = False) -> Wiki:
    """The pack for ``repo``/``pack``, built or refreshed as needed.

    A pack named alone is refreshed from the checkout its manifest names,
    when that checkout is still there.
    """
    if repo is None and pack is not None:
        manifest = pack_mod.read_manifest(pack)
        built_from = Path(manifest["repo"]) if manifest and manifest.get("repo") else None
        repo = built_from if built_from is not None and built_from.is_dir() else None
    if repo is not None:
        out = pack or pack_mod.default_cache_dir(repo)
        pack_mod.ensure(repo, out, force=force)
        return Wiki(out)
    if pack is None:
        raise SystemExit("trtllm-wiki: give --repo, --pack, or both")
    if pack_mod.read_manifest(pack) is None:
        print(
            f"trtllm-wiki: no pack at {pack}; build one with `trtllm-wiki --repo <checkout> "
            f"--pack {pack} build`",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return Wiki(pack)


def _answer(wiki: Wiki, args: argparse.Namespace) -> tuple[Any, str]:
    """``(data, text)`` for the command; ``data`` empty means nothing matched."""
    command, query = args.command, getattr(args, "query", "")
    if command == "index":
        return True, (wiki.path / "index.md").read_text(encoding="utf-8")
    if command == "field":
        found = wiki.fields(query)
        return found, "\n".join(render.render_field(f) for f in found)
    if command == "feature":
        page = wiki.feature(query)
        return page, render.render_feature(page) if page else ""
    if command == "family":
        family = wiki.family(query)
        return family, render.render_family(family) if family else ""
    if command == "op":
        found = wiki.ops(query)
        return found, "\n".join(render.render_op(op) for op in found)
    if command == "ci":
        answer = wiki.ci(query)
        return answer, render.render_ci(answer)
    if command == "accuracy":
        task = wiki.accuracy_task(query)
        refs = wiki.references(query) if task is None else {}
        data = {"task": task, "references": refs} if task or refs else None
        return data, json.dumps(data, indent=1) if data else ""
    if command == "env":
        found = wiki.env_vars(query)
        return found, "\n".join(f"{v['name']}: {', '.join(v['reads'][:3])}" for v in found)
    if command == "recipe":
        found = wiki.recipes(
            text=args.text,
            gpu=args.gpu,
            gpus=args.gpus,
            isl=args.isl,
            osl=args.osl,
            concurrency=args.concurrency,
            limit=args.limit,
        )
        return found, "\n".join(render.render_recipe(r) for r in found)
    if command == "validate":
        options = yaml.safe_load(args.options.read_text(encoding="utf-8")) or {}
        unknown = wiki.unknown_fields(options)
        text = "\n".join(f"not a field: {p}" for p in unknown) or "every field is declared"
        return {"unknown": unknown}, text
    raise SystemExit(f"trtllm-wiki: unknown command {command}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        if args.repo is None:
            raise SystemExit("trtllm-wiki build: --repo is required")
        out = args.pack or pack_mod.default_cache_dir(args.repo)
        manifest, _ = pack_mod.ensure(args.repo, out, force=args.force)
        print(json.dumps({**manifest, "pack": str(out)}, indent=1))
        return 0
    wiki = open_wiki(args.repo, args.pack)
    data, text = _answer(wiki, args)
    if (args.command == "validate" and data["unknown"]) or (
        args.command == "ci" and not data["entries"]
    ):
        print(json.dumps(data, indent=1) if args.json else text)
        return NOT_FOUND
    if not data:
        print(f"nothing matched {getattr(args, 'query', '') or args.command!r}", file=sys.stderr)
        return NOT_FOUND
    print(json.dumps(data, indent=1, default=str) if args.json else text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
