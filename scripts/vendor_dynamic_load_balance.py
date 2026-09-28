# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh the scheduler package and MegaMoE inference closure, replaying recorded patches."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n\n"
)


def revision(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _source_bytes(source: Path, *, header: bool) -> bytes:
    content = source.read_bytes().rstrip(b"\n") + b"\n"
    if header and b"Copyright" not in content[:2000]:
        content = HEADER.encode() + content
    return content


def copy_source(source: Path, target: Path, *, header: bool) -> None:
    content = _source_bytes(source, header=header)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != content:
        target.write_bytes(content)


def _read_downstream_patches(target: Path, source: Path, *, header: bool) -> list[dict]:
    manifest = json.loads((target / "VENDOR_MANIFEST.json").read_text())
    patches = manifest.get("downstream_patches", [])
    for patch in patches:
        patch_path = target / patch["patch"]
        if hashlib.sha256(patch_path.read_bytes()).hexdigest() != patch["patch_sha256"]:
            raise RuntimeError(f"Downstream patch digest mismatch: {patch_path}")
        for file in patch.get("files", [patch]):
            content = _source_bytes(source / file["file"], header=header)
            if hashlib.sha256(content).hexdigest() != file["unpatched_vendored_sha256"]:
                raise RuntimeError(
                    f"Upstream source changed for {file['file']}; reconcile the downstream patch "
                    "and manifest before re-vendoring."
                )
    return patches


def _replace_text(path: Path, transform: Callable[[str], str]) -> None:
    before = path.read_text()
    after = transform(before)
    if after != before:
        path.write_text(after)


def _prepare_patch_input(target: Path, patch_name: str) -> None:
    """Normalize upstream benchmark comments before replaying public patches."""
    if target.name != "cutedsl_megamoe" or patch_name != "TENSORRT_LLM_MAIN_COMPAT.patch":
        return

    topk = target / "kernel_src/blackwell/inference/mega/topk_reduce.py"
    _replace_text(
        topk,
        lambda text: re.sub(
            r"(# lowers to an ALU subnormal-normalization path) \(~[^)]*\)(\. Both helpers)",
            r"\1\2",
            text,
        ),
    )
    mainloop = target / (
        "kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py"
    )

    def normalize_mainloop(text: str) -> str:
        text = text.replace(
            "from ....schedulers.fc12_mapping import BlockPhase\nfrom . import dynamic_mainloop\n",
            "from ....schedulers.fc12_mapping import BlockPhase\n"
            "from ..local_mega.tma_gather import sm107_tma_gather4_load\n"
            "from . import dynamic_mainloop\n",
        ).replace("from .tma_gather import sm107_tma_gather4_load\n", "")
        text = re.sub(
            r"    # Token-side depth once the operands stop sharing one\..*?suggests\.\n",
            "    # Pin the token-side pipeline depth so wider token tiles do not consume the\n"
            "    # weight-side staging budget. Revalidate this specialization when its tile or\n"
            "    # shared-memory plan changes.\n",
            text,
            flags=re.S,
        )
        text = re.sub(
            r"    # The token tile this specialization applies to\..*?\n"
            r"    # already reaches .*?\n",
            "    # The token tile this specialization applies to; other tiles use the shared plan.\n",
            text,
        )
        return re.sub(
            r"        # Only the .*? token tile is specialized:.*?\n"
            r"        # already reaches .*?\n",
            "        # Only the selected token tile uses asymmetric operand stages.\n",
            text,
        )

    _replace_text(mainloop, normalize_mainloop)


def _sanitize_vendored_comments(target: Path) -> None:
    """Remove source-specific benchmark evidence while preserving implementation."""
    if target.name == "megamoe_scheduler_v2":
        halo = target / "cuda_scheduler/csrc/halo_q_scheduler.cu"

        def sanitize_halo(text: str) -> str:
            text = re.sub(
                r"/\* Pure-CUDA fused physical-slot scheduler for GB\d+ \(sm_\d+\)\.",
                "/* Pure-CUDA fused physical-slot scheduler for supported architectures.",
                text,
            )
            text = text.replace(
                "the\n         * measured CuTe reset-free rendezvous.",
                "the\n         * reset-free rendezvous.",
            )
            text = re.sub(
                r"/\* `routes` is cold on every iteration.*?"
                r"atomicAdd regrouping is safe because integer addition commutes\. \*/",
                "/* Issue a small load group before consuming it to expose memory-level\n"
                "     * parallelism. The binning is unchanged, and atomicAdd regrouping is\n"
                "     * safe because integer addition commutes. */",
                text,
                flags=re.S,
            )
            text = re.sub(
                r"/\* Four-way unrolling this.*?compiler already pipelines it\. \*/",
                "/* Keep the simple accumulation loop; the compiler provides the required\n"
                "     * pipelining without explicit batching. */",
                text,
                flags=re.S,
            )
            text = re.sub(
                r"/\* Widening these to int4.*?see below\.\) \*/",
                "/* Keep scalar remote stores for the publication path; the local mirror\n"
                "         * copy below uses vector transfers when alignment permits. */",
                text,
                flags=re.S,
            )
            text = re.sub(
                r"/\* Same 4-byte issue limit.*?\*/",
                "/* Vectorize the local mirror copy when shape and alignment permit. */",
                text,
                flags=re.S,
            )
            text = re.sub(
                r" \* Replaces a single-threaded.*?elements\.\n",
                " * This replaces serial insertion-sort and mask-construction loops while\n"
                " * preserving the same stable order.\n",
                text,
                flags=re.S,
            )
            text = re.sub(
                r"/\* The rank scan reads.*?write nothing\. \*/",
                "/* Read p_expert cooperatively and distribute values with shuffles. All\n"
                "     * lanes must participate before inactive lanes return; inactive lanes\n"
                "     * carry INT_MAX and never contribute to a real comparison. */",
                text,
                flags=re.S,
            )
            text = re.sub(
                r"/\* EP\d+ used to run one round.*?delta 0\. \*/",
                "/* Use two unconditional repair rounds. Phases with no required repair\n"
                "                 * are no-ops, so this preserves the conditional algorithm's result. */",
                text,
                flags=re.S,
            )
            text = re.sub(
                r"/\* The EP\d+ .*? repair scan lived here;.*?equivalent\. \*/",
                "/* The unconditional repair rounds above replace the conditional scan. */",
                text,
                flags=re.S,
            )
            text = re.sub(
                r"/\* build_route_prefix reads.*?kernel time\. \*/",
                "/* build_route_prefix is independent of warp 8's outputs. Defer their\n"
                "             * rendezvous until both paths reach their first shared consumer so\n"
                "             * prefix construction and coloring can overlap safely. */",
                text,
                flags=re.S,
            )
            return re.sub(r"/\* Release the \d+ worker CTAs", "/* Release the worker CTAs", text)

        _replace_text(halo, sanitize_halo)
        _replace_text(
            target / "csrc/in_switch_copy/tma_copy.h",
            lambda text: re.sub(
                r"the kernel/device thread limit \(GB\d+: at most \d+ warps with \d+ total slots\)\.",
                "the active kernel and device limits.",
                text,
            ),
        )
        _replace_text(
            target / "cuda_scheduler/runtime.py",
            lambda text: re.sub(
                r'"""Return the measured pure-CUDA CTA policy',
                '"""Return the bounded pure-CUDA CTA policy',
                text,
            )
            if "# The latency-oriented" not in text
            else re.sub(
                r"        # The latency-oriented EP\d+ policy.*?unoccupied\.\n",
                "        # Bound scheduler occupancy so independent same-stream work retains\n"
                "        # launch capacity.\n",
                re.sub(
                    r'"""Return the measured pure-CUDA CTA policy',
                    '"""Return the bounded pure-CUDA CTA policy',
                    text,
                ),
                flags=re.S,
            ),
        )
        _replace_text(
            target / "sami/fabric.py",
            lambda text: re.sub(
                r"That import sits inside a function body on purpose.*?conclude, again,\n",
                "That import sits inside a function body on purpose: importing this package at\n"
                "module scope triggers an eager native build. A module-level grep or AST closure\n"
                "therefore does not see it and may conclude\n",
                text,
                flags=re.S,
            ),
        )
        hierarchical = target / "sami/hierarchical.py"

        def sanitize_timeout(text: str) -> str:
            text = re.sub(
                r"# Bound on waiting for every peer.*?HierarchicalCopyEndpoint\.submit\.\n",
                "# Bound on waiting for every peer to publish its mapped plan. Steady state\n"
                "# surfaces a lost peer promptly; the first generation allows for compilation\n"
                "# inside the rendezvous and the resulting arrival skew.\n",
                text,
                flags=re.S,
            )
            start = "        ``timeout_ns=None`` picks the bound automatically, and the first\n"
            end = '        """\n\n        if timeout_ns is None:'
            if start not in text:
                return text
            prefix, tail = text.split(start, 1)
            _, suffix = tail.split(end, 1)
            neutral = (
                "        ``timeout_ns=None`` selects a wider first-generation bound because\n"
                "        ranks can enter the rendezvous at different times while required kernels are\n"
                "        compiled. Later generations use the steady-state bound so a lost peer surfaces\n"
                "        promptly. ``MEGAMOE_SAMI_PLAN_TIMEOUT_NS`` can raise both defaults; callers may\n"
                "        pass ``timeout_ns`` explicitly when a different diagnostic bound is required.\n"
                "        The timeout is an error bound, not a correctness barrier.\n"
                '        """\n\n        if timeout_ns is None:'
            )
            return prefix + neutral + suffix

        _replace_text(hierarchical, sanitize_timeout)
        _replace_text(
            target / "integrations/megamoe/README.md",
            lambda text: re.sub(
                r"native\n\d+-argument AOT\. Both Rubin token-tile",
                "native\nAOT interface. Supported token-tile",
                text,
            ),
        )
        return

    if target.name != "cutedsl_megamoe":
        return

    scheduler_mode_literals = (
        'token_back_ready_granularity == "token_tile"',
        'token_back_ready_granularity == "expert"',
        'token_back_schedule_mode != "atomic_counter"',
    )
    scanner_waiver = "  # nosec B105 -- scheduler mode name, not a credential"

    def annotate_scheduler_modes(text: str) -> str:
        lines = []
        for line in text.splitlines(keepends=True):
            body, ending = (line[:-1], "\n") if line.endswith("\n") else (line, "")
            if (
                any(literal in body for literal in scheduler_mode_literals)
                and "# nosec B105" not in body
            ):
                body += scanner_waiver
            lines.append(body + ending)
        return "".join(lines)

    for rel in (
        "communication/nvlink_domain/token_comm.py",
        "kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_epilogue.py",
        "kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py",
    ):
        _replace_text(target / rel, annotate_scheduler_modes)

    for rel in (
        "kernel_src/rubin/inference/local_mega/block_scaled_swap_ab_local_mega_moe_kernel.py",
        "kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py",
    ):
        _replace_text(
            target / rel,
            lambda text: text.replace(
                "because its multi-window path has no measured gain.",
                "because that multi-window configuration is unsupported.",
            ),
        )
    _replace_text(
        target
        / "kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py",
        lambda text: re.sub(
            r"    # Token-side depth once the operands stop sharing one\..*?suggests\.\n",
            "    # Pin the token-side pipeline depth so wider token tiles do not consume the\n"
            "    # weight-side staging budget. Revalidate this specialization when its tile or\n"
            "    # shared-memory plan changes.\n",
            text,
            flags=re.S,
        ),
    )
    _replace_text(
        target
        / "kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py",
        lambda text: re.sub(
            r"    # Tuned for Rubin, not portable\..*?cluster capacity\.\n",
            "    # The communication grid must preserve full residency of the persistent main\n"
            "    # kernel. Revalidate this specialization when device or cluster capacity changes.\n",
            text,
            flags=re.S,
        ),
    )


def _format_scheduler_sources(root: Path, target: Path) -> None:
    sources = sorted(
        path
        for path in target.rglob("*")
        if path.is_file() and path.suffix in {".c", ".cu", ".cuh", ".h"}
    )
    if not sources:
        return
    command = [
        sys.executable,
        "-m",
        "pre_commit",
        "run",
        "clang-format",
        "--files",
        *map(str, sources),
    ]
    # The first pass returns 1 when clang-format rewrites a file.  A clean
    # second pass distinguishes that expected result from hook/setup failures.
    subprocess.run(command, cwd=root, check=False)
    subprocess.run(command, cwd=root, check=True)


def _apply_downstream_patches(root: Path, target: Path, patches: list[dict]) -> None:
    for patch in patches:
        patch_path = target / patch["patch"]
        formatter = patch.get("format_before")
        if formatter:
            version = subprocess.check_output(["ruff", "--version"], text=True).strip()
            if version != formatter:
                raise RuntimeError(f"Downstream patch requires {formatter}; found {version}")
            subprocess.run(
                ["ruff", "format", "--config", str(root / "pyproject.toml"), str(target)],
                cwd=root,
                check=True,
            )
        _prepare_patch_input(target, patch["patch"])
        subprocess.run(
            ["git", "apply", "--recount", "--check", str(patch_path)], cwd=root, check=True
        )
        subprocess.run(["git", "apply", "--recount", str(patch_path)], cwd=root, check=True)
        for file in patch.get("files", [patch]):
            digest = hashlib.sha256((target / file["file"]).read_bytes()).hexdigest()
            if digest != file["patched_sha256"]:
                raise RuntimeError(f"Patched source digest mismatch: {file['file']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", required=True, type=Path)
    parser.add_argument("--megamoe", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target_root = root / "tensorrt_llm/_torch/cute_dsl_kernels"
    scheduler_src = args.scheduler / "megamoe_scheduler"
    scheduler_dst = target_root / "megamoe_scheduler_v2"
    mega_src = args.megamoe / "next/sources"
    mega_dst = target_root / "cutedsl_megamoe"
    scheduler_patches = _read_downstream_patches(scheduler_dst, scheduler_src, header=False)
    mega_patches = _read_downstream_patches(mega_dst, mega_src, header=True)
    scheduler_files = []
    for source in sorted(scheduler_src.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        rel = source.relative_to(scheduler_src)
        copy_source(source, scheduler_dst / rel, header=False)
        scheduler_files.append(str(rel))

    _sanitize_vendored_comments(scheduler_dst)
    _apply_downstream_patches(root, scheduler_dst, scheduler_patches)
    _format_scheduler_sources(root, scheduler_dst)

    pending = [
        path.relative_to(mega_dst) for path in mega_dst.rglob("*.py") if path.name != "__init__.py"
    ]
    pending.extend(
        (
            Path("kernel_src/schedulers/__init__.py"),
            Path("kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_kernel.py"),
        )
    )
    copied = set()
    while pending:
        rel = pending.pop()
        if rel in copied:
            continue
        source = mega_src / rel
        if not source.is_file():
            raise RuntimeError(f"Vendored inference module disappeared upstream: {rel}")
        copy_source(source, mega_dst / rel, header=True)
        copied.add(rel)
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            base = rel.parent
            for _ in range(node.level - 1):
                base = base.parent
            if node.module:
                base = base.joinpath(*node.module.split("."))
            candidates = [base.with_suffix(".py"), base / "__init__.py"]
            candidates.extend((base / name.name).with_suffix(".py") for name in node.names)
            for candidate in candidates:
                if not (mega_src / candidate).is_file():
                    continue
                if candidate.name == "__init__.py" and (mega_dst / candidate).exists():
                    continue
                pending.append(candidate)

    _apply_downstream_patches(root, mega_dst, mega_patches)
    _sanitize_vendored_comments(mega_dst)
    for repo, target, files, kind, upstream in (
        (
            args.scheduler,
            scheduler_dst,
            scheduler_files,
            "complete scheduler package",
            "external scheduler source",
        ),
        (
            args.megamoe,
            mega_dst,
            [str(path) for path in sorted(copied)],
            "inference closure",
            "external MegaMoE source",
        ),
    ):
        source_subdir = "next/sources" if repo == args.megamoe else "megamoe_scheduler"
        source_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain", "--", source_subdir], text=True
            ).strip()
        )
        previous_manifest = json.loads((target / "VENDOR_MANIFEST.json").read_text())
        record = {
            "commit": revision(repo),
            "source_tree_dirty": source_dirty,
            "kind": kind,
        }
        if "retained_legacy_host" in previous_manifest:
            record["retained_legacy_host"] = previous_manifest["retained_legacy_host"]
        record["files"] = {
            name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in files
        }
        patches = scheduler_patches if target == scheduler_dst else mega_patches
        if patches:
            record.update(
                source_tree_dirty_scope=(
                    "Upstream checkout used for vendoring only; downstream changes are recorded separately."
                ),
                downstream_modified=True,
                downstream_patches=patches,
            )
        (target / "VENDOR_MANIFEST.json").write_text(json.dumps(record, indent=2) + "\n")
        downstream = "downstream_modified: true\n" if patches else ""
        (target / "VENDOR_STAMP").write_text(
            f"upstream: {upstream}\ncommit: {record['commit']}\n"
            f"source_tree_dirty: {str(source_dirty).lower()}\nkind: {kind}\n"
            f"{downstream}manifest: VENDOR_MANIFEST.json\n"
            "Regenerate with scripts/vendor_dynamic_load_balance.py; recorded downstream patches are replayed.\n"
        )
        print(f"{target.name}: {record['commit']} ({len(files)} files)")


if __name__ == "__main__":
    main()
