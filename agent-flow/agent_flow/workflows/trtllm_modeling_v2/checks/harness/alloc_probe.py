#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The allocation probe, as each rank runs it: what a launch gives a rank, with no model loaded.

Started as an engine is started (one process per rank of one world), it answers five questions
in seconds: did the launch make the world it was asked for, does the build import, does a
collective cross every rank, do the cache directories take a file, is the dataset there::

    python alloc_probe.py --out DIR --expect-ranks R [--role ROLE] [--module NAME]...
                          [--cache-dir DIR]... [--dataset PATH]...

Each rank writes ``DIR/<role>.rank<k>.<host>.<pid>.json`` and one ``[alloc-probe]`` line, and
exits 0 when its own answers hold; ``infra/compute/probe.py`` judges the records together. Standard library
only until a rank imports what it is asked to import; ``mpi4py`` is the engine's own.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import sys
import time
from pathlib import Path


def _rank_from_environment() -> int:
    for name in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID"):
        value = os.environ.get(name, "")
        if value.isdigit():
            return int(value)
    return 0


def collective(expect: int) -> dict:
    """This rank's place in the world, and whether a reduction and a gather cross all of it."""
    try:
        from mpi4py import MPI
    except Exception as exc:  # no MPI binding: a world of one needs none
        ok = expect <= 1
        why = "" if ok else f"mpi4py does not import ({type(exc).__name__}: {exc})"
        return {"rank": _rank_from_environment(), "size": 1, "ok": ok, "why": why, "hosts": []}
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    total = comm.allreduce(rank + 1, op=MPI.SUM)
    hosts = comm.allgather(socket.gethostname())
    problems = []
    if size != expect:
        problems.append(f"the world has {size} rank(s), {expect} were asked for")
    if total != size * (size + 1) // 2:
        problems.append(f"a sum over the ranks gave {total}, not {size * (size + 1) // 2}")
    return {
        "rank": rank,
        "size": size,
        "ok": not problems,
        "why": "; ".join(problems),
        "hosts": hosts,
    }


def imports(modules: list[str]) -> dict:
    """Where each module comes from, or why it does not import."""
    found, problems = {}, []
    for name in modules:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            problems.append(f"{name} does not import ({type(exc).__name__}: {exc})")
            continue
        found[name] = {
            "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    return {"ok": not problems, "why": "; ".join(problems), "modules": found}


def takes_a_file(directory: str, rank: int) -> str:
    """``""`` when ``directory`` takes, returns and gives up a file of this rank; else why not."""
    path = Path(directory) / f".alloc-probe.{socket.gethostname()}.{os.getpid()}.{rank}"
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path.write_text("probe\n", encoding="utf-8")
        if path.read_text(encoding="utf-8") != "probe\n":
            return f"{directory} gave back other bytes than it took"
        path.unlink()
    except OSError as exc:
        return f"{directory} does not take a file ({exc.strerror or exc})"
    return ""


def reads(path: str) -> str:
    """``""`` when ``path`` is a readable file or a directory that holds something; else why not."""
    target = Path(path)
    try:
        if target.is_dir():
            return "" if any(target.iterdir()) else f"{path} is an empty directory"
        with target.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        return f"{path} cannot be read ({exc.strerror or exc})"
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alloc_probe.py", description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expect-ranks", type=int, default=1)
    parser.add_argument("--role", default=os.environ.get("MV2FLOW_ROLE") or "probe")
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--cache-dir", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    args = parser.parse_args(argv)

    world = collective(args.expect_ranks)
    rank = int(world["rank"])
    loaded = imports(args.module)
    caches = [why for why in (takes_a_file(d, rank) for d in args.cache_dir) if why]
    data = [why for why in (reads(p) for p in args.dataset) if why]
    problems = [why for why in (world["why"], loaded["why"], *caches, *data) if why]
    record = {
        "role": args.role,
        "rank": rank,
        "size": world["size"],
        "host": socket.gethostname(),
        "hosts": world["hosts"],
        "pid": os.getpid(),
        "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "modules": loaded["modules"],
        "ok": not problems,
        "problems": problems,
        "at": time.time(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    # One file per process, never per rank: two processes that take themselves for one rank
    # must both be seen.
    target = args.out / f"{args.role}.rank{rank}.{record['host']}.{os.getpid()}.json"
    staging = target.with_name(f".{target.name}.{os.getpid()}")
    staging.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, target)
    said = "ok" if not problems else "FAIL: " + "; ".join(problems)
    print(f"[alloc-probe] {args.role} rank {rank}/{world['size']} on {record['host']}: {said}")
    sys.stdout.flush()
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
