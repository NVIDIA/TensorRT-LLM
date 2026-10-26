# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Draft-window compare for the F0a dumps (TRTLLM_M31_F0A_DUMP_DIR). Not a pytest module.

Usage: ``python f0a_g1d_compare.py DUMP_DIR CTX_TP GEN_TP OUT_JSON``

``CTX_TP`` is the context side's TP ranks per attention-DP group: 1 under attention DP, where
one rank holds every drafter head and its dump is whichever rank wrote it; the TP size under
TP-attention (e.g. 2 for tp2 without attention DP), where TP rank ``c`` holds heads
``[c * H_ctx, (c + 1) * H_ctx)`` and each request has one dump per CTX rank, all with the same
``match_id``. ``GEN_TP`` is the generation side's. Dumps pair by
``match_id`` (the context request id both sides know). Generation rank ``r`` holds global
head ``r * H // GEN_TP`` (``H`` = the context side's ``heads_per_rank * CTX_TP``): head ``r``
at TP4, ``r // 2`` at TP8 for four drafter heads, as ``duplicate_kv_weight`` replicates them.
Every row the generation window holds below ``ctx_len`` is compared bitwise with the context
side's row, for every layer and K/V; the seeded ``(ctx_len, ctx_base)`` must equal the shipped
one. A context-side fallback (no window shipped) is listed as ``fallback``.

Verdict ``G1D_PASS`` when every compared request is bitwise equal with equal state and at least
one request was compared, else ``G1D_FAIL``. Exit status 0 on pass, 1 on fail.
"""

import glob
import json
import os
import sys

import torch

FORMAT = "f0a_g1d_v1"


def _load(directory):
    ctx, gen = {}, {}
    for path in sorted(glob.glob(os.path.join(directory, "*.pt"))):
        dump = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(dump, dict) or dump.get("format") != FORMAT:
            continue
        if dump["side"] == "ctx":
            ctx.setdefault(dump["match_id"], []).append(dump)
        elif dump["side"] == "gen":
            gen.setdefault(dump["match_id"], []).append(dump)
    return ctx, gen


def _rows_by_ordinal(dump, group):
    return {int(o): i for i, o in enumerate(dump["ordinals"][group].tolist())}


def _global_head(gen_rank, local_head, gen_heads, gen_tp, total_heads):
    """The drafter head generation rank ``gen_rank`` holds in ``local_head`` (HND mapper rule)."""
    return (gen_rank * gen_heads + local_head) * total_heads // (gen_tp * gen_heads) % total_heads


def _compare_rank(ctx_dumps, gen_dump, gen_rank, ctx_tp, gen_tp):
    """(K/V rows compared, mismatched K/V rows, problems) for one generation rank's dump.

    ``ctx_dumps`` maps a context TP rank within its DP group to its dump; under attention DP
    (``ctx_tp == 1``) it is the one rank that dumped the request.
    """
    problems = []
    ctx_heads = next(iter(ctx_dumps.values()))["heads_per_rank"]
    total_heads = ctx_heads * ctx_tp
    tpb = gen_dump["tokens_per_block"]
    ctx_len = gen_dump["ctx_len"]
    rows = mismatches = 0
    for gen_layer in gen_dump["layers"]:
        key = (gen_layer["layer_group"], gen_layer["global_layer_id"])
        group = key[0]
        for local_head in range(gen_dump["heads_per_rank"]):
            head = _global_head(
                gen_rank, local_head, gen_dump["heads_per_rank"], gen_tp, total_heads
            )
            ctx_dump = ctx_dumps.get(head // ctx_heads)
            if ctx_dump is None:
                problems.append(f"no context dump holds head {head}")
                continue
            ctx_layer = {
                (r["layer_group"], r["global_layer_id"]): r for r in ctx_dump["layers"]
            }.get(key)
            if ctx_layer is None:
                problems.append(f"layer {key} missing on the context side")
                continue
            ctx_index = _rows_by_ordinal(ctx_dump, group)
            ctx_head = head % ctx_heads
            for gen_i, ordinal in enumerate(gen_dump["ordinals"][group].tolist()):
                valid = max(0, min(tpb, ctx_len - int(ordinal) * tpb))
                if valid == 0:
                    continue
                ctx_i = ctx_index.get(int(ordinal))
                if ctx_i is None:
                    problems.append(f"ordinal {ordinal} held on generation, not on context")
                    rows += 2 * valid
                    mismatches += 2 * valid
                    continue
                for name in ("k", "v"):
                    got = gen_layer[name][gen_i, local_head, :valid]
                    want = ctx_layer[name][ctx_i, ctx_head, :valid]
                    rows += valid
                    mismatches += int((got != want).any(dim=-1).sum())
    return rows, mismatches, problems


def compare(directory, ctx_tp, gen_tp):
    ctx, gen = _load(directory)
    report = {"dump_dir": directory, "ctx_tp": ctx_tp, "gen_tp": gen_tp, "requests": {}}
    compared = 0
    passed = True
    for match_id in sorted(set(ctx) | set(gen)):
        ctx_dumps = ctx.get(match_id, [])
        entry = {"ctx_ranks": [d["rank"] for d in ctx_dumps]}
        report["requests"][str(match_id)] = entry
        shipped = [d for d in ctx_dumps if d["shipped"]]
        if ctx_dumps and not shipped:
            entry["status"] = "fallback"
            entry["reason"] = ctx_dumps[0].get("reason")
            continue
        gen_dumps = gen.get(match_id, [])
        if not shipped or not gen_dumps:
            entry["status"] = "unpaired"
            entry["gen_ranks"] = [d["rank"] for d in gen_dumps]
            passed = False
            continue
        # Under attention DP the context rank holding the request is whichever dumped it.
        ctx_dump = shipped[0]
        ctx_by_tp_rank = {d["rank"] % ctx_tp: d for d in shipped} if ctx_tp > 1 else {0: ctx_dump}
        entry.update(
            status="compared",
            ctx_len=ctx_dump["ctx_len"],
            ctx_base=ctx_dump["ctx_base"],
            gen_ranks={},
        )
        ok = True
        for gen_dump in sorted(gen_dumps, key=lambda d: d["rank"]):
            rows, mismatches, problems = _compare_rank(
                ctx_by_tp_rank, gen_dump, gen_dump["rank"] % gen_tp, ctx_tp, gen_tp
            )
            state_equal = (gen_dump["ctx_len"], gen_dump["ctx_base"]) == (
                ctx_dump["ctx_len"],
                ctx_dump["ctx_base"],
            )
            entry["gen_ranks"][str(gen_dump["rank"])] = {
                "rows_compared": rows,
                "mismatches": mismatches,
                "ctx_len_equal": gen_dump["ctx_len"] == ctx_dump["ctx_len"],
                "ctx_base_equal": gen_dump["ctx_base"] == ctx_dump["ctx_base"],
                "problems": problems,
            }
            ok = ok and mismatches == 0 and state_equal and not problems
        entry["bitwise"] = ok
        passed = passed and ok
        compared += 1
    report["requests_compared"] = compared
    report["verdict"] = "G1D_PASS" if passed and compared > 0 else "G1D_FAIL"
    return report


def main(argv):
    if len(argv) != 5:
        print(__doc__)
        return 2
    _, directory, ctx_tp, gen_tp, out_json = argv
    report = compare(directory, int(ctx_tp), int(gen_tp))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print(f"{report['verdict']}: {report['requests_compared']} request(s) compared")
    return 0 if report["verdict"] == "G1D_PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
