# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the two collective MTP draft-step entries of the GLM-5.2 tp16 target.

`glm_draft_ehproj_table` and `glm_draft_sample_glue` exchange through an MNNVL workspace across the TP group,
so neither can run in one process. Run plainly, this script is its own launcher: it spawns one rank per device
named in CUDA_VISIBLE_DEVICES, under a deadline it enforces by killing every rank's session, because a broken
exchange hangs (a rank polls for a packet that never lands) rather than raising.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _draft_step_op_matrix.py draft_ehproj_table

Each rank drives every cell of the entry, a draft loop of the target's shape (six table steps and the last glue
step, repeated) eagerly and as CUDA-graph replays, and the entry's guards. The 16-rank geometry of the table
kernel (a CTA's columns spanning two ranks' hidden-half shares) needs 16 ranks over 4 nodes: start one process
per GPU with an external launcher and `--rank-worker`, with RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR and
MASTER_PORT in the environment (SLURM_PROCID, SLURM_NTASKS and SLURM_LOCALID stand in for the first three):

    srun -N4 --ntasks-per-node=4 python _draft_step_op_matrix.py draft_ehproj_table --rank-worker

Not a pytest module: the collected entry points are `test_modeling_v2_draft_ehproj_table.py` and
`test_modeling_v2_draft_sample_glue.py` beside this file. The launcher half imports nothing from
tensorrt_llm, and the ranks reach the target's ops by absolute import.
"""

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 1500

HIDDEN = 6144
EPS = 1e-5
MAX_BLOCKS = 64
#: The DSA indexer-K cache layout of GLM-5.2: 64 tokens per block, 128 FP8 bytes and one FP32 scale per token.
TOKENS_PER_BLOCK = 64
DATA_BYTES = 128
SCALE_BYTES = 4
TOKEN_LANES = 8

# Bound in the rank body: importing the entries pulls in tensorrt_llm.
RANK = 0
WORLD = 1
HOST_GROUP: Any = None
table_op: Any = None
glue_op: Any = None
validating: Any = None
assert_within_ulp: Any = None
WORKSPACES: Dict[str, Any] = {}


def _generator(*key: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(repr(key).encode()) % 2**31)


def _logits(spec: dict, seed: int, rank: int) -> torch.Tensor:
    """Rank `rank`'s BF16 logits shard; every rank can rebuild every other rank's."""
    rows, shard = spec["rows"], spec["vocab_shard"]
    g = _generator(seed, 1, rank)
    logits = torch.randn(rows, shard, generator=g, device="cuda")
    if spec["logits"] == "tied":
        # Every rank's maximum is 8 at two local positions: the lower position of rank 0 wins.
        first = torch.randint(0, shard // 2, (rows,), generator=g, device="cuda")
        second = first + shard // 2
        rows_index = torch.arange(rows, device="cuda")
        logits[rows_index, first] = 8.0
        logits[rows_index, second] = 8.0
    return logits.to(torch.bfloat16)


class _LazyTables:
    """Every rank's token table, built on first use: the reference reads only the winners' rows."""

    def __init__(self, seed: int, shard: int):
        self._seed, self._shard, self._tables = seed, shard, {}

    def __getitem__(self, rank: int) -> torch.Tensor:
        if rank not in self._tables:
            g = _generator(self._seed, 2, rank)
            self._tables[rank] = torch.randn(
                self._shard, HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16
            )
        return self._tables[rank]

    def __len__(self) -> int:
        return WORLD


def _hidden_half(seed: int, rank: int, rows: int) -> torch.Tensor:
    g = _generator(seed, 3, rank)
    return torch.randn(rows, HIDDEN // WORLD, generator=g, device="cuda")


def _state(spec: dict, seed: int) -> Dict[str, Any]:
    """The step's state arguments, identical on every rank."""
    rows, draft_len, step = spec["rows"], spec["draft_len"], spec["step"]
    g = _generator(seed, 4)
    vocab = spec["vocab_shard"] * WORLD
    num_accepted = torch.randint(1, draft_len + 2, (rows,), generator=g, device="cuda").int()
    batch_indices = torch.randperm(rows, generator=g, device="cuda").int()
    if WORLD == 8 and draft_len == 3 and rows == 2:
        num_accepted.copy_(torch.tensor([1, 4], dtype=torch.int32, device="cuda"))
        batch_indices.copy_(torch.tensor([1, 0], dtype=torch.int32, device="cuda"))
    state: Dict[str, Any] = dict(
        draft_tokens=torch.randint(0, vocab, (rows, draft_len), generator=g, device="cuda").int(),
        draft_step=step,
        positions_in=None,
        positions_out=None,
        gather_ids=None,
        seq_lens=None,
        fill_seq_lens=spec["fill"],
        kv_lens=None,
        kv_update=spec["kv_update"],
        rewind_accepted=None,
        draft_len=draft_len,
        block_offsets=None,
        dsa_layout=[],
        req_idx_per_token=None,
        slot_mapping_fp8=None,
        slot_mapping_scale=None,
        gen_kv_indptr=None,
        gen_cached_token_indptr=None,
    )
    verify_rows = rows * (draft_len + 1)
    if spec["positions"] == "gather":
        state["positions_in"] = torch.randint(
            100, 100000, (verify_rows,), generator=g, device="cuda"
        ).int()
        state["gather_ids"] = batch_indices.long() * (draft_len + 1) + num_accepted.long() - 1
        state["positions_out"] = torch.full((rows,), -9, dtype=torch.int32, device="cuda")
    elif spec["positions"] == "plain":
        state["positions_in"] = torch.randint(
            100, 100000, (rows,), generator=g, device="cuda"
        ).int()
        state["positions_out"] = torch.full((rows,), -9, dtype=torch.int32, device="cuda")
    seq_before = draft_len + 1 if spec["fill"] else 1
    state["seq_lens"] = torch.full((rows,), seq_before, dtype=torch.int32, device="cuda")
    state["kv_lens"] = (
        1000 + 37 * torch.arange(rows, device="cuda", dtype=torch.int32) + 3 * step
    ).int()
    if spec["kv_update"] == 2:
        state["rewind_accepted"] = num_accepted
    if spec["dsa"]:
        num_tokens = rows
        state["block_offsets"] = torch.randint(
            0, 4096, (rows, MAX_BLOCKS), generator=g, device="cuda"
        ).int()
        state["dsa_layout"] = [
            num_tokens,
            TOKEN_LANES,
            TOKENS_PER_BLOCK,
            DATA_BYTES,
            SCALE_BYTES,
            TOKENS_PER_BLOCK * (DATA_BYTES + SCALE_BYTES),
            TOKENS_PER_BLOCK * DATA_BYTES,
        ]
        state["req_idx_per_token"] = torch.full((num_tokens + 3,), -5, dtype=torch.int32).cuda()
        state["slot_mapping_fp8"] = torch.full((num_tokens + 3,), -5, dtype=torch.int64).cuda()
        state["slot_mapping_scale"] = torch.full((num_tokens + 3,), -5, dtype=torch.int64).cuda()
        state["gen_kv_indptr"] = torch.full((rows + 4,), -5, dtype=torch.int64).cuda()
        state["gen_cached_token_indptr"] = torch.full((rows + 4,), -5, dtype=torch.int64).cuda()
    state["_num_accepted"] = num_accepted
    state["_batch_indices"] = batch_indices
    return state


def _snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in state.items()
        if not k.startswith("_")
    }


def _assert_exact(name: str, out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    try:
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    except AssertionError as exc:
        raise AssertionError(f"{where}: {name}: {exc}") from exc


def _compare(
    expected: Dict[str, torch.Tensor], actual: Dict[str, torch.Tensor], where: str
) -> None:
    """Integers and x bit for bit; the normalized rows within one RMSNorm's rounding.

    The RMSNorm outputs (`normed_out`, `glue_out`) differ from the FP32 reference in the order of the 6144
    squares and in the kernel's approximate reciprocal square root: a few 1e-7 relative on the scale, one BF16
    ulp on the rare element that close to a rounding midpoint, which is up to twice the band's unit (2**-8 of the
    row maximum) in the maximum's binade. So 2 ulp against the row scale and 0.25 ulp of relative RMS, as for the
    step-0 glue kernel.
    """
    for name, ref in expected.items():
        out = actual[name]
        if name in ("normed_out", "glue_out"):
            try:
                assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.25)
            except AssertionError as exc:
                raise AssertionError(f"{where}: {name}: {exc}") from exc
        else:
            _assert_exact(name, out, ref, where)


def _table_call(spec: dict, seed: int, logits: List[torch.Tensor], tables: _LazyTables):
    """One table-op call on this rank: (call kwargs, reference kwargs)."""
    rows = spec["rows"]
    state = _state(spec, seed)
    halves = [_hidden_half(seed, r, rows) for r in range(WORLD)]
    norm_g = _generator(seed, 5)
    norm_weight = (1.0 + 0.1 * torch.randn(HIDDEN, generator=norm_g, device="cuda")).to(
        torch.bfloat16
    )
    workspace = WORKSPACES["table"]
    kwargs = dict(
        logits=logits[RANK],
        exchange_buffer=workspace.buffer,
        exchange_flags=workspace.flags,
        group_size=WORLD,
        hidden_half=halves[RANK],
        token_table=tables[RANK],
        norm_weight=norm_weight,
        eps=EPS,
        normed_out=torch.full((rows, HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda"),
        residual_out=torch.full((rows, HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda"),
        **{k: v for k, v in state.items() if not k.startswith("_")},
    )
    ref_kwargs = dict(
        kwargs, **_snapshot(state), logits=logits, hidden_half=halves, token_table=tables
    )
    return kwargs, ref_kwargs


def _glue_call(spec: dict, seed: int, logits: List[torch.Tensor], embed_table: torch.Tensor):
    """One glue-op call on this rank: (call kwargs, reference kwargs)."""
    rows, draft_len = spec["rows"], spec["draft_len"]
    state = _state(spec, seed)
    g = _generator(seed, 6)
    window = 2 * HIDDEN // WORLD
    kwargs = dict(
        logits=logits[RANK],
        exchange_buffer=WORKSPACES["glue"].buffer,
        exchange_flags=WORKSPACES["glue"].flags,
        group_size=WORLD,
        embed_table=None,
        hidden=None,
        enorm_weight=None,
        hnorm_weight=None,
        eps=EPS,
        col_lo=0,
        col_hi=0,
        glue_out=None,
        accepted_tokens=None,
        batch_indices=None,
        num_accepted=None,
        next_new_tokens=None,
        **{k: v for k, v in state.items() if not k.startswith("_")},
    )
    if spec["glue"]:
        kwargs.update(
            embed_table=embed_table,
            hidden=(3.0 * torch.randn(rows, HIDDEN, generator=g, device="cuda")).to(torch.bfloat16),
            enorm_weight=(1.0 + 0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).to(
                torch.bfloat16
            ),
            hnorm_weight=(1.0 + 0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).to(
                torch.bfloat16
            ),
            col_lo=RANK * window,
            col_hi=(RANK + 1) * window,
            glue_out=torch.full((rows, window), 7.0, dtype=torch.bfloat16, device="cuda"),
        )
    if spec["next_new"]:
        kwargs.update(
            accepted_tokens=torch.randint(
                0, 1000, (rows, draft_len + 1), generator=g, device="cuda"
            ).int(),
            batch_indices=state["_batch_indices"],
            num_accepted=state["_num_accepted"],
            next_new_tokens=torch.full((rows, draft_len + 1), -3, dtype=torch.int32, device="cuda"),
        )
    ref_kwargs = dict(kwargs, **_snapshot(state), logits=logits)
    return kwargs, ref_kwargs


_TABLE_OUTPUTS = ("draft_tokens", "normed_out", "residual_out")
_GLUE_OUTPUTS = ("draft_tokens", "glue_out", "next_new_tokens")
_STATE_OUTPUTS = (
    "positions_out",
    "seq_lens",
    "kv_lens",
    "req_idx_per_token",
    "slot_mapping_fp8",
    "slot_mapping_scale",
    "gen_kv_indptr",
    "gen_cached_token_indptr",
)


def _written(kwargs: Dict[str, Any], names) -> Dict[str, torch.Tensor]:
    return {n: kwargs[n] for n in names if kwargs.get(n) is not None}


def check_table_cells() -> int:
    for index, cell in enumerate(table_op.CELLS):
        spec = cell.spec
        seed = zlib.crc32(cell.why.encode()) % 2**31
        logits = [_logits(spec, seed, r) for r in range(WORLD)]
        tables = _LazyTables(seed, spec["vocab_shard"])
        kwargs, ref_kwargs = _table_call(spec, seed, logits, tables)
        with validating(table_op):
            table_op(**kwargs)
        torch.cuda.synchronize()
        expected = table_op.reference(**ref_kwargs)
        _compare(expected, _written(kwargs, _TABLE_OUTPUTS + _STATE_OUTPUTS), f"cell {index}")
    return len(table_op.CELLS)


def check_glue_cells() -> int:
    embed_table = None
    for index, cell in enumerate(glue_op.CELLS):
        spec = cell.spec
        seed = zlib.crc32(cell.why.encode()) % 2**31
        if spec["glue"] and (
            embed_table is None or embed_table.shape[0] != spec["vocab_shard"] * WORLD
        ):
            embed_table = torch.randn(
                spec["vocab_shard"] * WORLD,
                HIDDEN,
                generator=_generator(9),
                device="cuda",
                dtype=torch.bfloat16,
            )
        logits = [_logits(spec, seed, r) for r in range(WORLD)]
        kwargs, ref_kwargs = _glue_call(spec, seed, logits, embed_table)
        with validating(glue_op):
            glue_op(**kwargs)
        torch.cuda.synchronize()
        expected = glue_op.reference(**ref_kwargs)
        _compare(expected, _written(kwargs, _GLUE_OUTPUTS + _STATE_OUTPUTS), f"cell {index}")
    return len(glue_op.CELLS)


def _draft_loop_specs(draft_len: int | None = None, rows: int = 1):
    """TP8 K3/K7 or TP16 K7: table steps 0..K-2, then final sample/glue."""
    if draft_len is None:
        draft_len = 3 if WORLD == 8 else 7
    shard = 19360 if WORLD == 8 else 9680
    table_steps = []
    for step in range(draft_len - 1):
        table_steps.append(
            dict(
                table_op.CELLS[0 if step == 0 else 1].spec,
                rows=rows,
                draft_len=draft_len,
                vocab_shard=shard,
                step=step,
            )
        )
    glue_step = dict(
        glue_op.CELLS[0].spec, rows=rows, draft_len=draft_len, vocab_shard=shard, step=draft_len - 1
    )
    return table_steps, glue_step


def check_draft_loop_eager_and_graph() -> None:
    shapes = ((2, 3), (1, 3), (1, 7)) if WORLD == 8 else ((1, 7),)
    replays = [_check_draft_loop_eager_and_graph(rows, k) for rows, k in shapes]
    if WORLD == 8:
        for repeat in range(3):
            for replay in replays:
                replay(repeat)


def _check_draft_loop_eager_and_graph(rows: int, draft_len: int) -> Callable[[int], None]:
    """Seven consecutive calls per decode step over two workspaces, eagerly and as graph replays.

    The Lamport flags rotate on every call and each call empties the buffer of the previous one, so the
    exchange must stay correct across steps, decode steps and replays of a captured loop.
    """
    table_steps, glue_step = _draft_loop_specs(draft_len=draft_len, rows=rows)
    calls = []
    for step, spec in enumerate(table_steps):
        seed = 1000 + step
        logits = [_logits(spec, seed, r) for r in range(WORLD)]
        tables = _LazyTables(seed, spec["vocab_shard"])
        calls.append(("table", _table_call(spec, seed, logits, tables)))
    seed = 1000 + len(table_steps)
    logits = [_logits(glue_step, seed, r) for r in range(WORLD)]
    calls.append(("glue", _glue_call(glue_step, seed, logits, None)))

    def run_loop() -> None:
        for kind, (kwargs, _) in calls:
            (table_op if kind == "table" else glue_op)(**kwargs)

    def reset_and_expect():
        """Restore every buffer the loop updates in place; return the expected outputs per call."""
        expected = []
        for kind, (kwargs, ref_kwargs) in calls:
            for name, value in ref_kwargs.items():
                # The snapshots of the buffers the op updates in place are separate tensors.
                live = kwargs.get(name)
                if isinstance(value, torch.Tensor) and isinstance(live, torch.Tensor):
                    if live is not value:
                        live.copy_(value)
            op = table_op if kind == "table" else glue_op
            expected.append(op.reference(**ref_kwargs))
        return expected

    for repeat in range(3):
        expected = reset_and_expect()
        run_loop()
        torch.cuda.synchronize()
        for (kind, (kwargs, _)), exp in zip(calls, expected):
            names = (_TABLE_OUTPUTS if kind == "table" else _GLUE_OUTPUTS) + _STATE_OUTPUTS
            _compare(exp, _written(kwargs, names), f"eager loop {repeat} {kind}")

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        with torch.cuda.graph(graph, stream=side):
            run_loop()
    torch.cuda.current_stream().wait_stream(side)

    def replay(repeat):
        expected = reset_and_expect()
        graph.replay()
        torch.cuda.synchronize()
        for (kind, (kwargs, _)), exp in zip(calls, expected):
            names = (_TABLE_OUTPUTS if kind == "table" else _GLUE_OUTPUTS) + _STATE_OUTPUTS
            _compare(exp, _written(kwargs, names), f"graph replay {repeat} {kind}")

    for repeat in range(3):
        replay(repeat)
    return replay


def check_tp8_worker_feedback_eager_and_graph() -> None:
    """Real K3/K7 worker/table feedback, with deterministic attention/head substitutes.

    Native gather, table and final sample/glue run on the production workspaces.
    The synthetic attention consumes both projected tensors; the next hidden half
    consumes that output, so independent per-step fixtures cannot satisfy this check.
    """
    if WORLD != 8:
        return
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency._mtp_worker import (  # noqa: E501
        MTPDraftWorker,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency._small_batch_draft import (  # noqa: E501
        SmallBatchDraftKernels,
    )

    shard = 19360
    tables = _LazyTables(812, shard)
    # Materialize local and reference winner tables before capture.
    local_table, winner_table = tables[RANK], tables[WORLD - 1]
    graphs = []

    for batch, draft_len, accepted in ((2, 3, [1, 4]), (1, 3, [1]), (1, 7, [1]), (1, 7, [8])):
        window = draft_len + 1
        rows = batch * window
        order = torch.tensor(list(reversed(range(batch))), dtype=torch.int32, device="cuda")
        counts = torch.tensor(accepted, dtype=torch.int32, device="cuda")
        ids = torch.arange(rows, dtype=torch.int32, device="cuda")
        positions = torch.arange(rows, dtype=torch.int64, device="cuda") + 1000
        hidden = torch.randn(rows, HIDDEN, generator=_generator(batch), device="cuda").to(
            torch.bfloat16
        )
        seq = torch.full((batch,), window, dtype=torch.int32, device="cuda")
        kv_initial = torch.arange(batch, dtype=torch.int32, device="cuda") * 37 + 1000
        kv = kv_initial.clone()
        meta = SimpleNamespace(
            _seq_lens=torch.full((batch,), window, dtype=torch.int32),
            seq_lens_cuda=seq,
            kv_lens_cuda=kv,
            num_contexts=0,
            use_spec_decoding=True,
            kv_cache_manager="target",
            updates=[],
        )
        meta.set_in_mtp_draft_loop = lambda value, m=meta: setattr(m, "in_mtp_draft_loop", value)
        meta.set_skip_topk = lambda value, m=meta: setattr(m, "indexer_skip_topk", value)
        meta.set_mtp_num_accepted = lambda value, m=meta: setattr(m, "mtp_num_accepted", value)
        meta.on_update = lambda: None
        meta.update_for_spec_dec = lambda m=meta, s=seq, k=kv: m.updates.append(
            (s.clone(), k.clone())
        )
        helper = object.__new__(SmallBatchDraftKernels)
        helper.use_table, helper.hidden_size = True, HIDDEN
        helper.DRAFT_LEN = draft_len
        helper.decode = SimpleNamespace(mapping=SimpleNamespace(tp_size=WORLD))
        helper.sample_exchange, helper.table_exchange = WORKSPACES["glue"], WORKSPACES["table"]
        helper.token_table, helper.half_done = local_table, torch.cuda.Event()
        helper.layer = SimpleNamespace(
            input_layernorm=SimpleNamespace(
                weight=torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda"), variance_epsilon=EPS
            )
        )
        helper.supports = (
            lambda metadata, first, selected, window=window: first == selected * window
        )
        values = []
        seen = []

        def head(value, *, next_step=False, h=helper, values=values, b=batch):
            step = len(values)
            values.append(value)
            logits = value.new_full((b, shard), -100)
            token = (torch.arange(b, device="cuda") * 101 + step * 37 + RANK * 7).long()
            logits.scatter_(1, token[:, None], RANK + 1)
            half = (
                value[:, RANK * (HIDDEN // WORLD) : (RANK + 1) * (HIDDEN // WORLD)]
                .float()
                .contiguous()
            )
            h.half_done.record()
            return logits, half if next_step else None

        helper.head = head

        @contextlib.contextmanager
        def draft_context(metadata, manager):
            original = metadata.kv_cache_manager
            metadata.kv_cache_manager = manager
            try:
                yield metadata
            finally:
                metadata.kv_cache_manager = original

        def forward(model, inputs, spec, step, seen=seen):
            seen.append((inputs["hidden_states"].shape[0], inputs["position_ids"].clone()))
            if step == 0:
                value, residual = inputs["row_select"](
                    inputs["hidden_states"], inputs["hidden_states"] + 10
                )
            else:
                value, residual = inputs["eh_proj_output"]
                assert "eh_proj_input" not in inputs and "row_select" not in inputs
            return (value.float() + 0.01 * residual.float()).to(torch.bfloat16), None

        worker = SimpleNamespace(
            _drafted_rows=None, draft_kv_cache_context=draft_context, _run_draft_forward=forward
        )
        spec = SimpleNamespace(runtime_draft_len=draft_len, batch_indices_cuda=order)
        inputs = dict(
            input_ids=ids,
            position_ids=positions,
            hidden_states=hidden,
            attn_metadata=meta,
            spec_metadata=spec,
        )

        def run(
            w=worker,
            h=helper,
            ins=inputs,
            m=meta,
            sp=spec,
            b=batch,
            accepted=counts,
            v=values,
            s=seen,
        ):
            v.clear()
            s.clear()
            m.updates.clear()
            drafted = MTPDraftWorker._forward_small_batch_draft_loop(
                w, h, ins, m, sp, object(), "draft", b, accepted
            )
            return drafted, w._drafted_rows[1], list(v), list(s), list(m.updates)

        def reset(m=meta, seq=seq, kv=kv, initial=kv_initial, window=window):
            m._seq_lens.fill_(window)
            seq.fill_(window)
            kv.copy_(initial)
            m.use_spec_decoding = True

        def check(
            result,
            b=batch,
            order=order,
            counts=counts,
            hidden=hidden,
            positions=positions,
            initial=kv_initial,
            m=meta,
            draft_len=draft_len,
            window=window,
        ):
            drafted, next_tokens, step_values, seen_rows, updates = result
            selected = order.long() * window + counts.long() - 1
            value = (hidden[selected].float() + 0.01 * (hidden[selected] + 10).float()).to(
                torch.bfloat16
            )
            expected_tokens = torch.stack(
                [
                    torch.arange(b, device="cuda", dtype=torch.int32) * 101
                    + step * 37
                    + (WORLD - 1) * 7
                    + (WORLD - 1) * shard
                    for step in range(draft_len)
                ],
                dim=1,
            )
            _assert_exact("draft_tokens", drafted, expected_tokens, "connected worker")
            _assert_exact("accepted", next_tokens[:, 0], selected.int(), "connected worker")
            _assert_exact("next drafts", next_tokens[:, 1:], drafted, "connected worker")
            assert [item[0] for item in seen_rows] == [b * window] + [b] * (draft_len - 1)
            for step in range(draft_len):
                assert_within_ulp(step_values[step], value, element_ulp=2.0, rms_ulp=0.5)
                if step:
                    _assert_exact(
                        "positions",
                        seen_rows[step][1],
                        positions[selected] + step,
                        "connected worker",
                    )
                _assert_exact(
                    "seq_lens", updates[step][0], torch.ones_like(counts), "connected worker"
                )
                _assert_exact(
                    "kv_lens",
                    updates[step][1],
                    initial - (draft_len - counts) + step,
                    "connected worker",
                )
                if step < draft_len - 1:
                    residual = (
                        winner_table[(expected_tokens[:, step] % shard).long()].float()
                        + value.float()
                    ).to(torch.bfloat16)
                    normed = (
                        residual.float()
                        * torch.rsqrt(residual.float().square().mean(-1, keepdim=True) + EPS)
                    ).to(torch.bfloat16)
                    value = (normed.float() + 0.01 * residual.float()).to(torch.bfloat16)
            assert m.kv_cache_manager == "target" and not m.in_mtp_draft_loop
            assert not m.indexer_skip_topk and m.mtp_num_accepted is None

        reset()
        eager = run()
        torch.cuda.synchronize()
        check(eager)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run()
        # Retain captured input buffers and the helper's CUDA events for every graph.
        graphs.append((graph, reset, check, result, hidden, positions, helper, inputs))

    for repeat in range(3):
        for graph, reset, check, result, hidden, positions, _helper, _inputs in graphs:
            hidden.copy_(
                torch.randn(
                    hidden.shape, generator=_generator(900 + repeat, hidden.shape[0]), device="cuda"
                ).to(torch.bfloat16)
            )
            positions.add_(17)
            reset()
            graph.replay()
            torch.cuda.synchronize()
            check(result)


def check_guards_refuse() -> None:
    """The guards refuse their silent-wrong inputs on every rank before anything is launched."""
    spec = dict(table_op.CELLS[0].spec)
    seed = 77
    logits = [_logits(spec, seed, r) for r in range(WORLD)]
    kwargs, _ = _table_call(spec, seed, logits, _LazyTables(seed, spec["vocab_shard"]))
    bad = dict(kwargs, positions_out=kwargs["positions_in"])
    try:
        with validating(table_op):
            table_op(**bad)
    except AssertionError as exc:
        assert "must not overlap" in str(exc), exc
    else:
        raise AssertionError("the table guard accepted positions_out aliasing positions_in")
    bad = dict(kwargs, draft_len=kwargs["draft_len"] + 1)
    try:
        with validating(table_op):
            table_op(**bad)
    except AssertionError as exc:
        assert "KV rewind" in str(exc), exc
    else:
        raise AssertionError("the table guard accepted a rewind with another draft length")
    spec = dict(glue_op.CELLS[0].spec)
    logits = [_logits(spec, seed, r) for r in range(WORLD)]
    kwargs, _ = _glue_call(spec, seed, logits, None)
    wide = torch.zeros(spec["rows"], spec["draft_len"] + 2, dtype=torch.int32, device="cuda")
    try:
        with validating(glue_op):
            glue_op(**dict(kwargs, next_new_tokens=wide))
    except AssertionError as exc:
        assert "next_new_tokens must be exactly" in str(exc), exc
    else:
        raise AssertionError("the glue guard accepted a wider next_new_tokens")


def _run_one_rank(entry: str, expected_cells: int) -> int:
    """Body of one rank: build the workspaces, run the entry's checks, exit hard on a failure."""
    global RANK, WORLD, HOST_GROUP, table_op, glue_op, validating, assert_within_ulp
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comm"))
    from _mnnvl_ranks import init_rank

    RANK, WORLD, mapping = init_rank()
    HOST_GROUP = dist.new_group(backend="gloo")

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.spec import (  # noqa: E501
        _draft_exchange,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.spec.draft_ehproj_table import (  # noqa: E501
        glm_draft_ehproj_table,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.spec.draft_sample_glue import (  # noqa: E501
        glm_draft_sample_glue,
    )

    table_op, glue_op = glm_draft_ehproj_table, glm_draft_sample_glue
    # Spawned, not collected: reach the tree's _validating the plain way. tests/unittest/_torch/modeling_v2 is
    # four levels above this directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import assert_within_ulp as _within
    from _validating import validating as _validating_cm

    validating, assert_within_ulp = _validating_cm, _within

    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    if WORLD == 8:
        assert MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16), (
            "TP8 requires the actual MNNVL capability"
        )
    assert _draft_exchange.ehproj_table_supports_tp(WORLD), (
        f"the table op does not serve {WORLD} ranks"
    )
    WORKSPACES["table"] = _draft_exchange.DraftExchangeWorkspace.for_ehproj_table(mapping)
    WORKSPACES["glue"] = _draft_exchange.DraftExchangeWorkspace.for_sample_glue(mapping)
    assert WORKSPACES["table"].buffer.data_ptr() != WORKSPACES["glue"].buffer.data_ptr()

    checks = {
        "draft_ehproj_table": (check_table_cells,),
        "draft_sample_glue": (check_glue_cells,),
    }[entry] + (
        check_draft_loop_eager_and_graph,
        check_tp8_worker_feedback_eager_and_graph,
        check_guards_refuse,
    )
    for check in checks:
        try:
            ran = check()
            if check in (check_table_cells, check_glue_cells) and expected_cells >= 0:
                assert ran == expected_cells, (
                    f"ran {ran} cells, the entry declares {expected_cells}"
                )
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard: a rank that leaves an exchange early leaves its peers polling.
            os._exit(13)
        dist.barrier(group=HOST_GROUP)
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {entry}: {len(checks)} checks passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks(entry: str, expected_cells: int) -> None:
    """One worker per device in CUDA_VISIBLE_DEVICES; every session is killed on a failure or the deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    devices = [d for d in visible.split(",") if d.strip()]
    world_size = len(devices)
    assert world_size >= 2, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); an exchange needs 2"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )
    # The children rendezvous through torch.distributed's env://, not the launching step's PMI namespace.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world_size} ranks for {entry}, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                entry,
                str(expected_cells),
                _WORKER_FLAG,
            ],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
            start_new_session=True,
        )
        for r in range(world_size)
    ]

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for r in sorted(live):
            code = children[r].poll()
            if code is None:
                continue
            live.discard(r)
            if code != 0:
                failure = f"rank {r} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        _kill_all()
        raise AssertionError(f"the {world_size}-rank {entry} run failed: {failure}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != _WORKER_FLAG]
    entry_name = args[0]
    cells = int(args[1]) if len(args) > 1 else -1
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(entry_name, cells))
    _spawn_ranks(entry_name, cells)
    print("OK")
