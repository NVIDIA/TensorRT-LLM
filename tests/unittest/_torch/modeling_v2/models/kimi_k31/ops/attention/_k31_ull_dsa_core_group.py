# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA attention core's 16 Helix ranks on one GPU, for the dsa_sparse_dcp_decode and dsa_dcp_value
tests: one real ``dsa.FlatDsaCore`` per rank (the served compile: push (6, 16), cap 8), each over
its own q mailbox, with the DcpMailbox peer rendezvous replaced by 16 mailboxes on this GPU whose
peer address tables list each other (the kernels only read those tables)."""

from __future__ import annotations

import math
import zlib
from types import SimpleNamespace

import torch

WORLD, LOCAL, HEADS, ROW, LATENT, TOPK, CAP = 16, 6, 96, 576, 512, 2048, 64
PACKED_PITCH, GATE_COLUMN = 3040, 2272
LAYER, PARITY = 3, 1
SERVED_ROWS = (1, 4, 8)

_TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"


def _modules():
    import importlib

    dsa = importlib.import_module(_TARGET + ".modeling")
    main = importlib.import_module(_TARGET.rpartition(".")[0] + ".kernels.vendor.vllm_main")
    return dsa, main.module("cutedsl.mla.dcp"), main.module("cutedsl.common.host")


class _Mail:
    """``DcpMailbox("bf16")`` of one rank without the rendezvous; ``peers`` set by the group."""

    def __init__(self, dcp, cap):
        self.format = "bf16"
        self.partial = torch.empty((2, WORLD, cap, LOCAL, LATENT), dtype=torch.bfloat16,
                                   device="cuda")  # fmt: skip
        self.lse = torch.empty((2, WORLD, cap, LOCAL), dtype=torch.float32, device="cuda")
        self.partial.view(torch.int32).fill_(dcp.PAIR_SENT - (1 << 32))
        self.lse.view(torch.int32).fill_(-1)
        self.peers = self.lse_peers = None

    def armed(self, dcp) -> bool:
        return bool(
            (self.partial.view(torch.int32) == dcp.PAIR_SENT - (1 << 32)).all()
            and (self.lse.view(torch.int32) == -1).all()
        )


class CoreGroup:
    def __init__(self, monkeypatch):
        dsa, dcp, host = _modules()
        self.dsa, self.dcp, self.host = dsa, dcp, host
        g = torch.Generator(device="cuda").manual_seed(96)
        cap = max(SERVED_ROWS)
        self.mails = [_Mail(dcp, cap) for _ in range(WORLD)]
        i64 = dict(dtype=torch.int64, device="cuda")
        peers = torch.tensor([m.partial.data_ptr() for m in self.mails], **i64)
        lse_peers = torch.tensor([m.lse.data_ptr() for m in self.mails], **i64)
        for m in self.mails:
            m.peers, m.lse_peers = peers, lse_peers
        self.v_b = (torch.randn(HEADS, 128, LATENT, generator=g, device="cuda") * 0.05).bfloat16()
        self.cores = []
        for rank in range(WORLD):
            mailbox = torch.full((2, CAP, HEADS, ROW), 0xFF, dtype=torch.uint8, device="cuda")
            spins = torch.zeros(4, dtype=torch.int32, device="cuda")
            query = SimpleNamespace(
                mailbox=mailbox,
                group=None,
                parity={LAYER: PARITY},
                spins=spins,
                spins_view=host.to_cute(spins, 4),
            )
            mail = self.mails[rank]
            monkeypatch.setattr(dcp, "DcpMailbox", lambda *args, _m=mail: _m)
            core = dsa.FlatDsaCore(
                mla_query=query, v_b={LAYER: self.v_b}, rank=rank, world=WORLD,
                rows=SERVED_ROWS, device=torch.device("cuda", torch.cuda.current_device()),
            )  # fmt: skip
            self.cores.append(core)
        monkeypatch.undo()

    def step(self, rows: int, kept: tuple[int, ...], seed: int):
        """One step's inputs: the gathered FP8 query in every rank's mailbox, each rank's KV pool
        rows, kept rows / counts (rank-dependent counts, the cell's zero rows zero everywhere),
        the gate rows and the outputs."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        q = (torch.randn(rows, HEADS, ROW, generator=g, device="cuda") * 0.5).to(
            torch.float8_e4m3fn
        )
        steps = []
        for rank, core in enumerate(self.cores):
            core.mla_query.mailbox[PARITY, :rows].copy_(q.view(torch.uint8))
            pool = 3000 + 64 * rank
            kv = (torch.randn(pool, ROW, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
            indices = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
            counts = []
            for b in range(rows):
                n = kept[b] if (rank + b) % 3 else kept[b] // 2
                n = min(n, pool)
                indices[b, :n] = torch.randperm(pool, generator=g, device="cuda")[:n].int()
                counts.append(n)
            valid = torch.tensor(counts, dtype=torch.int32, device="cuda")
            packed = torch.randn(rows, PACKED_PITCH, generator=g, device="cuda").bfloat16()
            steps.append(
                SimpleNamespace(
                    kv=kv.view(torch.uint8),
                    indices=indices,
                    valid=valid,
                    gate=packed[:, GATE_COLUMN:],
                    out=torch.full((rows, LOCAL * 128), float("nan"), device="cuda").bfloat16(),
                    latent=core.latent[:rows],
                    lse=core.lse[:rows],
                )
            )
        return q, steps

    def stream(self):
        return self.host.stream(torch.device("cuda", torch.cuda.current_device()))


def seed_of(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


def union_reference(group: CoreGroup, q, steps, rows):
    """FP32 end to end: per owner, softmax attention of its heads over every rank's kept rows,
    projected by W_UV and gated (no intermediate rounding)."""
    scale = group.cores[0].scale_log2 * math.log(2.0)
    outs = []
    qf = q.float()
    for owner, core in enumerate(group.cores):
        heads = range(LOCAL * owner, LOCAL * (owner + 1))
        out = torch.zeros(rows, LOCAL, 128, device="cuda")
        for b in range(rows):
            keys = [s.kv[s.indices[b, : int(s.valid[b])].long()] for s in steps]
            k = torch.cat(keys).view(torch.float8_e4m3fn).float()
            if k.shape[0] == 0:
                continue
            for j, h in enumerate(heads):
                p = torch.softmax((qf[b, h] @ k.t()) * scale, dim=0)
                lat = p @ k[:, :LATENT]
                out[b, j] = lat @ core.w_uv[LAYER][j].float()
        g = torch.sigmoid(steps[owner].gate.float().reshape(rows, LOCAL, 128))
        outs.append((out * g).reshape(rows, LOCAL * 128))
    return outs
