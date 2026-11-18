# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers of the ULL target's 16-rank GPU tests (GB300 or Rubin, 4 nodes x 4 GPUs).

The tests run under one MPI world of 16 ranks, every rank running the same pytest session
(``srun -N4 --ntasks-per-node=4 python -m pytest ...``): each test's collectives line up across
ranks because every rank runs the same tests in the same order. Outside such a world they skip.

* :func:`world`: ``(comm, rank)``; skips unless 16 ranks on SM103 / SM107 GPUs.
* :func:`text_model_config`: the text model's ``ModelConfig`` of the Kimi K3 NVFP4 checkpoint
  (``$K3_CKPT``: its directory; only ``config.json`` / ``hf_quant_config.json`` are read), as the
  target's text model sees it: the Helix CP16 mapping repurposed as TP16, the CP original in
  ``_helix_mapping_with_cp``, the quantization exclude list remapped from ``language_model.``;
  optionally the serving KV cache dtype (the checkpoint has none: BF16 KV).
* :func:`fill`: deterministic finite values for every parameter of a module (the unit input and
  KV-cache scales stay 1, as a checkpoint without them loads).
* :func:`post_load`: the model loader's post-load walk over a module.
* :func:`finalize_mla`: an MLA runtime's post-load (FP8 projections swapped for the weight-read
  GEMM, then the post-load walk) with the target's own loader helpers.
* :class:`MlaCache`: a KV cache manager V2 of MLA latent pages with random context rows on every
  rank and the Helix generation metadata of one step (``q`` tokens per request).
"""

from __future__ import annotations

import copy
import os
import zlib
from types import SimpleNamespace as NS
from typing import Tuple

import pytest
import torch

WORLD = 16
GPUS_PER_NODE = 4
_PKG = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl"
T = f"{_PKG}.kimi_k3_nvfp4__sm_107__helix16_moetp16ep1"


def world() -> Tuple[object, int]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (10, 7),
    ):
        pytest.skip("needs SM103 or SM107 GPUs")
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() != WORLD:
        pytest.skip(f"needs an MPI world of {WORLD} ranks (srun -N4 --ntasks-per-node=4)")
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return comm, rank


def checkpoint_dir() -> str:
    path = os.environ.get("K3_CKPT")
    if not path or not os.path.isfile(os.path.join(path, "config.json")):
        pytest.skip("set K3_CKPT to the Kimi K3 NVFP4 checkpoint directory")
    return path


def helix_mapping(rank: int):
    from tensorrt_llm.mapping import CpType, Mapping

    return Mapping(
        world_size=WORLD,
        rank=rank,
        gpus_per_node=GPUS_PER_NODE,
        tp_size=1,
        cp_size=WORLD,
        # The deployment's Helix all-to-all: the FIFO (v2) kernel, not NCCL.
        cp_config={"cp_type": CpType.HELIX, "use_nccl_for_alltoall": False, "fifo_version": 2},
        moe_tp_size=WORLD,
        moe_ep_size=1,
    )


def text_model_config(rank: int, num_layers=None, spec_config=None, kv_cache_quant_algo=None):
    """``(model_config, text_config, mapping_cp)``; ``kv_cache_quant_algo`` (a ``QuantAlgo``)
    sets the KV cache dtype serving selects (``kv_cache_config.dtype``), e.g. ``NVFP4``."""
    from tensorrt_llm._torch.model_config import ModelConfig

    mapping_cp = helix_mapping(rank)
    vlm = ModelConfig.from_pretrained(checkpoint_dir(), trust_remote_code=True, mapping=mapping_cp)
    cfg = copy.deepcopy(vlm.pretrained_config.text_config)
    if num_layers is not None:
        cfg.num_hidden_layers = num_layers
    mc = copy.copy(vlm)
    mc._frozen = False
    mc.pretrained_config = cfg
    mc.spec_config = spec_config
    mc.quant_config = copy.copy(mc.quant_config)
    if kv_cache_quant_algo is not None:
        mc.quant_config.kv_cache_quant_algo = kv_cache_quant_algo
    if mc.quant_config.exclude_modules:
        mapped = []
        for m in mc.quant_config.exclude_modules:
            if m.startswith("language_model."):
                m = m[len("language_model.") :]
                if m.startswith("layers."):
                    m = "model." + m
            mapped.append(m)
        mc.quant_config.exclude_modules = mapped
    repurposed = mapping_cp.repurpose_helix_cp_to_tp()
    repurposed.moe_tp_ep_user_specified = mapping_cp.moe_tp_ep_user_specified
    mc.mapping = repurposed
    mc._helix_mapping_with_cp = mapping_cp
    mc._frozen = True
    return mc, cfg, mapping_cp


# Static input / KV-cache scales the checkpoint does not carry: they load as 1.
_UNIT_SCALES = frozenset(
    ("input_scale", "inv_input_scale", "kv_scale_orig_quant", "kv_scale_quant_orig")
)


def _values(name: str, p: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=p.device)
    gen.manual_seed(zlib.crc32(name.encode()) * 7 + seed)
    if p.dtype == torch.float8_e4m3fn:
        return (torch.randn(p.shape, device=p.device, generator=gen) * 0.5).to(p.dtype)
    if p.dtype in (torch.int32, torch.uint8, torch.int8):
        return torch.zeros_like(p)
    if name.rsplit(".", 1)[-1] in _UNIT_SCALES:
        return torch.ones_like(p)
    if "scale" in name:
        return torch.exp2(
            torch.randint(-10, -6, p.shape, device=p.device, generator=gen).float()
        ).to(p.dtype)
    if "norm" in name:
        return (1 + 0.1 * torch.randn(p.shape, device=p.device, generator=gen)).to(p.dtype)
    return (0.02 * torch.randn(p.shape, device=p.device, generator=gen)).to(p.dtype)


@torch.no_grad()
def fill(module: torch.nn.Module, seed: int) -> None:
    """Deterministic finite values for every parameter (seeded by name and ``seed``); the
    module is then frozen, as the served model runs (no autograd)."""
    for name, p in module.named_parameters():
        p.data.copy_(_values(name, p.data, seed))
    module.requires_grad_(False)


def post_load(module: torch.nn.Module) -> None:
    """``ModelLoader._walk_full_post_load`` over ``module``: every submodule's
    ``post_load_weights``, parents before children (an MLA mixer reads kv_b's raw FP8 codes
    before kv_b's own transform; a swapped FP8 projection packs its scales here)."""
    for m in module.modules():
        post_load_weights = getattr(m, "post_load_weights", None)
        if post_load_weights is not None:
            post_load_weights()


def finalize_mla(runtime: torch.nn.Module, pkg: str) -> None:
    """The MLA runtime's post-load with ``pkg``'s (a target's) weight-read swap."""
    from importlib import import_module

    fwr = import_module(f"{pkg}.modeling")
    mixer = runtime.mixer
    for name, linear in list(mixer.named_children()):
        if fwr._is_fp8_block_projection(linear):
            fwr._swap_linear_to_fp8_weight_read(mixer, name, (type(linear),))
    post_load(runtime)
    runtime.requires_grad_(False)


class MlaCache:
    """A KV cache manager V2 of ``num_layers`` MLA latent layers (576 wide, ``dtype``) holding
    ``ctx_len`` random context tokens per request (``ctx_len / 16`` on each rank), and the Helix
    generation metadata of one step of ``q`` tokens per request (the last rank appends them).
    :meth:`snapshot` / :meth:`restore` keep the pages so two runs read the same cache."""

    def __init__(self, mapping_cp, *, batch, q, ctx_len, num_layers, dtype, seed=0):
        from tensorrt_llm._torch.attention.backends.interface import KVCacheParams
        from tensorrt_llm._torch.attention.backends.utils import get_attention_backend

        import_helix = _helix_utils()
        scenario = NS(
            ctx_len=ctx_len,
            batch=batch,
            kv_cache_tokens_per_block=64,
            num_layers=num_layers,
            kv_cache_dtype=dtype,
        )
        self.manager, _ = import_helix.setup_kv_and_metadata(
            scenario,
            mapping_cp,
            cache_type=import_helix.CACHE_TYPE_SELFKONLY,
            num_kv_heads=1,
            head_dim=576,
        )
        self.layers = num_layers
        gen = torch.Generator(device="cuda").manual_seed(seed * 101 + mapping_cp.rank)
        for layer in range(num_layers):
            buf = self.manager.get_buffers(layer)
            raw = buf.view(torch.uint8)
            # Finite FP8 codes (|x| < 2^8) or small BF16 values, both signs.
            vals = torch.randint(
                0, 0x70, raw.shape, device="cuda", generator=gen, dtype=torch.int32
            )
            if buf.dtype == torch.float8_e4m3fn:
                sign = torch.randint(0, 2, raw.shape, device="cuda", generator=gen) * 0x80
                raw.copy_((vals | sign).to(torch.uint8))
            else:
                buf.copy_(torch.randn(buf.shape, device="cuda", generator=gen).to(buf.dtype) * 0.1)
        per_rank = ctx_len // WORLD
        last = mapping_cp.cp_rank == WORLD - 1
        md = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor([q] * batch, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=0,
            prompt_lens=[per_rank] * batch,
            max_num_tokens=batch * q,
            kv_cache_manager=self.manager,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[per_rank] * batch
            ),
            mapping=mapping_cp,
        )
        md.enable_helix = True
        md.helix_is_inactive_rank = torch.tensor(
            [not last] * batch, dtype=torch.bool, device="cuda"
        )
        md.helix_is_inactive_rank_cpu = md.helix_is_inactive_rank.cpu().pin_memory()
        md.helix_position_offsets = torch.tensor(
            [ctx_len + t for _ in range(batch) for t in range(q)], dtype=torch.int, device="cuda"
        )
        md.helix_position_offsets_cpu = md.helix_position_offsets.cpu().pin_memory()
        if q > 1:
            md.update_spec_dec_param(
                batch_size=batch,
                is_spec_decoding_enabled=True,
                is_spec_dec_tree=False,
                is_spec_dec_dynamic_tree=False,
                max_draft_len=q - 1,
                max_total_draft_tokens=q - 1,
            )
        md.prepare()
        self.metadata = md
        self._saved = None

    def snapshot(self) -> None:
        self._saved = [self.manager.get_buffers(i).clone() for i in range(self.layers)]

    def restore(self) -> None:
        for i, saved in enumerate(self._saved):
            self.manager.get_buffers(i).copy_(saved)


def _helix_utils():
    import importlib.util
    import pathlib

    here = pathlib.Path(__file__).resolve()
    path = (
        here.parents[4] / "attention" / "multi_gpu" / "helix_test_utils.py"
    )  # tests/unittest/_torch
    spec = importlib.util.spec_from_file_location("_k3_ull_helix_test_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.uint8)


def finite(t: torch.Tensor) -> bool:
    """Every value finite: two NaN / inf outputs can have the same bits, so a bitwise match is
    evidence only where the reference is finite."""
    return bool(torch.isfinite(t.float()).all())


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def diff_summary(want: torch.Tensor, got: torch.Tensor) -> str:
    """How ``got`` differs from ``want``: differing elements, rows and the largest difference."""
    if want.shape != got.shape or want.dtype != got.dtype:
        return f"{tuple(got.shape)} {got.dtype} vs {tuple(want.shape)} {want.dtype}"
    ne = bits(want) != bits(got)
    rows = ne.reshape(want.shape[0], -1).any(dim=1).nonzero().flatten().tolist()
    gap = (want.float() - got.float()).abs().max().item()
    return (
        f"{int(ne.sum())}/{want.numel()} differ in rows {rows}, max |diff| {gap:.3g} "
        f"(reference max {want.float().abs().max().item():.3g})"
    )


def all_ranks(comm, ok: bool) -> bool:
    """Whether ``ok`` holds on every rank (every rank asserts the same)."""
    return all(comm.allgather(bool(ok)))
