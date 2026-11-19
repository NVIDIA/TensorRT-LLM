# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 workers of the sm_107 NVFP4 targets: acceptance and CUDA graph replay (one SM 10.3 /
10.7 GPU).

The rig (``build_rig``) builds a case's worker and drafter container on a small drafter of the
checkpoint's structure (``_draft_config``, ``_weights``), with the drafter's rows in the core's weight
table as the target's weight loader writes them and the unified-cache state of a generation step,
all in fixed tensors, so that a captured step reads the same storage at replay. ``step_inputs`` draws
one step's inputs and the draft cache it starts from, ``load_inputs`` copies them into the rig,
``run_step`` runs the worker's step, and ``snapshot`` / ``assert_same`` compare two runs bit for bit.

Each target's ``DFlashWorker`` (a subclass of the engine's) drives its drafter container on a toy
drafter of the checkpoint's structure. Its generation step is target-token acceptance
(``sample_and_accept_draft_tokens``), the fused preparation of the drafter's inputs from the unified
draft cache (``_prepare_paged_latent_gen_inputs``) and the drafter's block forward
(``dflash_forward``).

- Both acceptance paths, greedy strict acceptance and the sampled block verification, end in the
  forced-acceptance hook ``_apply_force_accepted_tokens(..., spec_metadata=...)``, once per step.
- A generation step captured in a CUDA graph on one set of inputs and replayed on another equals the
  eager step on the second set, outputs and draft cache alike. helix8's drafter runs across CP
  ranks, so one GPU captures no step of it.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch

__extra_import_path__ = [".."]
from _k31_utils import on_sm

# The checkpoint's rope_parameters.
_ROPE_PARAMS = dict(
    rope_type="yarn",
    rope_theta=50000.0,
    factor=32.0,
    original_max_position_embeddings=32768,
    beta_fast=32,
    beta_slow=1,
    mscale=1.0,
    mscale_all_dim=1.0,
)


# A small drafter of the checkpoint's structure.
_H, _NH, _QL, _KVL, _NOPE, _ROPE_D, _V, _I, _VOCAB, _L = 128, 6, 32, 512, 16, 64, 16, 256, 512, 2


# The window is a multiple of 4: the TRTLLM-Gen window attention reads index lists of 4k rows.
_WINDOW, _GROUP, _TAPS = 8, 16, 2


# The selector shape the CuTe DSL selector ops take: 32 candidates over a rank of whole 256 blocks.
_RANK = 256


def _draft_config(block_size=5, selector_rank=_RANK):
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

    cfg = PretrainedConfig(
        architectures=["DFlash2DraftModel"],
        hidden_size=_H,
        num_attention_heads=_NH,
        num_key_value_heads=_NH,
        q_lora_rank=_QL,
        kv_lora_rank=_KVL,
        qk_nope_head_dim=_NOPE,
        qk_rope_head_dim=_ROPE_D,
        v_head_dim=_V,
        intermediate_size=_I,
        hidden_act="silu",
        num_hidden_layers=_L,
        rms_norm_eps=1e-5,
        vocab_size=_VOCAB,
        max_position_embeddings=4096,
        rope_parameters=dict(_ROPE_PARAMS),
        rope_theta=50000.0,
        sliding_window=_WINDOW,
        use_sliding_window=True,
        layer_types=["sliding_attention"] * _L,
        is_causal=True,
        dflash_config=dict(
            attention_mode="mla",
            block_size=block_size,
            conv_kernel_size=_TAPS,
            conv_group_size=_GROUP,
            selector_rank=selector_rank,
            # The candidates per slot that the verification kernels hold
            # (k31_dflash2_proposal_load's TOP_K, the targets' DRAFT_DFLASH selector_top_k).
            selector_top_k=32,
            mask_token_id=_VOCAB - 1,
            target_layer_ids=[0, 1],
        ),
    )
    cfg.model_type = "qwen3"
    cfg.torch_dtype = torch.bfloat16
    return ModelConfig(pretrained_config=cfg, mapping=Mapping(world_size=1, tp_size=1, rank=0))


def _weights(seed=0, selector_rank=_RANK):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=None):
        scale = scale if scale is not None else 1.0 / math.sqrt(shape[-1])
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    def norm_w(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    w = {}
    for i in range(_L):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = norm_w(_H)
        w[p + "post_attention_layernorm.weight"] = norm_w(_H)
        w[p + "self_attn.q_a_proj.weight"] = rnd(_QL, _H)
        w[p + "self_attn.q_a_layernorm.weight"] = norm_w(_QL)
        w[p + "self_attn.q_b_proj.weight"] = rnd(_NH * (_NOPE + _ROPE_D), _QL)
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = rnd(_KVL + _ROPE_D, _H)
        w[p + "self_attn.kv_a_layernorm.weight"] = norm_w(_KVL)
        w[p + "self_attn.kv_b_proj.weight"] = rnd(_NH * (_NOPE + _V), _KVL)
        w[p + "self_attn.o_proj.weight"] = rnd(_H, _NH * _V)
        w[p + "mlp.gate_proj.weight"] = rnd(_I, _H)
        w[p + "mlp.up_proj.weight"] = rnd(_I, _H)
        w[p + "mlp.down_proj.weight"] = rnd(_H, _I)
        for conv in ("attention_conv", "mlp_conv"):
            base = 0.2 * torch.randn(2, _TAPS, _H, generator=g)
            base[:, 0] += 1.0
            w[p + conv + ".base_kernel"] = base.to(torch.bfloat16)
            w[p + conv + ".kernel_projection.weight"] = rnd(
                2 * _TAPS * _H // _GROUP, _H, scale=0.02
            )
    w["fc.weight"] = rnd(_H, 2 * _H)
    w["hidden_norm.weight"] = norm_w(_H)
    w["norm.weight"] = norm_w(_H)
    w["candidate_selector.predecessor_codebook"] = rnd(_VOCAB, selector_rank)
    w["candidate_selector.successor_codebook"] = rnd(_VOCAB, selector_rank)
    w["candidate_selector.hidden_projection.weight"] = rnd(selector_rank, _H)
    return w


GPU = on_sm(103, 107, reason="the DFlash2 worker's kernels are certified on sm_103 / sm_107")


_PACKAGE = "tensorrt_llm._torch.modeling_v2.models.kimi_k31"


# The worker and the drafter container every specialized target's modeling.py defines.
WORKER, CONTAINER = "DFlashWorker", "DraftModel"


# The keyword arguments the worker calls dflash_forward with.
DFLASH_FORWARD_KW = (
    "noise_embedding",
    "query_positions",
    "num_ctx_per_req",
    "ctx_k_cache",
    "ctx_v_cache",
    "ctx_cache_batch_idx",
    "ctx_kv_cache",
    "ctx_page_table",
    "block_placement",
)


# The toy drafter's candidate-selector rank.
_SELECTOR_RANK = 8


_PAGE = 32


# Three requests and one CUDA-graph padding row (the shared dummy slot).
BATCH, DUMMY_SLOT, _PAGES_PER_ROW = 4, 3, 6


FP8 = torch.float8_e4m3fn


# The context lengths of the requests of two steps' inputs.
A_CTX, B_CTX = (30, 63, 97), (33, 64, 150)


@dataclass(frozen=True)
class Case:
    """A worker of a target: its drafter container, and whether its drafter runs on a helix
    group."""

    target: str
    helix: bool = False
    container: str = CONTAINER

    @property
    def id(self) -> str:
        name = self.target.rsplit("__", 1)[-1]
        return name if self.container == CONTAINER else f"{name}-{self.container}"


# Every specialized SM107 target that ships a DFlash2 worker.
WORKERS = (
    Case("kimi_k31_pro_nvfp4__sm_107__dep16"),
    Case("kimi_k31_pro_nvfp4__sm_107__helix8", helix=True),
    Case("kimi_k31_pro_nvfp4__sm_107__tep8"),
)


def case(target: str) -> Case:
    """``target``'s worker (its ``WORKERS`` entry). helix8's drafter runs across CP ranks, so one
    GPU captures no step of it (``Case.helix``)."""
    (found,) = [c for c in WORKERS if c.target == target]
    return found


class _ToyRt(NamedTuple):
    """What a drafter reads off the target's runtime (``core._rt``): the TP group's all-reduce, and
    ``draft``, which its ``bind_runtime`` replaces with the drafter's own."""

    all_reduce: Any
    draft: Any


class _ToyCore(torch.nn.Module):
    """The target core a specialized drafter is built on, with what the drafter reads off it: the
    target's model config (``spec_config``), the helix mapping (``cp_mapping``, the model
    config's), the per-rank widths (``dims.kda_tp``, ``dims.draft_context_kv_only``), the weight
    table ``w`` (the embedding the drafter shares and the drafter's own ``d_*`` rows) and the
    runtime (``_rt``)."""

    def __init__(
        self, embed: torch.Tensor, model_config: Any, draft_rows: dict, history_only: bool
    ) -> None:
        super().__init__()
        from tensorrt_llm._torch.distributed import AllReduce

        self.model_config = model_config
        self.cp_mapping = model_config.mapping
        self.dims = SimpleNamespace(
            kda_tp=model_config.mapping.tp_size, draft_context_kv_only=history_only
        )
        rows = {"embed": embed, **draft_rows}
        self.w = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v, requires_grad=False) for k, v in rows.items()}
        )
        # One GPU: the all-reduce of a one-rank TP group returns its input.
        self._rt = _ToyRt(AllReduce(mapping=model_config.mapping, dtype=torch.bfloat16), None)


def _draft_rows(history_only: bool) -> dict:
    """The drafter's ``core.w["d_*"]`` rows from the toy checkpoint, in the layout the target's
    weight loader (``w3_rows``) writes on one rank: ``d_kv_a`` and ``d_kv_a_norm`` stack every
    layer's latent projection and norm, ``kv_b`` holds every head's ``k_nope`` rows then every
    head's ``v`` rows, ``gate_up`` the gate rows then the up rows. A history-only drafter
    (``context_kv_only``) has the history's rows only."""
    w = _weights(selector_rank=_SELECTOR_RANK)
    layers = range(_L)
    rows = {
        "d_fc": w["fc.weight"],
        "d_hidden_norm": w["hidden_norm.weight"],
        "d_kv_a": torch.cat([w[f"layers.{j}.self_attn.kv_a_proj_with_mqa.weight"] for j in layers]),
        "d_kv_a_norm": torch.stack(
            [w[f"layers.{j}.self_attn.kv_a_layernorm.weight"] for j in layers]
        ),
    }
    if not history_only:
        rows["d_final_norm"] = w["norm.weight"]
        rows["d_sel_pred"] = w["candidate_selector.predecessor_codebook"]
        rows["d_sel_succ"] = w["candidate_selector.successor_codebook"]
        rows["d_sel_hidden_proj"] = w["candidate_selector.hidden_projection.weight"]
        for j in layers:
            p, d = f"layers.{j}.", f"d_l{j}_"
            kv_b = w[p + "self_attn.kv_b_proj.weight"].view(_NH, _NOPE + _V, _KVL)
            rows[d + "in_norm"] = w[p + "input_layernorm.weight"]
            rows[d + "post_norm"] = w[p + "post_attention_layernorm.weight"]
            rows[d + "q_a"] = w[p + "self_attn.q_a_proj.weight"]
            rows[d + "q_a_norm"] = w[p + "self_attn.q_a_layernorm.weight"]
            rows[d + "q_b"] = w[p + "self_attn.q_b_proj.weight"]
            rows[d + "kv_b"] = torch.cat(
                (kv_b[:, :_NOPE].reshape(-1, _KVL), kv_b[:, _NOPE:].reshape(-1, _KVL))
            )
            rows[d + "o"] = w[p + "self_attn.o_proj.weight"]
            rows[d + "gate_up"] = torch.cat(
                (w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"])
            )
            rows[d + "down"] = w[p + "mlp.down_proj.weight"]
            rows[d + "attn_conv_base"] = w[p + "attention_conv.base_kernel"]
            rows[d + "attn_conv_proj"] = w[p + "attention_conv.kernel_projection.weight"]
            rows[d + "mlp_conv_base"] = w[p + "mlp_conv.base_kernel"]
            rows[d + "mlp_conv_proj"] = w[p + "mlp_conv.kernel_projection.weight"]
    return {k: v.contiguous().cuda() for k, v in rows.items()}


def _target_mapping(case: Case):
    """The target's Mapping on this one GPU: one rank, of helix8's CP8 helix group for a helix
    case (the drafter shards its history by the target's helix mapping)."""
    from tensorrt_llm.mapping import CpType, Mapping

    if case.helix:
        return Mapping(world_size=8, rank=0, cp_size=8, cp_config={"cp_type": CpType.HELIX})
    return Mapping(world_size=1, tp_size=1, rank=0)


def target_module(case: Case):
    """The case's target ``modeling`` module."""
    return importlib.import_module(f"{_PACKAGE}.{case.target}.modeling")


def build_container(case: Case, module, block: int, monkeypatch, *, history_only=False):
    """The case's drafter container at toy size, with seeded weights, on the GPU."""
    draft_config = _draft_config(block, selector_rank=_SELECTOR_RANK)
    g = torch.Generator().manual_seed(7)
    embed = (torch.randn(_VOCAB, _H, generator=g) * 0.5).to(torch.bfloat16).cuda()
    spec = SimpleNamespace(max_draft_len=block - 1, attention_backend="VANILLA")
    model_config = SimpleNamespace(spec_config=spec, extra_attrs={}, mapping=_target_mapping(case))
    # The drafter holds no weight of its own: its rows sit in the core's table, as the target's
    # weight loader fills them.
    core = _ToyCore(embed, model_config, _draft_rows(history_only), history_only)
    # As the shell builds it: the drafter's ModelConfig from draft_config_for, then the container
    # on the core.
    monkeypatch.setattr(module, "draft_config_for", lambda _: draft_config)
    lm_head = torch.nn.Linear(_H, _VOCAB, bias=False, dtype=torch.bfloat16)
    try:
        container = module.DraftModel(core, module.draft_config_for(model_config), lm_head, None)
    except (TypeError, AttributeError) as error:
        pytest.fail(
            f"{case.target}.DraftModel no longer builds from (core, model_config, lm_head, "
            f"logits_processor): {error}; update build_container to its constructor"
        )
    container = container.cuda()
    # A drafter may hold the target core by weak reference only (it shares the embedding).
    object.__setattr__(container, "_contract_core", core)
    embed_tokens = container.draft_model_full.model.embed_tokens
    if embed_tokens is not None and hasattr(embed_tokens, "weight"):
        with torch.no_grad():
            embed_tokens.weight.copy_(embed)
    # The worker sizes the drafter's window workspace for its largest step.
    container._max_block_rows = BATCH * block
    return container


def build_worker(case: Case, module, container, k: int):
    """The case's worker with draft length ``k``, driving ``container``."""
    from tensorrt_llm.mapping import Mapping

    spec_config = SimpleNamespace(
        max_draft_len=k,
        attention_backend="VANILLA",
        use_rejection_sampling=False,
        tokens_per_gen_step=k + 1,
    )
    worker = getattr(module, WORKER)(spec_config, Mapping(world_size=1, tp_size=1, rank=0))
    worker.set_draft_model(container)
    return worker


@dataclass
class Rig:
    """One worker, its toy drafter, the unified-cache state of a generation step, and the
    step's inputs, all in fixed tensors: a captured step reads the same storage at replay."""

    worker: Any
    container: Any
    k: int
    pools: list
    logits: torch.Tensor
    draft_tokens: torch.Tensor
    captured: torch.Tensor
    slots: torch.Tensor
    attn: SimpleNamespace
    spec: SimpleNamespace

    @property
    def block(self) -> int:
        return self.k + 1

    def state(self) -> dict:
        w = self.worker
        return {
            "ctx_len": w._ctx_len,
            "ctx_position_offset": w._ctx_position_offset,
            "seq_lens": self.attn._seq_lens_cuda,
            **{f"pool{i}": p for i, p in enumerate(self.pools)},
        }


def build_rig(case: Case, k: int, pool_dtype: torch.dtype, monkeypatch, force: int = 0) -> Rig:
    """The case's worker at draft length ``k`` on a ``pool_dtype`` draft cache, forcing ``force``
    accepted tokens (0: natural acceptance), bound to the unified draft cache of ``BATCH`` slots."""
    torch.manual_seed(0)
    module = target_module(case)
    block = k + 1
    container = build_container(case, module, block, monkeypatch)
    worker = build_worker(case, module, container, k)
    worker.force_num_accepted_tokens = float(force)
    dev = torch.device("cuda")
    pages = BATCH * _PAGES_PER_ROW
    pools = [
        torch.zeros(pages, 1, 1, _PAGE, _KVL + _ROPE_D, dtype=pool_dtype, device=dev)
        for _ in range(_L)
    ]
    # The unified draft cache as _lazy_init_ctx_buffers binds it (slots 0..2, dummy slot 3).
    worker._ctx_kv_manager = SimpleNamespace(has_standalone_draft_cache=True)
    worker._ctx_buf_inited = True
    worker._compute_block_size = block
    worker._dummy_slot = DUMMY_SLOT
    worker._max_ctx = 1024
    worker._ctx_page_size = _PAGE
    worker._ctx_len = torch.zeros(BATCH, dtype=torch.long, device=dev)
    worker._ctx_position_offset = torch.zeros(BATCH, dtype=torch.long, device=dev)
    # No pages below the slots' first stored positions: nothing to rebase.
    worker._ctx_base_pages = torch.zeros(BATCH, dtype=torch.long, device=dev)
    worker._ctx_block_tables = torch.zeros(BATCH, _PAGES_PER_ROW, dtype=torch.int32, device=dev)
    worker._ctx_block_counts = torch.full((BATCH,), _PAGES_PER_ROW, dtype=torch.long, device=dev)
    # The rows the block decode reads, shifted past each slot's base (written each step).
    worker._ctx_block_tables_read = torch.zeros_like(worker._ctx_block_tables)
    worker._ctx_kv_buf = pools
    worker._ctx_kv_ptrs = torch.tensor([p.data_ptr() for p in pools], dtype=torch.int64, device=dev)
    worker._ctx_k_buf = worker._ctx_v_buf = None
    slots = torch.zeros(BATCH, dtype=torch.long, device=dev)
    worker._batch_to_slot_buffer = SimpleNamespace(tensor=slots)
    # Then the drafter's runtime, as the worker's _lazy_init_ctx_buffers binds it.
    container.bind_runtime(BATCH * worker._compute_block_size, worker._ctx_page_size)
    rows = BATCH * block
    captured = torch.zeros(rows, 2 * _H, dtype=torch.bfloat16, device=dev)
    attn = SimpleNamespace(
        num_seqs=BATCH,
        num_contexts=0,
        num_ctx_tokens=0,
        _seq_lens_cuda=torch.zeros(BATCH, dtype=torch.int32, device=dev),
        _seq_lens=torch.zeros(BATCH, dtype=torch.int32),
    )
    draft_tokens = torch.zeros(BATCH * k, dtype=torch.int32, device=dev)
    spec = SimpleNamespace(
        runtime_draft_len=k,
        draft_tokens=draft_tokens,
        is_ragged_verify=False,
        enable_penalty=False,
        use_rejection_sampling=False,
        is_all_greedy_sample=True,
        is_cuda_graph=False,
        batch_slot_ids=None,
        dummy_slot_row=0,
        max_num_requests=BATCH,
        captured_hidden_states=captured,
        get_hidden_states=lambda n: captured[:n],
    )
    return Rig(
        worker=worker,
        container=container,
        k=k,
        pools=pools,
        logits=torch.zeros(rows, _VOCAB, dtype=torch.float32, device=dev),
        draft_tokens=draft_tokens,
        captured=captured,
        slots=slots,
        attn=attn,
        spec=spec,
    )


def step_inputs(rig: Rig, seed: int, ctx_lens: tuple[int, int, int]) -> dict:
    """One generation step's inputs and the cache state it starts from (host tensors)."""
    g = torch.Generator().manual_seed(seed)
    k, block = rig.k, rig.block
    logits = torch.randn(BATCH * block, _VOCAB, generator=g) * 3.0
    target = logits.view(BATCH, block, _VOCAB).argmax(-1)
    drafts = torch.randint(0, _VOCAB, (BATCH, k), generator=g)
    for row in range(BATCH):
        agree = (seed + row) % (k + 1)  # the leading drafts that match: 0 .. k accepted
        drafts[row, :agree] = target[row, :agree]
    real = torch.randperm(DUMMY_SLOT, generator=g)
    ctx_len = torch.zeros(BATCH, dtype=torch.long)
    ctx_len[real] = torch.tensor(ctx_lens, dtype=torch.long)
    pools = []
    for _ in range(_L):
        latent = torch.randn(BATCH * _PAGES_PER_ROW, 1, 1, _PAGE, _KVL + _ROPE_D, generator=g)
        pools.append(latent.to(torch.bfloat16).to(rig.pools[0].dtype))
    host = {
        "logits": logits,
        "draft_tokens": drafts.reshape(-1).to(torch.int32),
        "captured": (torch.randn(BATCH * block, 2 * _H, generator=g) * 0.5).to(torch.bfloat16),
        # Rows 0..2 are requests, row 3 the padding row on the dummy slot.
        "slots": torch.cat([real, torch.tensor([DUMMY_SLOT])]),
        "ctx_len": ctx_len,
        "ctx_position_offset": torch.zeros(BATCH, dtype=torch.long),
        "block_tables": torch.randperm(BATCH * _PAGES_PER_ROW, generator=g)
        .view(BATCH, _PAGES_PER_ROW)
        .to(torch.int32),
        "pools": pools,
    }
    # Pinned, so a step's inputs are staged asynchronously, as the executor stages them.
    return {
        name: [t.pin_memory() for t in value] if name == "pools" else value.pin_memory()
        for name, value in host.items()
    }


def load_inputs(rig: Rig, inputs: dict, names=None) -> None:
    """Copy inputs into the rig's fixed tensors, asynchronously."""
    w = rig.worker
    targets = {
        "logits": rig.logits,
        "draft_tokens": rig.draft_tokens,
        "captured": rig.captured,
        "slots": rig.slots,
        "block_tables": w._ctx_block_tables,
        "ctx_len": w._ctx_len,
        "ctx_position_offset": w._ctx_position_offset,
    }
    for name in names or (*targets, "pools"):
        if name == "pools":
            for dst, src in zip(rig.pools, inputs["pools"]):
                dst.copy_(src, non_blocking=True)
        else:
            targets[name].copy_(inputs[name], non_blocking=True)


def run_step(rig: Rig) -> dict:
    """One generation step of the worker: acceptance, the fused drafter-input preparation from
    the unified draft cache, the drafter's block forward."""
    w, c = rig.worker, rig.container
    accepted, num_accepted = w.sample_and_accept_draft_tokens(rig.logits, rig.attn, rig.spec)
    inputs = w._prepare_paged_latent_gen_inputs(
        accepted,
        num_accepted,
        rig.attn,
        rig.spec,
        c,
        BATCH * rig.block,
        c.mask_token_id,
        getattr(c, "mask_token_embedding", None),
    )
    hidden = c.dflash_forward(**{name: inputs[name] for name in DFLASH_FORWARD_KW})
    return {
        "accepted": accepted,
        "num_accepted": num_accepted,
        "num_ctx": inputs["num_ctx_per_req"],
        "query_positions": inputs["query_positions"],
        "hidden": hidden,
    }


def snapshot(rig: Rig, out: dict) -> dict:
    """Copies of a step's outputs and of the rig's state after it."""
    snap = {}
    for name, t in out.items():
        if name == "accepted":
            # Only each row's accepted prefix is defined; acceptance leaves the rest unwritten.
            defined = torch.arange(t.shape[1], device=t.device) < out["num_accepted"].unsqueeze(1)
            t = torch.where(defined, t, torch.full_like(t, -1))
        snap[f"out.{name}"] = t.clone()
    snap.update({f"state.{name}": t.clone() for name, t in rig.state().items()})
    return snap


def assert_same(got: dict, want: dict) -> None:
    """Two snapshots are equal bit for bit, and the dummy slot holds no context."""
    assert got.keys() == want.keys()
    for name in want:
        a, b = got[name], want[name]
        if a.dtype == FP8:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        assert torch.equal(a, b), f"{name} differs"
    dummy = want["state.ctx_len"][DUMMY_SLOT].item()
    assert dummy == 0, f"the dummy slot holds a context of {dummy}"


def sampled_metadata(rig: Rig) -> SimpleNamespace:
    """The rig's step metadata for the sampled block verification: temperature 1, no top-k /
    top-p / min-p, fixed per-request seeds."""
    from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

    rows = BATCH * rig.block
    dev = rig.logits.device
    return SimpleNamespace(
        **{
            **vars(rig.spec),
            **dict(
                use_rejection_sampling=True,
                is_all_greedy_sample=False,
                advanced_sampling_mode=AdvancedSamplingMode.FULL,
                temperatures=torch.ones(rows, device=dev),
                top_ks=torch.zeros(rows, dtype=torch.int32, device=dev),
                top_ps=torch.ones(rows, device=dev),
                min_ps=torch.zeros(rows, device=dev),
                request_temperatures=torch.ones(BATCH, device=dev),
                request_seeds=torch.arange(BATCH, dtype=torch.int64, device=dev) + 1234,
                request_offsets=torch.zeros(BATCH, dtype=torch.int64, device=dev),
                rng_accept_slot=0,
                batch_slot_ids=torch.arange(BATCH, dtype=torch.int64, device=dev),
                draft_probs=None,
                num_seq_slots=BATCH,
            ),
        }
    )


_K = 7


def _target(layout: str) -> str:
    return f"kimi_k31_pro_nvfp4__sm_107__{layout}"


@GPU
@pytest.mark.parametrize("layout", ["dep16", "helix8", "tep8"])
def test_both_accept_paths_call_the_force_hook(layout, monkeypatch):
    rig = build_rig(case(_target(layout)), _K, torch.bfloat16, monkeypatch, force=_K)
    load_inputs(rig, step_inputs(rig, 11, A_CTX))
    calls = []
    inner = rig.worker._apply_force_accepted_tokens

    def hook(num_accepted, num_contexts, runtime_draft_len, spec_metadata=None):
        calls.append((runtime_draft_len, spec_metadata))
        return inner(num_accepted, num_contexts, runtime_draft_len, spec_metadata=spec_metadata)

    rig.worker._apply_force_accepted_tokens = hook

    def accept(name, spec):
        calls.clear()
        _, num = rig.worker.sample_and_accept_draft_tokens(rig.logits, rig.attn, spec)
        assert len(calls) == 1, f"{name} called the hook {len(calls)} times"
        assert calls[0][0] == _K and calls[0][1] is spec
        assert torch.equal(num.cpu(), torch.full((BATCH,), _K + 1, dtype=num.dtype))

    accept("greedy acceptance", rig.spec)
    accept("sampled block verification", sampled_metadata(rig))


@GPU
@pytest.mark.parametrize("layout", ["dep16", "tep8"])
def test_replay_on_new_inputs_equals_eager(layout, monkeypatch):
    rig = build_rig(case(_target(layout)), _K, FP8, monkeypatch, 0)
    a, b = step_inputs(rig, 1, A_CTX), step_inputs(rig, 2, B_CTX)
    load_inputs(rig, a)
    run_step(rig)
    torch.cuda.synchronize()
    load_inputs(rig, a)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run_step(rig)
    torch.cuda.synchronize()
    load_inputs(rig, b)
    graph.replay()
    torch.cuda.synchronize()
    replayed = snapshot(rig, out)
    load_inputs(rig, b)
    eager = snapshot(rig, run_step(rig))
    assert_same(replayed, eager)
