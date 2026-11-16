# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_fused_moe.op import (
    _EXCHANGE_SLOT_WORDS,
    _SF_VEC,
    HIDDEN_SIZE,
    NUM_EXPERTS,
    TOP_K,
    _compile_key,
    _view,
)

_modules: dict[tuple, object] = {}
_compiled: dict[tuple, object] = {}


# ---------------------------------------------------------------------------------------------------------------------
# trtllm::k3_moe_m1 / trtllm::k3_moe_m2: the routed experts of one or two decode tokens as weight-stream kernels
# (k3_moe_m1_kernel.py, k3_moe_m2_kernel.py), on a caller-owned workspace.
# ---------------------------------------------------------------------------------------------------------------------
_ENGINE_KERNEL_PATHS = {
    "m1": os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_moe_m1_kernel.py"),
    "m2": os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_moe_m2_kernel.py"),
}
_M1_MIN_CTAS = HIDDEN_SIZE // 32  # one 32-row block of the down projection per CTA
# (alignment, leading dim) per tensor argument of k3_moe_m1 / k3_moe_m2, in _engine_args' order.
_ENGINE_ALIGNS = [16, 16, 16, 16, 16, 4, 2, 4, 16, 16, 4, 4, 2, 16, 4]
_ENGINE_LEADING = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def _engine_module(kind: str, config: dict):
    """One ``k3_moe_m1`` / ``k3_moe_m2`` module per configuration (shapes are trace-time constants), kept with
    k3_moe's."""
    key = (kind,) + tuple(sorted(config.items()))
    mod = _modules.get(key)
    if mod is None:
        name = f"{__name__}_{kind}_kernel_" + "_".join(f"{k}{v}" for k, v in key[1:])
        spec = importlib.util.spec_from_file_location(name, _ENGINE_KERNEL_PATHS[kind])
        mod = importlib.util.module_from_spec(spec)
        mod.K3_CONFIG = dict(config)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _modules[key] = mod
    return mod


def m1_supported(
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    local_num_experts: int,
    intermediate_size: int,
) -> Tuple[bool, str]:
    """Whether these TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers fit ``k3_moe_m1``. ``intermediate_size`` is a rank's
    logical intermediate; the buffers may hold it zero-padded to a multiple of 128, as TRT-LLM's loader lays out an
    unaligned shard (e.g. 192 -> 256 at TP16). Only metadata is read, so the buffers may still be on the meta
    device."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        return False, "needs sm_100 or sm_103"
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False, "CuTe DSL (nvidia-cutlass-dsl) is not installed"
    e, two_i, _ = w3_w1_weight.shape
    i_pad = two_i // 2
    expected = {
        "w3_w1_weight": (w3_w1_weight, (local_num_experts, two_i, HIDDEN_SIZE // 2)),
        "w3_w1_weight_scale": (
            w3_w1_weight_scale,
            (local_num_experts, two_i, HIDDEN_SIZE // _SF_VEC),
        ),
        "w2_weight": (w2_weight, (local_num_experts, HIDDEN_SIZE, i_pad // 2)),
        "w2_weight_scale": (w2_weight_scale, (local_num_experts, HIDDEN_SIZE, i_pad // _SF_VEC)),
    }
    for name, (t, shape) in expected.items():
        if t.dtype != torch.uint8 or tuple(t.shape) != shape or not t.is_contiguous():
            return False, f"{name} is {t.dtype} {tuple(t.shape)}, expected contiguous uint8 {shape}"
    if i_pad % 128 != 0 or not 0 < intermediate_size <= i_pad or intermediate_size % 32 != 0:
        return False, f"intermediate {intermediate_size} in buffers of {i_pad}"
    if local_num_experts > NUM_EXPERTS:
        return False, f"{local_num_experts} local experts"
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if sms < _M1_MIN_CTAS:
        return False, f"{sms} SMs, the kernel needs {_M1_MIN_CTAS}"
    return True, ""


def m2_supported(
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    local_num_experts: int,
    intermediate_size: int,
) -> Tuple[bool, str]:
    """Whether these TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers fit ``k3_moe_m2``: ``k3_moe_m1``'s conditions, and an
    intermediate of at most 256 (the TP16 slice; its FC2 tiles of every expert slot stay in shared memory)."""
    ok, why = m1_supported(
        w3_w1_weight,
        w3_w1_weight_scale,
        w2_weight,
        w2_weight_scale,
        local_num_experts,
        intermediate_size,
    )
    if ok and w3_w1_weight.shape[1] // 2 > 256:
        return False, f"padded intermediate {w3_w1_weight.shape[1] // 2} > 256"
    return ok, why


def _engine_config(i_tp: int, i_pad: int, num_local: int, num_ctas: int, m_max: int, push_world: int = 0,
                   copies: int = 1) -> dict:  # fmt: skip
    """The kernel options of one ``k3_moe_m1`` / ``k3_moe_m2`` build (trace-time constants). ``push_world``: the push
    build for a latent exchange of that many slots, this rank filling ``copies`` of them."""
    config = {
        "i_tp": i_tp,
        "i_pad": i_pad,
        "num_local": num_local,
        "num_ctas": num_ctas,
        "m_max": m_max,
    }
    if push_world:
        config.update(push=1, push_world=push_world, push_copies=copies)
    return config


def _engine_args(
    weights, x_fp8, x_sf, topk_ids, topk_weights, hbuf, counts, epochs, y, lat_mc, lat_flags
) -> tuple:
    """The tensor arguments of ``k3_moe_m1`` / ``k3_moe_m2`` in their order: the weight / activation views, then the
    flat arrays (ids, weights, w2 scale words, intermediate rows and their words, counts, epochs, output, the latent
    exchange's multicast words and flags)."""
    w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale = weights
    e, two_i, _ = w3_w1_weight.shape
    m = topk_ids.shape[0]
    return (
        w3_w1_weight.view(torch.int8).permute(2, 1, 0), x_fp8.view(torch.uint8).permute(1, 0),
        w3_w1_weight_scale.view(e, two_i // 128, HIDDEN_SIZE // 128, 512).permute(3, 2, 1, 0),
        x_sf.view(torch.uint8).view(m, HIDDEN_SIZE // _SF_VEC), w2_weight.view(torch.int8).permute(2, 1, 0),
        topk_ids.view(-1), topk_weights.view(torch.int16).view(-1), w2_weight_scale.view(-1).view(torch.int32), hbuf,
        hbuf.view(torch.int32), counts, epochs, y.view(-1), lat_mc.view(-1), lat_flags.view(-1),
    )  # fmt: skip


def _engine_scalars(kind: str, local_expert_offset: int, num_tokens: int, lat_rank: int) -> tuple:
    if kind == "m1":
        return (local_expert_offset, lat_rank)
    return (local_expert_offset, num_tokens, lat_rank)


def _engine_compile(kind: str, mod, args, scalars):
    """TVM-FFI build of ``mod``'s kernel for these torch arguments' types and layouts."""
    import cutlass.cute as cute

    assert len(args) == len(_ENGINE_ALIGNS)
    signature = [_view(t, a, d) for t, a, d in zip(args, _ENGINE_ALIGNS, _ENGINE_LEADING)]
    return cute.compile(
        getattr(mod, f"k3_moe_{kind}"),
        *signature,
        *scalars,
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _engine_workspace_sizes(kind: str, mod, m: int) -> Tuple[int, int]:
    """(intermediate row bytes, count words) of one build's workspace for ``m`` tokens."""
    if kind == "m1":
        return mod.G_CAP * m * mod.H_ROW, 4
    return mod.G_CAP * mod.M_MAX * mod.H_ROW, 2 * mod.GROUPS2 * mod.CW


def _engine_call(kind, x_fp8, x_sf, topk_ids, topk_weights, weights, hbuf, counts, epochs, local_expert_offset,
                 intermediate_size, lat_mc, lat_flags, lat_rank, copies, out, compile_num_local=0):  # fmt: skip
    """The body of ``trtllm::k3_moe_m1`` / ``trtllm::k3_moe_m2``: checks, then the build's launch (compiled on its
    first call). ``compile_num_local``: compile the build for ``compile_num_local`` local experts and launch nothing
    (``weights`` may then be stand-ins of any expert count)."""
    name = f"k3_moe_{kind}"
    m = topk_ids.shape[0]
    if (m not in (1, 2)) if kind == "m1" else m != 2:
        raise ValueError(
            f"{name}: {'1 or 2 tokens' if kind == 'm1' else '2 tokens'} per call, got {m}"
        )
    if (
        topk_ids.dtype != torch.int32
        or tuple(topk_ids.shape) != (m, TOP_K)
        or topk_weights.dtype != torch.bfloat16
        or tuple(topk_weights.shape) != (m, TOP_K)
        or x_fp8.dtype != torch.float8_e4m3fn
        or tuple(x_fp8.shape) != (m, HIDDEN_SIZE)
        or x_sf.numel() != m * (HIDDEN_SIZE // _SF_VEC)
        or not (topk_ids.is_contiguous() and topk_weights.is_contiguous())
        or not (x_fp8.is_contiguous() and x_sf.is_contiguous())
    ):
        raise ValueError(f"{name}: expects {m} token(s)' routing and MXFP8 latents")
    num_local = compile_num_local or weights[0].shape[0]
    if not compile_num_local:
        supported = m1_supported if kind == "m1" else m2_supported
        ok, why = supported(*weights, num_local, intermediate_size)
        if not ok:
            raise ValueError(f"{name}: {why}")
    push = lat_mc is not None
    if push != (lat_flags is not None):
        raise ValueError(f"{name}: exchange_mc and exchange_flags go together")
    slots = 0
    if push:
        slots = lat_mc.numel() // _EXCHANGE_SLOT_WORDS
        if (
            lat_mc.dtype != torch.int32
            or slots == 0
            or lat_mc.numel() != slots * _EXCHANGE_SLOT_WORDS
            or lat_flags.dtype != torch.int32
            or lat_flags.numel() < 1
        ):
            raise ValueError(
                f"{name}: the exchange is int32 words [2][8][slots][1792] and int32 flags"
            )
        if copies < 1 or not 0 <= lat_rank * copies <= slots - copies:
            raise ValueError(
                f"{name}: slots {lat_rank * copies}..+{copies} outside the exchange's {slots} slots"
            )
        if out is not None:
            raise ValueError(f"{name}: the push build takes no out")
    num_ctas = epochs.numel()
    i_pad = weights[0].shape[1] // 2
    config = _engine_config(
        intermediate_size, i_pad, num_local, num_ctas, m if kind == "m1" else 2, slots, copies
    )
    mod = _engine_module(kind, config)
    hbuf_bytes, count_words = _engine_workspace_sizes(kind, mod, m)
    if (
        num_ctas < _M1_MIN_CTAS
        or epochs.dtype != torch.int32
        or hbuf.dtype != torch.int8
        or hbuf.numel() != hbuf_bytes
        or counts.dtype != torch.int32
        or counts.numel() != count_words
        or not all(t.is_contiguous() for t in (hbuf, counts, epochs))
    ):
        raise ValueError(f"{name}: hbuf / counts / epochs are not this build's workspace")
    if out is None:
        y = torch.empty(m, HIDDEN_SIZE, dtype=torch.bfloat16, device=x_fp8.device)
    else:
        if (
            out.dtype != torch.bfloat16
            or tuple(out.shape) != (m, HIDDEN_SIZE)
            or not out.is_contiguous()
        ):
            raise ValueError(f"{name}: out must be contiguous bf16 [{m}, 3584]")
        y = out
    # The plain build never touches the exchange arguments: the counts stand in for them.
    args = _engine_args(weights, x_fp8, x_sf, topk_ids, topk_weights, hbuf, counts, epochs, y,
                        lat_mc if push else counts, lat_flags if push else counts)  # fmt: skip
    scalars = _engine_scalars(kind, local_expert_offset, m, lat_rank)
    key = (kind,) + _compile_key(x_fp8.device, config)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"trtllm::{name} compiles on its first call for each build: call it before capture"
            )
        fn = _compiled[key] = _engine_compile(kind, mod, args, scalars)
    if compile_num_local:
        return None
    fn(*args, *scalars, torch.cuda.current_stream(x_fp8.device).cuda_stream)
    if out is not None or push:
        return y.new_empty((0, HIDDEN_SIZE))
    return y


@torch.library.custom_op(
    "trtllm::k3_moe_m1",
    mutates_args=("hbuf", "counts", "epochs", "exchange_mc", "out"),
)
def k3_moe_m1(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    hbuf: torch.Tensor,
    counts: torch.Tensor,
    epochs: torch.Tensor,
    local_expert_offset: int,
    intermediate_size: int,
    exchange_mc: Optional[torch.Tensor] = None,
    exchange_flags: Optional[torch.Tensor] = None,
    exchange_rank: int = 0,
    copies: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """This rank's routed partial ``[M, 3584]`` bf16 for M = 1 or 2 decode tokens from the weight-stream kernel
    ``k3_moe_m1``: FC1 + SiTU + FC2 with k3_moe's combine over this rank's experts.

    ``x_fp8`` float8_e4m3fn ``[M, 3584]``, ``x_sf`` its E8M0 scales (``M * 112`` bytes), ``topk_ids`` int32
    ``[M, 16]`` global expert ids and ``topk_weights`` bf16 ``[M, 16]``: the outputs of ``trtllm::k3_route_quant`` or
    ``trtllm::k3_moe_front``. The weights are the TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers of a rank's experts (global ids
    ``[local_expert_offset, local_expert_offset + E)``), read in place, holding an ``intermediate_size`` slice
    zero-padded to a multiple of 128. ``hbuf``, ``counts``, ``epochs``: a :class:`K3MoeM1State`'s workspace for M
    tokens; every call re-arms the count slot the next call uses and advances the epochs. ``out``: contiguous bf16
    ``[M, 3584]``; the call writes it and returns an empty ``[0, 3584]`` instead of a new tensor.

    ``exchange_mc`` / ``exchange_flags``: a ``K3LatentExchange``'s multicast words and flags. With them the call is
    the push build: the partial goes into slots ``exchange_rank * copies .. + copies - 1`` of half
    ``exchange_flags[0] & 1`` of every rank's exchange (bf16 pairs, -0.0 stored as +0.0) for
    ``trtllm::k3_latent_reduce``, and the call returns an empty ``[0, 3584]``. Each push of M tokens is followed by one
    reduce of M tokens on that exchange before the next push."""
    weights = (w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)
    return _engine_call("m1", x_fp8, x_sf, topk_ids, topk_weights, weights, hbuf, counts, epochs,
                        local_expert_offset, intermediate_size, exchange_mc, exchange_flags, exchange_rank, copies,
                        out)  # fmt: skip


@k3_moe_m1.register_fake
def _(x_fp8, x_sf, topk_ids, topk_weights, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, hbuf, counts,
      epochs, local_expert_offset, intermediate_size, exchange_mc=None, exchange_flags=None, exchange_rank=0, copies=1,
      out=None):  # fmt: skip
    rows = 0 if out is not None or exchange_mc is not None else topk_ids.shape[0]
    return x_fp8.new_empty((rows, HIDDEN_SIZE), dtype=torch.bfloat16)


@torch.library.custom_op(
    "trtllm::k3_moe_m2",
    mutates_args=("hbuf", "counts", "epochs", "exchange_mc", "out"),
)
def k3_moe_m2(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    hbuf: torch.Tensor,
    counts: torch.Tensor,
    epochs: torch.Tensor,
    local_expert_offset: int,
    intermediate_size: int,
    exchange_mc: Optional[torch.Tensor] = None,
    exchange_flags: Optional[torch.Tensor] = None,
    exchange_rank: int = 0,
    copies: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``trtllm::k3_moe_m1``'s contract for exactly two tokens, from the weight-stream kernel ``k3_moe_m2`` (an
    intermediate slice of at most 256; the FC2 tiles of every expert slot stay in shared memory) on a
    :class:`K3MoeM2State`'s workspace."""
    weights = (w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)
    return _engine_call("m2", x_fp8, x_sf, topk_ids, topk_weights, weights, hbuf, counts, epochs,
                        local_expert_offset, intermediate_size, exchange_mc, exchange_flags, exchange_rank, copies,
                        out)  # fmt: skip


@k3_moe_m2.register_fake
def _(x_fp8, x_sf, topk_ids, topk_weights, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, hbuf, counts,
      epochs, local_expert_offset, intermediate_size, exchange_mc=None, exchange_flags=None, exchange_rank=0, copies=1,
      out=None):  # fmt: skip
    rows = 0 if out is not None or exchange_mc is not None else topk_ids.shape[0]
    return x_fp8.new_empty((rows, HIDDEN_SIZE), dtype=torch.bfloat16)


class _K3MoeEngineState:
    """The workspace of ``trtllm::k3_moe_m1`` / ``trtllm::k3_moe_m2`` on one device, shared by the layers that run on
    it: the intermediate rows (``hbuf``), the FC1 -> FC2 counts (``counts``, two sets by epoch parity) and the CTAs'
    epochs (``epochs``). Every call re-arms the count set the next call uses and advances the epochs, so the
    workspace never needs a reset; its layers run in one stream order. Build it with ``create`` before CUDA-graph
    capture and keep it with the model: ``create`` compiles the plain build and the push builds it is given. A build
    it did not compile compiles on its first call, which must come before capture too."""

    kind = ""

    def __init__(
        self, device: torch.device, i_tp: int, i_pad: int, num_local: int, num_tokens: int
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{type(self).__name__} allocates its workspace: build it outside CUDA-graph capture"
            )
        num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        self.config = _engine_config(i_tp, i_pad, num_local, num_ctas, num_tokens)
        self.mod = mod = _engine_module(self.kind, self.config)
        self.device = device
        self.i_tp = i_tp
        self.i_pad = i_pad
        self.num_local = num_local
        self.num_tokens = num_tokens
        hbuf_bytes, count_words = _engine_workspace_sizes(self.kind, mod, num_tokens)
        kw = dict(device=device)
        self.hbuf = torch.zeros(hbuf_bytes, dtype=torch.int8, **kw)
        self.counts = torch.zeros(count_words, dtype=torch.int32, **kw)
        self.epochs = torch.zeros(num_ctas, dtype=torch.int32, **kw)

    def _create(self, push: Tuple[Tuple[int, int], ...]) -> "_K3MoeEngineState":
        """Compile the plain build and a push build per (exchange slots, copies) in ``push``, launching nothing: two
        experts' zero buffers and zero tokens stand in for a layer's arguments (the builds take any shapes of these
        types and layouts)."""
        kw = dict(device=self.device)
        m, i_pad = self.num_tokens, self.i_pad
        weights = (
            torch.zeros(2, 2 * i_pad, HIDDEN_SIZE // 2, dtype=torch.uint8, **kw),
            torch.zeros(2, 2 * i_pad, HIDDEN_SIZE // _SF_VEC, dtype=torch.uint8, **kw),
            torch.zeros(2, HIDDEN_SIZE, i_pad // 2, dtype=torch.uint8, **kw),
            torch.zeros(2, HIDDEN_SIZE, i_pad // _SF_VEC, dtype=torch.uint8, **kw),
        )
        tokens = (
            torch.zeros(m, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, **kw),
            torch.zeros(m, HIDDEN_SIZE // _SF_VEC, dtype=torch.uint8, **kw),
            torch.zeros(m, TOP_K, dtype=torch.int32, **kw),
            torch.zeros(m, TOP_K, dtype=torch.bfloat16, **kw),
        )
        builds = [(None, None, 1)]
        for slots, copies in push:
            lat_mc = torch.zeros(slots * _EXCHANGE_SLOT_WORDS, dtype=torch.int32, **kw)
            builds.append((lat_mc, torch.zeros(4, dtype=torch.int32, **kw), copies))
        for lat_mc, lat_flags, copies in builds:
            _engine_call(self.kind, *tokens, weights, self.hbuf, self.counts, self.epochs, 0, self.i_tp, lat_mc,
                         lat_flags, 0, copies, None, compile_num_local=self.num_local)  # fmt: skip
        return self

    @property
    def compiled(self) -> bool:
        """Whether the plain build has been compiled on this device (by any state's ``create`` or first call)."""
        return (self.kind,) + _compile_key(self.device, self.config) in _compiled

    def push_compiled(self, slots: int, copies: int = 1) -> bool:
        """Whether the push build for an exchange of ``slots`` slots, this rank filling ``copies``, is compiled."""
        config = _engine_config(self.i_tp, self.i_pad, self.num_local, self.epochs.numel(), self.config["m_max"],
                                slots, copies)  # fmt: skip
        return (self.kind,) + _compile_key(self.device, config) in _compiled


class K3MoeM1State(_K3MoeEngineState):
    """``trtllm::k3_moe_m1``'s workspace on one device for ``num_tokens`` (1 or 2) tokens per call: see
    :class:`_K3MoeEngineState`. The count set is two words (one per epoch parity)."""

    kind = "m1"

    @classmethod
    def create(
        cls,
        device: torch.device,
        i_tp: int,
        i_pad: int,
        num_local: int,
        num_tokens: int = 1,
        push: Tuple[Tuple[int, int], ...] = (),
    ) -> "K3MoeM1State":
        """The state on ``device`` with its builds compiled: the plain build, and the push build for each (exchange
        slots, copies) in ``push`` (``((tp_size, 1),)`` for a TP group's ``K3LatentExchange``). Eager: it allocates and
        compiles, so it refuses to run under CUDA-graph capture. Not collective."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3MoeM1State.create allocates and compiles: call it before CUDA-graph capture"
            )
        return cls(torch.device(device), i_tp, i_pad, num_local, num_tokens)._create(push)

    def __init__(
        self, device: torch.device, i_tp: int, i_pad: int, num_local: int, num_tokens: int = 1
    ):
        if num_tokens not in (1, 2):
            raise ValueError(f"k3_moe_m1 takes 1 or 2 tokens per call, not {num_tokens}")
        super().__init__(device, i_tp, i_pad, num_local, num_tokens)

    def layer(
        self,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ) -> "K3MoeM1Layer":
        """A layer's handle: its experts' TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers, read in place."""
        return K3MoeM1Layer(self, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)


class K3MoeM2State(_K3MoeEngineState):
    """``trtllm::k3_moe_m2``'s workspace on one device (two tokens per call): see :class:`_K3MoeEngineState`. The
    count sets are per FC1 -> FC2 group."""

    kind = "m2"
    num_tokens = 2

    @classmethod
    def create(
        cls,
        device: torch.device,
        i_tp: int,
        i_pad: int,
        num_local: int,
        push: Tuple[Tuple[int, int], ...] = (),
    ) -> "K3MoeM2State":
        """The state on ``device`` with its builds compiled: as :meth:`K3MoeM1State.create`."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3MoeM2State.create allocates and compiles: call it before CUDA-graph capture"
            )
        return cls(torch.device(device), i_tp, i_pad, num_local)._create(push)

    def __init__(self, device: torch.device, i_tp: int, i_pad: int, num_local: int):
        super().__init__(device, i_tp, i_pad, num_local, 2)

    def layer(
        self,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ) -> "K3MoeM2Layer":
        """A layer's handle: its experts' TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers, read in place."""
        return K3MoeM2Layer(self, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)


class _K3MoeEngineLayer:
    """One MoE layer on a ``k3_moe_m1`` / ``k3_moe_m2`` state: its experts' weight buffers, read in place."""

    def __init__(
        self,
        state: _K3MoeEngineState,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ):
        supported = m1_supported if state.kind == "m1" else m2_supported
        ok, why = supported(
            w3_w1_weight,
            w3_w1_weight_scale,
            w2_weight,
            w2_weight_scale,
            state.num_local,
            state.i_tp,
        )
        if not ok or w3_w1_weight.shape[1] != 2 * state.i_pad:
            raise ValueError(
                f"k3_moe_{state.kind} layer: {why or 'padded intermediate differs from the state'}"
            )
        self.state = state
        self.weights = (w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)

    def _op(self):
        return torch.ops.trtllm.k3_moe_m1 if self.state.kind == "m1" else torch.ops.trtllm.k3_moe_m2

    def __call__(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This rank's routed partial ``[M, 3584]`` bf16 for the state's M tokens (``out``, written, when given): the
        op on this layer's experts and its state's workspace. See ``trtllm::k3_moe_m1``."""
        st = self.state
        y = self._op()(x_fp8, x_sf, topk_ids, topk_weights, *self.weights, st.hbuf, st.counts, st.epochs,
                       local_expert_offset, st.i_tp, out=out)  # fmt: skip
        return y if out is None else out

    def push(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        lat_mc: torch.Tensor,
        lat_flags: torch.Tensor,
        lat_rank: int,
        copies: int = 1,
    ) -> None:
        """The push build: the same routed partial, stored into slots ``lat_rank * copies .. + copies - 1`` of every
        rank's latent exchange (``lat_mc``: the multicast int32 words of a ``K3LatentExchange``, ``lat_flags`` its
        flags) for ``trtllm::k3_latent_reduce``, instead of returned. See ``trtllm::k3_moe_m1``."""
        st = self.state
        self._op()(x_fp8, x_sf, topk_ids, topk_weights, *self.weights, st.hbuf, st.counts, st.epochs,
                   local_expert_offset, st.i_tp, lat_mc, lat_flags, lat_rank, copies)  # fmt: skip


class K3MoeM1Layer(_K3MoeEngineLayer):
    """One MoE layer on a :class:`K3MoeM1State`."""


class K3MoeM2Layer(_K3MoeEngineLayer):
    """One MoE layer on a :class:`K3MoeM2State`."""
