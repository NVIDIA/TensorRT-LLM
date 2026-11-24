# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 ultra-low-latency routed experts (TRTLLM_K3_ULL_EXPERTS).

``TRTLLM_K3_ULL_EXPERTS=1``: for at most 32 tokens (8 for MoE-TP intermediates below 1024 without
``TRTLLM_K3_ULL_MOETP``) the CUTEDSL_FC12 backend runs the small-batch NVFP4 expert kernel
(:mod:`..kernels.k3_moe_ull`: FC1 + SiTU + NVFP4 requantization + FC2 + combine
in one dynamically scheduled persistent grid; the combine adds BF16 per-token sums with vector reds
by default, or sums FP32 slices deterministically) instead of the fused FC12 kernel. It reads the
same weights (``NvFp4WeightView``), the same NVFP4 latent (``x`` [M, H/2] e2m1 pairs, ``x_sf``
[M, H/16] E4M3 linear) and the same tile-128 routing tables (``moe_sort`` or the routing tail's
``_k3_presorted`` outputs), and writes every element of the routed output (no memset before it).
W1 and its block scales are read in the row order the backend's quantization method gives them
(:func:`fc1_gate_up16`); a backend whose order the kernel does not read keeps FC12. Other calls,
other activations, all-to-all and other tile sizes keep FC12.

Switches: ``TRTLLM_K3_ULL_EXPERTS_CTAS`` persistent CTAs (``"<N>"`` for every token tile or
``"<n_tok>:<N>,..."`` per token tile 8 / 16 / 32; default one per SM), ``_TRIGGER``, ``_COMBINE``
(below). ``TRTLLM_K3_ULL_MOETP=1``: MoE-TP layouts (TP4xEP4 / TP8xEP2 / TP16 slices of every
expert's intermediate) are served up to 32 tokens with the MoE-TP schedule (below).
"""

import os
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.logger import logger

from ..features import active as _features

ULL_EXPERTS_ENV = "TRTLLM_K3_ULL_EXPERTS"
# CTAs per launch: features.ull_experts_ctas ("<N>" or "<n_tok>:<N>,..."; empty: every SM).
# PDL trigger point, features.ull_experts_trigger: "end" (every CTA after all of its warps) or "wait"
# (right after griddepcontrol.wait: the dependent grid launches and sets up while this one runs).
# FC2 combine: True: BF16 vector reds into the output the kernel zeroes first, unspecified add order
# like FC12's finalize, no round trip at the end of the kernel (-1.3..-3 us per layer); False (tests):
# deterministic FP32 slice sum, one BF16 write per element.
_COMBINE_RED = True
# MoE-TP layouts (moe_tp_size > 1, e.g. TP4xEP4 / TP8xEP2 under helix16): the experts kernel serves
# MoE-TP intermediates (< 3072 per rank) up to 32 tokens (instead of 8) with the MoE-TP schedule:
# token-indexed FC1 output + register FC2 epilogue over a shared-memory slot table (k3_moe_ull
# ``tokidx``, same bits as the permuted layout with the slice combine), FC2 tasks claimed only once
# 4 ring stages are free for them, 24 combine slices (16 ranks over 4 nodes, experts + latent
# all-reduce per layer: TP4xEP4 is faster at 8 / 16 / 32 rows than EP16 with the default kernel).
# EP shapes keep the permuted layout, claiming one unit ahead and 16 slices. With die-major
# weights (TRTLLM_K3_ULL_EXPERTS_DIE_MAJOR=1, see fused_moe_cute_dsl_fc12) the ring is 12 stages
# deep, else 8 (uniform routing, 16 ranks, experts + latent all-reduce per layer at 8 / 16 / 32
# rows: 12 stages are slightly faster than 8 for both TP4xEP4 and EP16, and die-major EP16 is
# faster than EP16 on default placement).
# TRTLLM_K3_ULL_MOEBLK=1 (with TRTLLM_K3_ULL_FRONT=1 and the token-indexed layout): the front's
# route kernel publishes each token's top-k (no moe_sort tables, no cluster mask exchange) and the
# experts kernel builds its tables itself from the published ids / weights, so its weight stream
# starts before the route kernel ends (k3_moe_ull ``moeblk``; same bits).
MOEBLK_ENV = "TRTLLM_K3_ULL_MOEBLK"


def _sm_die_words_any(device: torch.device) -> Tuple[int, ...]:
    """Packed SM -> die table of ``device`` for die-major claiming: the measured one, else the
    fallback ``(smid >> 1) & 1`` (an approximation; only locality depends on it)."""
    from tensorrt_llm._torch.die_locality import pack_sm_die_words, sm_die_table

    index = device.index if device.index is not None else torch.cuda.current_device()
    table = sm_die_table(index)
    if table is None:
        table = tuple(
            (s >> 1) & 1
            for s in range(torch.cuda.get_device_properties(index).multi_processor_count)
        )
    return pack_sm_die_words(table)


def die_major_layout(t: torch.Tensor) -> torch.Tensor:
    """[L, R, C] -> [2 L, R / 2, C]: slot e = rows [0, R / 2) of expert e, slot L + e = rows
    [R / 2, R) (128-row scale blocks stay whole, so the 128x4-swizzled scales split the same)."""
    rows = t.size(1)
    assert rows % 256 == 0, "die-major halves need whole 128-row blocks"
    return torch.cat([t[:, : rows // 2], t[:, rows // 2 :]], dim=0).contiguous()


def die_major_weights(
    w1: torch.Tensor, w1_sf: torch.Tensor, w2: torch.Tensor, w2_sf: torch.Tensor
) -> Tuple[Tuple[torch.Tensor, ...], bool]:
    """Die-major copies of an NVFP4 expert slice (``K3UllExpertsRunner.run(die_major=True)``):
    each tensor in :func:`die_major_layout`, its first half (every expert's first row half) homed
    on die 0 and the second on die 1 (``die_locality.die_split_copy``). Returns the four tensors and
    whether the placement took (False: same layout on default memory, e.g. one die)."""
    from tensorrt_llm._torch.die_locality import die_split_copy

    out, placed = [], True
    for t in (w1, w1_sf, w2, w2_sf):
        dm = die_major_layout(t)
        nbytes = dm.numel() * dm.element_size()
        c = die_split_copy(dm, nbytes // 2)
        if c is None:
            placed = False
            c = dm
        out.append(c)
    return tuple(out), placed


#: Largest token count of the kernel (MMA N 32).
MAX_TOKENS = 32
#: Tiles the counter buffer of a caller holds.
MAX_TILES = 1024


def ull_experts_mode() -> int:
    """0: off; 1: the stage-1 kernel (k3_moe_ull)."""
    try:
        return int(os.environ.get(ULL_EXPERTS_ENV, "0"))
    except ValueError:
        return 0


def ull_moetp() -> bool:
    """TRTLLM_K3_ULL_MOETP=1: the MoE-TP schedule of the experts kernel."""
    return _features().ull_moetp


def _moetp_shape(n1: int) -> bool:
    """Whether W1 rows ``n1`` (2 x intermediate) are a MoE-TP slice served with the MoE-TP
    schedule (intermediate < 3072 and TRTLLM_K3_ULL_MOETP=1)."""
    return n1 // 2 < 3072 and ull_moetp()


_UNPROBED = object()


def _probe_fc1_gate_up16(backend, n1: int) -> Optional[bool]:
    from tensorrt_llm._torch.moe.fused_moe.quantization import (
        interleave_gate_and_linear,
        interleave_linear_and_gate,
    )

    interleave = getattr(getattr(backend, "quant_method", None), "_interleave_fc1_rows", None)
    if interleave is None or n1 % 128:
        return None
    rows = torch.arange(n1, dtype=torch.int32).view(n1, 1)
    order = interleave(rows)
    if torch.equal(order, interleave_gate_and_linear(rows, group_size=16, dim=0)):
        return True
    if torch.equal(order, interleave_linear_and_gate(rows, group_size=64, dim=0)):
        return False
    return None


def fc1_gate_up16(backend) -> Optional[bool]:
    """The row order ``backend``'s quantization method gives every expert's W1 and its block
    scales (its ``_interleave_fc1_rows``, applied once to the row numbers and cached on the
    backend): True for ``[gate16, up16]`` groups, False for ``[up64, gate64]`` groups, None for
    any other order (or none), which the ULL experts kernel does not read."""
    layout = getattr(backend, "_ull_fc1_gate_up16", _UNPROBED)
    if layout is _UNPROBED:
        w1 = getattr(backend, "w3_w1_weight", None)
        layout = None
        if w1 is not None and w1.dim() == 3:
            layout = _probe_fc1_gate_up16(backend, w1.size(1))
        if layout is None:
            logger.warning_once(
                f"{ULL_EXPERTS_ENV}: the expert backend's FC1 row order is neither [gate16, up16] "
                "nor [up64, gate64]; its calls keep FC12",
                key="k3_ull_fc1_layout_unknown",
            )
        else:
            order = "[gate16, up16]" if layout else "[up64, gate64]"
            logger.info_once(
                f"{ULL_EXPERTS_ENV}: expert FC1 rows in {order} groups",
                key=f"k3_ull_fc1_layout_{order}",
            )
        backend._ull_fc1_gate_up16 = layout
    return layout


def ull_moeblk_applies(backend, num_tokens: int) -> bool:
    """TRTLLM_K3_ULL_MOEBLK: whether this call of ``backend`` (a CUTEDSL_FC12 backend) is served
    by the ULL experts kernel with the token-indexed layout, so its tables can be built in the
    kernel from the front's published top-k."""
    if os.environ.get(MOEBLK_ENV, "0") != "1" or ull_experts_mode() <= 0:
        return False
    if getattr(backend, "activation_type", None) != ActivationType.SiTu:
        return False
    w1, w2 = getattr(backend, "w3_w1_weight", None), getattr(backend, "w2_weight", None)
    if w1 is None or w2 is None or w1.dim() != 3 or w2.dim() != 3:
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    if fc1_gate_up16(backend) is None:
        return False
    n1, k1 = w1.size(1), w2.size(1)
    if not K3UllExpertsRunner.applies(
        num_tokens, n1, k1, 128, getattr(backend, "scaling_vector_size", 16)
    ):
        return False
    # the token-indexed layout: MoE-TP shapes
    return _moetp_shape(n1)


def ull_experts_max_tokens(intermediate: int = 3072) -> int:
    """Largest token count served (0: disabled): 32, or 8 for MoE-TP intermediates (< 1024 per
    rank: FC2's K is one or two k-tiles and the many small FC2 tile epilogues make the
    permuted-layout kernel slower than FC12 above 8 tokens at TP8xEP2) unless
    TRTLLM_K3_ULL_MOETP=1 (token-indexed epilogue)."""
    if ull_experts_mode() <= 0:
        return 0
    return MAX_TOKENS if intermediate >= 1024 or ull_moetp() else 8


def _per_tile(spec: str, n_tok: int, default: int) -> int:
    """``"<N>"`` or ``"<n_tok>:<N>,..."`` (N > 0) from ``spec`` for token tile ``n_tok``."""
    spec = spec.strip()
    value = default
    if not spec:
        return value
    for item in spec.split(","):
        if ":" in item:
            key, val = item.split(":")
            if int(key) == n_tok and int(val) > 0:
                value = int(val)
        elif int(item) > 0:
            value = int(item)
    return value


def n_tok_for(num_tokens: int) -> int:
    for n_tok in (8, 16, 32):
        if num_tokens <= n_tok:
            return n_tok
    raise ValueError(f"{num_tokens} tokens exceed the ULL expert kernel ({MAX_TOKENS})")


class K3UllExpertsRunner:
    """Compiles (once per configuration) and launches :class:`K3MoeUllKernel`."""

    kernel_cache: Dict[tuple, object] = {}
    _workspaces: Dict[Tuple[int, int], torch.Tensor] = {}
    _padded_h_sf: Dict[int, torch.Tensor] = {}

    def __init__(self, top_k: int, situ_beta: float, situ_linear_beta: float):
        self.top_k = top_k
        self.situ_beta = float(situ_beta)
        self.situ_linear_beta = float(situ_linear_beta)

    @staticmethod
    def applies(
        num_tokens: int, n1: int, k1: int, tile_size: int, scaling_vector_size: int
    ) -> bool:
        from ..kernels.k3_moe_ull import K3MoeUllKernel

        max_tokens = ull_experts_max_tokens(n1 // 2)
        return (
            1 <= num_tokens <= max_tokens
            and tile_size == 128
            and scaling_vector_size == 16
            and K3MoeUllKernel.supports(n_tok_for(num_tokens), n1, k1)
        )

    @staticmethod
    def new_counters(device: torch.device) -> torch.Tensor:
        """A zeroed counter buffer for one caller; every launch leaves it zero again."""
        from ..kernels.k3_moe_ull import counter_words

        return torch.zeros(counter_words(MAX_TILES), dtype=torch.int32, device=device)

    @classmethod
    def workspace(cls, device: torch.device, n2: int) -> torch.Tensor:
        """Combine workspace shared by every caller on ``device`` (launches are stream-ordered;
        nothing in it needs initialization)."""
        from ..kernels.k3_moe_ull import MAX_SLICES, workspace_words

        key = (device.index if device.index is not None else torch.cuda.current_device(), n2)
        ws = cls._workspaces.get(key)
        if ws is None:
            ws = torch.empty(
                workspace_words(n2, MAX_TOKENS, MAX_SLICES), dtype=torch.int32, device=device
            )
            cls._workspaces[key] = ws
        return ws

    def run(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        w1: torch.Tensor,
        w1_sf: torch.Tensor,
        alpha1: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        alpha2: torch.Tensor,
        out: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        token_final_scales: torch.Tensor,
        counters: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        h_sf: Optional[torch.Tensor] = None,
        num_ctas: Optional[int] = None,
        ar_push: Optional[Tuple] = None,
        expanded_idx_to_permuted_idx: Optional[torch.Tensor] = None,
        die_major: bool = False,
        moeblk_offset: Optional[int] = None,
        gate_up16: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``ar_push = (multicast base address, LamportFlags int32 tensor, rank, world)`` of the
        routed latent all-reduce's own workspace: the combine (slice or red) also stores the
        output rows into this rank's slot of every rank's current Lamport buffer (the consumer then
        runs ``run_k3_latent_ar_norm_row(..., prepushed=True)`` or ``k3_moe_tail1``).
        ``expanded_idx_to_permuted_idx`` (``moe_sort``'s [tokens, top_k], -1 off this rank)
        enables the token-indexed layout on MoE-TP shapes (TRTLLM_K3_ULL_MOETP); without it, or on
        other shapes, the permuted layout runs. With the token-indexed layout row ``tile * 128 + j``
        of ``h`` is token j's intermediate. ``die_major``: the four weight tensors are
        :func:`die_major_weights` copies ([2 L, rows / 2, cols], each die homing one half of every
        expert): FC1 / FC2 work is claimed per die. ``moeblk_offset`` (TRTLLM_K3_ULL_MOEBLK):
        ``expanded_idx_to_permuted_idx`` holds the top-k ids and ``token_final_scales`` the weights
        the front's route kernel publishes into word ``CTR_ROUTE`` of ``counters``; the kernel
        builds the tables of this rank's experts (``moeblk_offset`` its first expert) and the
        moe_sort tables are not read. ``gate_up16``: W1 and its scales hold ``[gate16, up16]`` row
        groups (:func:`fc1_gate_up16`), else ``[up64, gate64]``."""
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda

        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import get_max_activate_clusters
        from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL, make_ptr

        from ..kernels.k3_moe_ull import K3MoeUllKernel, counter_words

        orig_m = x.size(0)
        num_local, n1 = w1.size(0), w1.size(1)
        k1 = w2.size(1)
        if die_major:
            num_local, n1, k1 = num_local // 2, 2 * n1, 2 * k1
        k2 = n1 // 2
        m = permuted_idx_to_expanded_idx.size(0)
        assert m // 128 <= MAX_TILES, f"{m // 128} tiles exceed the counter buffer"
        assert counters.numel() >= counter_words(m // 128)
        n_tok = n_tok_for(orig_m)
        if num_ctas is None:
            num_sms = get_max_activate_clusters(1)
            want = _per_tile(_features().ull_experts_ctas, n_tok, 0)
            num_ctas = min(want, num_sms) if want > 0 else num_sms
        # MoE-TP schedule (TRTLLM_K3_ULL_MOETP, above)
        moetp = _moetp_shape(n1)
        stages = 12 if die_major else 8
        slice_tiles = 1
        max_slices = 24 if moetp else 16
        epi_wg = 2
        tokidx = expanded_idx_to_permuted_idx is not None and moetp
        moeblk = moeblk_offset is not None
        if moeblk and not tokidx:
            raise RuntimeError(f"{MOEBLK_ENV}: needs the token-indexed layout")
        if tokidx:
            assert expanded_idx_to_permuted_idx.numel() == orig_m * self.top_k
        claim_free = 4 if moetp else 0
        die_words = _sm_die_words_any(x.device) if die_major else None
        early_trigger = _features().ull_experts_trigger == "wait"
        combine_red = _COMBINE_RED
        ar_world = 0
        ar_args = [0, 0, 0]
        if ar_push is not None:
            mc_addr, flags, ar_rank, ar_world = ar_push
            ar_args = [int(mc_addr), int(flags.data_ptr()), int(ar_rank)]
        if h is None:
            h = torch.empty(m, k2 // 2, dtype=torch.uint8, device=x.device)
        h_sf_bytes = (m // 128) * K3MoeUllKernel.h_sf_tile_bytes(n1)
        if h_sf is None:
            if k2 % 256 == 0:
                h_sf = torch.empty(h_sf_bytes, dtype=torch.uint8, device=x.device)
            else:
                h_sf = self.padded_h_sf(x.device, h_sf_bytes)
        assert h_sf.numel() >= h_sf_bytes, "FC1-output scale buffer too small"
        ws = self.workspace(x.device, k1)

        def ptr(dtype, t, align=None):
            if align is None:
                return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem)
            return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

        ptrs = [
            ptr(cutlass.Float4E2M1FN, x, 32),
            ptr(cutlass.Float4E2M1FN, w1, 32),
            ptr(cutlass.Float8E4M3FN, x_sf, 16),
            ptr(cutlass.Float8E4M3FN, w1_sf, 16),
            ptr(cutlass.Float4E2M1FN, h, 32),
            ptr(cutlass.Float8E4M3FN, h_sf, 16),
            ptr(cutlass.Float32, alpha1),
            ptr(cutlass.Float4E2M1FN, w2, 32),
            ptr(cutlass.Float8E4M3FN, w2_sf, 16),
            ptr(cutlass.Float32, alpha2),
            ptr(cutlass.BFloat16, out, 16),
            ptr(cutlass.Int32, tile_idx_to_group_idx),
            ptr(cutlass.Int32, tile_idx_to_mn_limit),
            ptr(cutlass.Int32, permuted_idx_to_expanded_idx),
            ptr(cutlass.Int32, num_non_exiting_tiles),
            ptr(cutlass.Float32, global_sf),
            ptr(cutlass.Float32, token_final_scales),
            ptr(
                cutlass.Int32,
                expanded_idx_to_permuted_idx if tokidx else permuted_idx_to_expanded_idx,
            ),
            ptr(cutlass.Int32, counters),
            ptr(cutlass.Int32, ws, 16),
        ]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        key = (
            n_tok,
            n1,
            k1,
            self.top_k,
            self.situ_beta,
            self.situ_linear_beta,
            num_ctas,
            stages,
            slice_tiles,
            max_slices,
            epi_wg,
            TRTLLM_ENABLE_PDL,
            early_trigger,
            combine_red,
            ar_world,
            die_words,
            tokidx,
            claim_free,
            die_major,
            moeblk,
            gate_up16,
        )
        compiled = self.kernel_cache.get(key)
        if compiled is None:
            kernel = K3MoeUllKernel(
                n_tok,
                n1,
                k1,
                self.top_k,
                self.situ_beta,
                self.situ_linear_beta,
                num_epi_wg=epi_wg,
                slice_tiles=slice_tiles,
                max_slices=max_slices,
                max_ab_stage=stages,
                use_pdl=TRTLLM_ENABLE_PDL,
                early_trigger=early_trigger,
                combine_red=combine_red,
                ar_world=ar_world,
                die_sm_words=die_words,
                tokidx=tokidx,
                claim_free=claim_free,
                die_major=die_major,
                moeblk=moeblk,
                gate_up16=gate_up16,
            )
            compiled = cute.compile(
                kernel.wrapper,
                *ptrs,
                cutlass.Int64(ar_args[0]),
                cutlass.Int64(ar_args[1]),
                cutlass.Int32(ar_args[2]),
                orig_m,
                m,
                num_local,
                cutlass.Int32(0),
                num_ctas,
                stream,
            )
            self.kernel_cache[key] = compiled
        compiled(*ptrs, *ar_args, orig_m, m, num_local, int(moeblk_offset or 0), stream)
        return h, h_sf

    @classmethod
    def padded_h_sf(cls, device: torch.device, nbytes: int) -> torch.Tensor:
        """FC1-output scale buffer whose K padding (an intermediate that is not a multiple of 256,
        e.g. 384 at MoE TP8) stays zero: allocated zeroed once per device (outside graph capture,
        warmup runs the shapes first) and only ever written at the valid columns."""
        key = device.index if device.index is not None else torch.cuda.current_device()
        buf = cls._padded_h_sf.get(key)
        if buf is None or buf.numel() < nbytes:
            buf = torch.zeros(max(nbytes, 1 << 20), dtype=torch.uint8, device=device)
            cls._padded_h_sf[key] = buf
        return buf


@torch.library.custom_op(
    "trtllm::k3_ull_experts_nvfp4", mutates_args=("output", "counters"), device_types="cuda"
)
def k3_ull_experts_nvfp4(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight1: torch.Tensor,
    weight1_scale: torch.Tensor,
    alpha1: torch.Tensor,
    weight2: torch.Tensor,
    weight2_scale: torch.Tensor,
    alpha2: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    global_sf: torch.Tensor,
    token_final_scales: torch.Tensor,
    counters: torch.Tensor,
    top_k: int,
    situ_beta: float,
    situ_linear_beta: float,
    slot_mc: int = 0,
    slot_flags: Optional[torch.Tensor] = None,
    slot_rank: int = 0,
    slot_world: int = 0,
    expanded_idx_to_permuted_idx: Optional[torch.Tensor] = None,
    die_major: bool = False,
    moeblk_offset: int = -1,
    gate_up16: bool = False,
) -> None:
    """Routed NVFP4 SiTU MoE of at most 32 tokens in one kernel: writes the sum of this rank's
    experts' weighted down-projections into ``output`` (every element; TRTLLM_K3_ULL_EXPERTS).
    With ``slot_world > 0`` (TRTLLM_K3_ULL_TAIL1): the combine also stores the output rows into
    slot ``(token, slot_rank)`` of every rank's current buffer of the Lamport workspace
    (``slot_mc`` multicast base, ``slot_flags`` its state), the all-reduce push, for
    ``k3_moe_tail1``. ``expanded_idx_to_permuted_idx`` allows the token-indexed layout (MoE-TP
    shapes). ``die_major``: the four weight tensors hold die-major bytes under their
    [L, rows, cols] shapes (TRTLLM_K3_ULL_EXPERTS_DIE_MAJOR). ``moeblk_offset`` >= 0
    (TRTLLM_K3_ULL_MOEBLK): ``expanded_idx_to_permuted_idx`` holds the front's published top-k
    ids, the tables are built in the kernel (``K3UllExpertsRunner.run``). ``gate_up16``: the FC1
    row order (:func:`fc1_gate_up16`)."""
    ar_push = None
    if slot_world > 0:
        ar_push = (slot_mc, slot_flags, slot_rank, slot_world)
    if die_major:

        def _dm(t):
            return t.view(2 * t.size(0), t.size(1) // 2, *t.shape[2:])

        weight1, weight1_scale, weight2, weight2_scale = (
            _dm(weight1),
            _dm(weight1_scale),
            _dm(weight2),
            _dm(weight2_scale),
        )
    K3UllExpertsRunner(top_k, situ_beta, situ_linear_beta).run(
        input,
        input_scale,
        weight1,
        weight1_scale,
        alpha1,
        weight2,
        weight2_scale,
        alpha2,
        output,
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        global_sf,
        token_final_scales,
        counters,
        ar_push=ar_push,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        die_major=die_major,
        moeblk_offset=moeblk_offset if moeblk_offset >= 0 else None,
        gate_up16=gate_up16,
    )


@torch.library.register_fake("trtllm::k3_ull_experts_nvfp4")
def _(
    input,
    input_scale,
    weight1,
    weight1_scale,
    alpha1,
    weight2,
    weight2_scale,
    alpha2,
    output,
    tile_idx_to_group_idx,
    tile_idx_to_mn_limit,
    permuted_idx_to_expanded_idx,
    num_non_exiting_tiles,
    global_sf,
    token_final_scales,
    counters,
    top_k,
    situ_beta,
    situ_linear_beta,
    slot_mc=0,
    slot_flags=None,
    slot_rank=0,
    slot_world=0,
    expanded_idx_to_permuted_idx=None,
    die_major=False,
    moeblk_offset=-1,
    gate_up16=False,
) -> None:
    return None


def ull_experts_applies(
    backend, num_tokens: int, weight_view, tile_size: int, enable_alltoall: bool
) -> bool:
    """Whether the ULL expert kernel serves a CUTEDSL_FC12 call of ``backend``."""
    if ull_experts_mode() <= 0 or enable_alltoall:
        return False
    if getattr(backend, "activation_type", None) != ActivationType.SiTu:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        return False
    if fc1_gate_up16(backend) is None:
        return False
    return K3UllExpertsRunner.applies(
        num_tokens,
        weight_view.w3_w1_weight.size(1),
        weight_view.w2_weight.size(1),
        tile_size,
        getattr(backend, "scaling_vector_size", 16),
    )


def run_ull_experts(
    backend,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    moe_output: torch.Tensor,
    weight_view,
    token_final_scales: torch.Tensor,
    routing: Tuple[torch.Tensor, ...],
    top_k: int,
) -> torch.Tensor:
    """The routed experts of ``backend`` for this call in one kernel; ``routing`` holds the six
    tile-128 ``moe_sort`` outputs of this rank. The counter buffer (``_ull_counters``) is created
    on first use (outside graph capture: warmup runs the same shapes first). With
    ``backend.ull_slot_push = (multicast base, state flags, rank, world)`` (TRTLLM_K3_ULL_TAIL1,
    set by the caller for this call) the kernel writes ``moe_output`` and pushes its rows into the
    workspace's slots; ``backend.ull_slot_push_used`` then becomes True. With
    ``backend.ull_moeblk = (top-k ids, first local expert)`` (TRTLLM_K3_ULL_MOEBLK, set by the
    front for this call) the kernel builds its tables from the published top-k and ``routing``'s
    tables are not read."""
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _,
        num_non_exiting_tiles,
    ) = routing
    # Leaves that quantize the input on an aux stream defer the join to their first reader.
    join_x_ready = getattr(backend, "_join_pending_x_ready", None)
    if join_x_ready is not None:
        join_x_ready()
    counters = getattr(backend, "_ull_counters", None)
    if counters is None:
        counters = K3UllExpertsRunner.new_counters(x.device)
        backend._ull_counters = counters
    slot_kwargs = {}
    slot_push = getattr(backend, "ull_slot_push", None)
    if slot_push is not None:
        mc, flags, rank, world = slot_push
        slot_kwargs = dict(slot_mc=mc, slot_flags=flags, slot_rank=rank, slot_world=world)
        backend.ull_slot_push_used = True
    moeblk = getattr(backend, "ull_moeblk", None)
    moeblk_offset = -1
    if moeblk is not None:
        backend.ull_moeblk = None
        expanded_idx_to_permuted_idx, moeblk_offset = moeblk
    torch.ops.trtllm.k3_ull_experts_nvfp4(
        input=x.view(torch.float4_e2m1fn_x2),
        input_scale=x_sf.view(torch.uint8),
        weight1=weight_view.w3_w1_weight.view(torch.float4_e2m1fn_x2),
        weight1_scale=weight_view.fc1_weight_scale.view(torch.uint8),
        alpha1=weight_view.fc1_global_scale,
        weight2=weight_view.w2_weight.view(torch.float4_e2m1fn_x2),
        weight2_scale=weight_view.fc2_weight_scale.view(torch.uint8),
        alpha2=weight_view.fc2_global_scale,
        output=moe_output,
        tile_idx_to_group_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        global_sf=backend.fc2_input_scale,
        token_final_scales=token_final_scales,
        counters=counters,
        top_k=top_k,
        # SiTU soft-caps as the leaf holds them: gate_softcap in act_alpha, linear_softcap in
        # act_beta (ull_experts_applies admits SiTU layers only).
        situ_beta=float(backend.act_alpha),
        situ_linear_beta=float(backend.act_beta),
        **slot_kwargs,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        die_major=getattr(backend, "_ull_die_major", False),
        moeblk_offset=int(moeblk_offset),
        gate_up16=bool(fc1_gate_up16(backend)),
    )
    return moe_output
