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
"""Host side of the MiniMax-M3 routed-expert kernel on one TP4 rank (routed and shared intermediate 768 per rank),
over the TRTLLM-Gen NVFP4 buffers read in place.

``m3_moe`` (m3_moe_kernel.py, :class:`M3MoeState`): 1..8 decode tokens (decode batches, spec-decode verify). The call
takes the bf16 hidden states and folds in the router GEMV (bit-exact with m3_router_gemv_fp4) and the activation's
NVFP4 quantization. With ``shared=True`` it also runs the shared expert on dedicated CTAs and returns
bf16(shared + routed); otherwise it returns the rank's routed partial, bit for bit what the stock fused MoE returns.

Usage (build before CUDA-graph capture; every call is capture-safe afterwards)::

    state = M3MoeState(
        device, num_tokens=6, shared=True
    )  # workspace + compiled kernel, shared by all layers
    layer = state.layer(
        w3_w1_weight,
        w3_w1_weight_scale,
        w2_weight,
        w2_weight_scale,
        fc31_alpha,
        fc31_scale_c,
        act_alpha,
        act_beta,
        act_clamp,
        fc2_alpha,
        e_score_correction_bias,
        router_weight=router_weight,
        fc31_input_scale=fc31_input_scale,
        shared_weights=(gate_up_w, gate_up_sf, down_w, down_sf),
    )
    out = layer(hidden)  # bf16 [M, 6144]
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.phase_plan import (
    MoePlanGeometry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.phase_plan import (
    make_phase_plan as _shared_make_phase_plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.runtime import (
    M3MoeLayer as M3MoeLayer,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.runtime import (
    _view as _view,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.runtime import (
    ar_push_desc as ar_push_desc,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.runtime import (
    fine_ranges,
    pretile_shared,
    shared_views,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.runtime import (
    is_supported as is_supported,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.state import (
    M3MoeStateBase,
    MoeStateBindings,
)
from tensorrt_llm.logger import logger

HIDDEN = 6144  # capacity: fixed: the checkpoint's hidden size
INTER = 768  # capacity: fixed: the routed experts' width per tp4 rank
NUM_EXPERTS = 128
SHARED_INTER = 768  # capacity: fixed: the shared expert's width per tp4 rank
SHARED_CTAS = 24  # capacity: fixed: an SM split, not a size (m3_moe's shared-expert CTAs (beside num_ctas routed CTAs))
TOP_K = 4
U1 = 2 * INTER // 32  # FC1 units per expert
# The router GEMV takes at most 2 experts per CTA (router weights held in registers).
MIN_CTAS = 64  # capacity: fixed: an SM split, not a size

_UNIT_BYTES = 32 * (HIDDEN // 2 + HIDDEN // 16)  # one FC1 unit's weight + scale bytes
_ROW2_BYTES = INTER // 2 + INTER // 16  # one FC2 weight row's bytes (values + scales)
_S1 = 6  # FC1 stages (items) per unit  # capacity: fixed: pipeline stages
_ITEM_BYTES = _UNIT_BYTES // _S1  # capacity: fixed: a ring item's bytes
_modules: Dict[tuple, object] = {}
_compiled: Dict[tuple, object] = {}
# m3_moe trace-time options by token count 1..8 (index num_tokens - 1).
# ``sfa_cp``: 18 KB ring stages, the A scales copied by cp.async into the instruction layout (every token count).
SFA_CP_BY_M: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1)
# ``b2img``: FC1 -> FC2 hand-off through per-group global B images + release counters. Off at TP4 (at 4 / 8 tokens
# 0.5 / 0.8 us/call slower than the tagged records in the TP4 sweep).
B2IMG_BY_M: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0)
# ``ar_pre``: the finalize's all-reduce push with its call parities / descriptor preloaded (all but 2 tokens).
AR_PRE_BY_M: Tuple[int, ...] = (1, 0, 1, 1, 1, 1, 1, 1)
# Ring ``stages`` (0 = the kernel's default, the most that fit). 5 at 8 tokens (with a 50 ns FC2 record poll below)
# measured 0.8 us/call faster than 6 in the TP4 sweep (same MMAs, bit-identical outputs); at 4 tokens the default is
# fastest with heavy_fc2_only.
STAGES_BY_M: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 5)
# The FC2 B builders' record poll back-off in ns (0 = the kernel's default, 200); 50 at 8 tokens.
B2_POLL_NS_BY_M: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 50)
# ``early0``: warp 4 streams group 0 right after its own top-k. Off at 8 tokens (0.6 us/layer faster in the chain:
# there it is on every routing warp's critical path and the ring holds only 5 stages).
EARLY0_BY_M: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 0)
# The plan's ``heavy_fc2_only``: 192 FC2 output blocks over 188 routed CTAs leave 4 CTAs with two blocks, whose
# trailing FC2 after the last hand-off otherwise ends the kernel 3-7 us after the rest; they stream no FC1 instead
# (1.7 / 0.9 us/call faster at 4 / 8 tokens in the TP4 sweep; bit-identical outputs).
HEAVY_FC2_ONLY_BY_M: Tuple[int, ...] = (0, 0, 0, 1, 0, 0, 0, 1)
# ``hb_repl``: copies of the tagged FC1 -> FC2 records (applied when > 1); CTA bx polls copy bx % hb_repl, which
# spreads the polls of the few hot record lines over L2. 7 and 8 tokens hand off through the B images instead.
HB_REPL_BY_M: Tuple[int, ...] = (
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
)  # off at TP4 (0.2 us/call slower at 4 tokens)
# ``lg_repl``: replicas of the router logit exchange at up to LG_REPL_MAX_TOKENS tokens (fewer stores before the
# exchange; bit-identical outputs). Above that the kernel's default applies.
LG_REPL = 32
LG_REPL_MAX_TOKENS = 0  # capacity: fixed: measured crossover (off at TP4: flat in the sweep)
# The TP4 FLAT stack: the TP2 FLAT values (no B2 images, 16 router-logit replicas, AR push parities loaded in the
# finalize, PDL release right after the grid wait) over the per-m tables above. Bitwise-identical output (m 1..8 x 4
# routings, eager + graph); standalone m=4 -0.49 us/call (late_trigger). The values are part of the engine's module
# key.
FLAT_STACK = {"b2img": 0, "lg_repl": 16, "ar_pre": 0, "late_trigger": 0}
# FLAT only up to 4 tokens: at 8 tokens the per-m tables are faster (TP2: 2.96 us/call; TP4 in-model at 2 requests
# +175 us/step with FLAT). Bit-identical either way.
FLAT_MAX_TOKENS = 4  # capacity: fixed: measured crossover


def flat_applies(num_tokens: int) -> bool:
    """Whether an engine state for ``num_tokens`` tokens runs the FLAT stack."""
    return num_tokens <= FLAT_MAX_TOKENS


# Byte-balanced FC1 ranges below one unit per CTA (:func:`_fine_ranges`) for groups too small to give every CTA a
# whole unit; False gives such groups whole units on the CTAs with the least FC2 work.
FINE_PLAN = True
# m3_moe phase lag: phase p streams FC1 of group p and FC2 of group p - M3_MOE_LAG (the kernel's default lag).
M3_MOE_LAG = 2


_ENGINE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../kernels/moe/m3_moe_kernel.py")
)


def _kernel_module(config: dict, path: str):
    key = (os.path.basename(path),) + tuple(sorted(config.items()))
    mod = _modules.get(key)
    if mod is None:
        name = (
            f"{__name__}_"
            + os.path.splitext(os.path.basename(path))[0]
            + "_"
            + "_".join(f"{k}{v}" for k, v in key[1:])
        )
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        mod.M3_CONFIG = dict(config, tp_size=4)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _modules[key] = mod
    return mod


# ---------------------------------------------------------------------------------------------------------------------
# m3_moe (m3_moe_kernel.py): 1..8 tokens per call, group-phased weight stream.
# ---------------------------------------------------------------------------------------------------------------------
# Pre-tiled shared weights per layer, keyed by the source buffers (which the engines read in place): every token-count
# state of a layer streams the same copy. An entry holds its sources, so their addresses are not reused while cached.
_PRETILED: Dict[tuple, tuple] = {}


def _pretiled(gu8, gusf8, dn8, dnsf8) -> torch.Tensor:
    key = (gu8.device, gu8.data_ptr(), gusf8.data_ptr(), dn8.data_ptr(), dnsf8.data_ptr())
    hit = _PRETILED.get(key)
    if hit is None:
        hit = _PRETILED[key] = ((gu8, gusf8, dn8, dnsf8), _pretile_shared(gu8, gusf8, dn8, dnsf8))
    return hit[1]


def make_phase_plan(
    num_ctas: int,
    m_max: int,
    lag: int = 2,
    tail_w: float = 1.0,
    early0: bool = False,
    fine: bool = FINE_PLAN,
    heavy_fc2_only: bool = False,
):
    """The work split of ``m3_moe`` for each distinct-expert count G (4 .. 4 m_max): FC2 32-row output blocks in
    contiguous per-CTA ranges (the same for every group), and per group p (slots 4 p ..) a range of its unit-major FC1
    stages per CTA, balancing the bytes each CTA streams in phase p (FC1 of group p plus FC2 of group p - lag, and the
    trailing FC2s with the last group). Ranges are empty or >= 6 stages (a unit is split over at most two CTAs); a
    group too small for every CTA gets whole units on the CTAs with the least FC2 work, or (``fine``) byte-balanced
    ranges of any length with at most one CTA boundary inside a unit (:func:`_fine_ranges`). Returns int32
    [4 m_max + 1, num_ctas, m_max + 1] (per group i0 | i1 << 16, then b0 | nb << 16) and the largest block count."""
    return _shared_make_phase_plan(
        num_ctas=num_ctas,
        m_max=m_max,
        lag=lag,
        tail_w=tail_w,
        early0=early0,
        fine=fine,
        heavy_fc2_only=heavy_fc2_only,
        geometry=MoePlanGeometry(
            top_k=TOP_K,
            hidden=HIDDEN,
            u1=U1,
            stages=_S1,
            row2_bytes=_ROW2_BYTES,
            item_bytes=_ITEM_BYTES,
        ),
        fine_ranges_fn=_fine_ranges,
    )


class M3MoeState(M3MoeStateBase):
    """Workspace + compiled kernel of ``m3_moe`` (1..8 tokens per call) on one device, shared by the layers that run
    on it in one stream order. The call takes the bf16 hidden states and computes the router logits (bit-exact with
    m3_router_gemv_fp4) and the NVFP4 activation itself. ``shared``: the shared expert runs on SHARED_CTAS dedicated
    CTAs beside the ``num_ctas`` routed ones and the call returns bf16(shared + routed). Build before CUDA-graph
    capture; the first call compiles."""

    fused_router = True

    BINDINGS = MoeStateBindings(
        inter=INTER,
        shared_inter=SHARED_INTER,
        engine_path=_ENGINE_PATH,
        compiled=_compiled,
        pretiled=_pretiled,
        cache_module_config=True,
    )

    def __init__(
        self,
        device: torch.device,
        num_tokens: int,
        num_ctas: Optional[int] = None,
        shared: bool = False,
        inter: int = INTER,
    ):
        """``inter``: the routed intermediate per rank; this engine is built for INTER (768) only."""
        if inter != INTER:
            raise ValueError(
                f"m3_moe (TP4): routed intermediate {inter} per rank is not supported (only {INTER})"
            )
        self.inter = inter
        logger.info(f"[M3 MoE] Rubin m3_moe engine (tp4 tables): state for {num_tokens} tokens")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "M3MoeState allocates and compiles: build it before CUDA-graph capture"
            )
        sms_all = torch.cuda.get_device_properties(device).multi_processor_count
        resolved_ctas = (
            (sms_all - SHARED_CTAS if shared else sms_all) if num_ctas is None else num_ctas
        )
        if shared and resolved_ctas + SHARED_CTAS > sms_all:
            raise ValueError(
                f"m3_moe (shared): num_ctas + {SHARED_CTAS} shared-expert CTAs must fit the {sms_all} SMs "
                f"(all resident at once)"
            )
        if not 1 <= num_tokens <= 8:
            raise ValueError(f"m3_moe takes 1..8 tokens per call, not {num_tokens}")
        self.device = torch.device(device)
        ok, why = is_supported(self.device)
        if not ok:
            raise RuntimeError(f"m3_moe: {why}")
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.num_ctas = resolved_ctas
        if not MIN_CTAS <= self.num_ctas <= sms:
            raise ValueError(
                f"m3_moe runs {MIN_CTAS} .. {sms} CTAs (all resident at once), not {self.num_ctas}"
            )
        if self.num_ctas > HIDDEN // 32:
            # a routed CTA that owns no 32-row FC2 output block deadlocks the FC2 hand-off on its second call
            raise ValueError(
                f"m3_moe runs at most {HIDDEN // 32} routed CTAs (one FC2 output block each), not {self.num_ctas}"
            )
        self.m = num_tokens
        # One token: the trailing FC2s weigh nothing in the last phase's balance. Group 0's FC1 ranges are the same
        # for every distinct-expert count (early0).
        plan, nb_cap = make_phase_plan(
            self.num_ctas,
            num_tokens,
            M3_MOE_LAG,
            0.0 if num_tokens == 1 else 1.0,
            True,
            heavy_fc2_only=bool(HEAVY_FC2_ONLY_BY_M[num_tokens - 1]),
        )
        self.config = {
            "m_max": num_tokens,
            "num_ctas": self.num_ctas,
            "nb2_cap": nb_cap,
            "fused_router": 1,
            "timing": 0,
            "i_tp": INTER,
        }
        if HB_REPL_BY_M[num_tokens - 1] > 1:
            self.config["hb_repl"] = HB_REPL_BY_M[num_tokens - 1]
        if num_tokens <= LG_REPL_MAX_TOKENS:
            self.config["lg_repl"] = LG_REPL
        if shared:
            self.config["sh_i"] = SHARED_INTER
            self.config["shared"] = 1
        self.shared = shared
        # 18 KB ring stages (A scales by cp.async into the instruction layout) per token count, unless configured
        self.config["sfa_cp"] = SFA_CP_BY_M[num_tokens - 1]
        # FC1 -> FC2 hand-off through per-group global B images + release counters, per token count
        self.config["b2img"] = B2IMG_BY_M[num_tokens - 1]
        # the finalize's all-reduce push with its call parities / descriptor preloaded, per token count
        self.config["ar_pre"] = AR_PRE_BY_M[num_tokens - 1]
        if STAGES_BY_M[num_tokens - 1] > 0:
            self.config["stages"] = STAGES_BY_M[num_tokens - 1]
        if B2_POLL_NS_BY_M[num_tokens - 1] > 0:
            self.config["b2_poll_ns"] = B2_POLL_NS_BY_M[num_tokens - 1]
        if not EARLY0_BY_M[num_tokens - 1]:
            self.config["early0"] = 0
        if flat_applies(num_tokens):
            self.config.update(FLAT_STACK)
            # engagement line (every rank; once per engine state = once per token count)
            from tensorrt_llm.logger import logger as _flog

            _flog.info(
                "MiniMax-M3 MoE engine FLAT (TP4): "
                + " ".join(f"{k}={self.config[k]}" for k in FLAT_STACK)
                + f" tokens={num_tokens}"
            )
        else:
            from tensorrt_llm.logger import logger as _flog

            _flog.info(
                f"MiniMax-M3 MoE engine FLAT (TP4) off above {FLAT_MAX_TOKENS} tokens (per-m tables): "
                f"tokens={num_tokens}"
            )
        # E0@m4: at 4 tokens the early group-0 FC1 stages are issued by FC2 B builder warp 8 instead of the routing
        # warp 4 (issue order only, bitwise; TP4 -0.19 / TP8 -0.21 us/call).
        if num_tokens == 4:
            from tensorrt_llm.logger import logger as _e0log

            _e0log.info(
                "MiniMax-M3 MoE engine E0M4 (%s): early group-0 stages on warp 8, tokens=4" % "TP4"
            )
        try:
            self.mod = mod = _kernel_module(self.config, _ENGINE_PATH)
        except AssertionError as err:
            raise ValueError(
                f"m3_moe: num_tokens={num_tokens}, num_ctas={self.num_ctas} does not fit the shared "
                f"memory / TMEM budget (fewer CTAs means more FC2 blocks per CTA): {err}"
            ) from err
        # Every CTA waits on others (cross-CTA spins on partials, hand-off records, the shared expert's h and y): the
        # grid must be resident at once, one CTA per SM. The module's shared-memory / TMEM budget assertion makes one
        # CTA fit an SM; the grid must not exceed the SMs. (The kernel also traps after HANG_NS in any such spin.)
        if self.num_ctas + mod.NSH > sms:
            raise ValueError(
                f"m3_moe: a grid of {self.num_ctas} + {mod.NSH} CTAs exceeds the {sms} SMs; its CTAs wait on each "
                "other and must all be resident at once"
            )
        kw = dict(device=self.device)
        self.plan = plan.to(self.device)
        self.partials = torch.full(
            (mod.G_CAP * mod.U1 * num_tokens * 32,), -1, dtype=torch.int32, **kw
        )
        # the FC1 -> FC2 records; with HB_REPL > 1 copies, HB_STRIDE int32 words apart
        hb_words = mod.G_CAP * num_tokens * mod.U1 * 4
        if mod.HB_REPL > 1:
            hb_words = mod.HB_REPL * mod.HB_STRIDE
        self.hbuf32 = torch.zeros(hb_words, dtype=torch.int32, **kw)
        self.epochs = torch.zeros(self.num_ctas + mod.NSH, dtype=torch.int32, **kw)
        self.route_dbg = torch.zeros(num_tokens, 8, dtype=torch.int32, **kw)
        self.tdbg = torch.zeros(self.num_ctas + mod.NSH, 64, dtype=torch.int64, **kw)
        self.lgbuf = torch.zeros(mod.LG_REPL * mod.LG_STRIDE, dtype=torch.int64, **kw)
        self._dummy = torch.zeros(4, dtype=torch.int32, **kw)
        self._ar_off = torch.zeros(8, dtype=torch.int64, **kw)
        # B2IMG: per-group FC2 B images and their unit counters (int32; every group gains CNT_PER_CALL per call)
        self.img32 = torch.zeros(max(1, mod.NG_CAP * mod.IMG_BYTES // 4), dtype=torch.int32, **kw)
        self.cnt = torch.zeros(mod.NG_CAP * 8 + 8, dtype=torch.int32, **kw)
        # shared expert: h and y_sh as (call tag << 16 | bf16), tag 0 = never written
        self.hx = torch.zeros(num_tokens * SHARED_INTER, dtype=torch.int32, **kw)
        self.dnx = torch.zeros(num_tokens * HIDDEN, dtype=torch.int32, **kw)
        self._fn = None


def _shared_views(shared_weights, who):
    return shared_views(shared_weights, who, shared_inter=SHARED_INTER)


def _pretile_shared(gu8, gusf8, dn8, dnsf8, n_cta: int = 24):
    return pretile_shared(gu8, gusf8, dn8, dnsf8, n_cta, shared_inter=SHARED_INTER)


def _fine_ranges(weights, total: int):
    return fine_ranges(weights, total, stages=_S1)
