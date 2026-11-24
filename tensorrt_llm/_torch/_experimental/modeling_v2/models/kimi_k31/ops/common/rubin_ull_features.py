# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 ULL target (``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1``): its recipe (routing's
facts), launch settings and profile selection.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import sys
from typing import Dict, Iterator, Optional, Tuple, Union

from tensorrt_llm._torch.decode_switches import DecodeSwitches
from tensorrt_llm.quantization import QuantAlgo

#: The target's topology: the ULL generation layout.
TOPOLOGIES = ("ull16",)
#: ULL serves generation only.
ROLES = ("generation",)
#: ULL does not select a batch-dependent sub-profile.
ULL_SUBS = ("",)
#: The values a field that encodes a mode may hold: the plain path's and the ones a profile runs.
MODE_VALUES: Dict[str, Tuple[object, ...]] = {
    "attn_res_topology": (None, 1),
    "attn_res_pdl": (0, 2),
    "latent_ar_cute": (0, 1),
    "ld_fused": (0, 1),
    "router_exact": ("0", "kda"),
    "route_tail": (0, 1),
    "dsa_core": ("sparse", "fmha"),
}


@dataclasses.dataclass(frozen=True)
class K31Features:
    """One deployment's settings, read inside the package. Each default is the plain path's value.

    The comment of each field names the environment switch it replaces. Where that switch named a
    mode, the field encodes the mode as written next to it and takes only the values of
    :data:`MODE_VALUES`: a profile with another one fails when it is built, before the model is.
    """

    # Text model (text_model.py) and its kernels.
    # KIMI_K3_ATTN_RES_TOPOLOGY: None = unset (the per-token attn_res kernel), 1 = "split".
    attn_res_topology: Optional[int] = None
    # KIMI_K3_ROUTER_BF16; the profiles hold the value mapping.tp_group_shares_requests() gives on
    # each deployment.
    router_bf16: bool = False
    fc12_prezero: bool = False  # TRTLLM_K3_FC12_PREZERO
    fuse_moe_residual: bool = False  # TRTLLM_K3_FUSE_MOE_RESIDUAL
    ar_finish_fused: bool = False  # TRTLLM_K3_AR_FINISH_FUSED
    ar_push_amid: bool = False  # TRTLLM_K3_AR_PUSH_AMID
    attn_res_early_trigger: bool = False  # TRTLLM_K3_ATTN_RES_EARLY_TRIGGER
    attn_res_mxfp8: bool = False  # TRTLLM_K3_ATTN_RES_MXFP8
    attn_res_pdl: int = 0  # TRTLLM_K3_ATTN_RES_PDL: 0 or 2
    attn_res_pdl_max_m: int = 64  # TRTLLM_K3_ATTN_RES_PDL_MAX_M
    latent_ar_cute: int = 0  # TRTLLM_K3_LATENT_AR_CUTE: 0 = "0", 1 = "1"
    latent_ar_oneshot_max_bytes: int = 0  # TRTLLM_K3_LATENT_AR_ONESHOT_MAX_BYTES
    latent_ar_own_ws: bool = False  # TRTLLM_K3_LATENT_AR_OWN_WS
    latent_ar_own_ws_max_m: int = 32  # TRTLLM_K3_LATENT_AR_OWN_WS_MAX_M
    latent_ar_row: bool = False  # TRTLLM_K3_LATENT_AR_ROW
    latent_ar_row_max_m: int = 128  # TRTLLM_K3_LATENT_AR_ROW_MAX_M
    ld_fused: int = 0  # TRTLLM_K3_LD_FUSED: 0 or 1
    lu_mid_max_m: int = 0  # TRTLLM_K3_LU_MID_MAX_M
    lu_mid_mxfp8: bool = False  # TRTLLM_K3_LU_MID_MXFP8
    mid_ar_push: bool = False  # TRTLLM_K3_MID_AR_PUSH
    mid_ar_push_max_m: int = 0  # TRTLLM_K3_MID_AR_PUSH_MAX_M
    mid_mxfp8_in: bool = False  # TRTLLM_K3_MID_MXFP8_IN
    mid_mxfp8_in_max_m: int = 128  # TRTLLM_K3_MID_MXFP8_IN_MAX_M
    router_exact: str = "0"  # TRTLLM_K3_ROUTER_EXACT: "0" or "kda"
    router_first: bool = False  # TRTLLM_K3_ROUTER_FIRST
    route_approx: bool = False  # TRTLLM_K3_ROUTE_APPROX
    route_tail: int = 0  # TRTLLM_K3_ROUTE_TAIL: 0 = "0", 1 = "1"
    # TRTLLM_K3_ROUTE_TAIL_CFG: "" = the default launch bands, "helix8" = HELIX8_ROUTE_TAIL_CFG.
    route_tail_bands: str = ""
    rp_tail: bool = False  # TRTLLM_K3_RP_TAIL
    shared_sm: int = 0  # TRTLLM_K3_SHARED_SM
    # Global scale of the converted experts' FC2 input, set by the load-time quantization hook.
    fc2_input_global: int = 1  # TLLM_LOAD_QUANTIZE_FC2_INPUT_GLOBAL
    # ULL set of the TEP8 generation server.
    ull_front: bool = False  # TRTLLM_K3_ULL_FRONT
    ull_front_early: bool = False  # TRTLLM_K3_ULL_FRONT_EARLY
    ull_hygiene: bool = False  # TRTLLM_K3_ULL_HYGIENE
    ull_oproj_sandwich: bool = False  # TRTLLM_K3_ULL_OPROJ_SANDWICH
    ull_tail_sandwich: bool = False  # TRTLLM_K3_ULL_TAIL_SANDWICH
    ull2_coll_cp_tail: bool = False  # TRTLLM_K3_ULL2_COLL_CP_TAIL
    # KV cache (ops/ssm/rubin_ull_kda_dsa_cache_manager.py). Zero-based MLA layers whose latent KV
    # is stored in FP8 while the KV cache is NVFP4. None: the last two layers of
    # linear_attn_config.full_attn_layers with an NVFP4 KV cache, none otherwise; (): none; explicit
    # ids: those layers (MLA layers in range, NVFP4 KV cache required). Every profile must hold the
    # same value: a context server hands each layer's KV to the generation server in the layer's
    # own format.
    fp8_kv_layers: Optional[Tuple[int, ...]] = None  # TLLM_KIMI_K3_FP8_KV_LAYERS
    # MLA (mla/).
    mla_gate_mxfp8_in: bool = False  # TRTLLM_K3_MLA_GATE_MXFP8_IN
    # KDA (kda/).
    kda_decode_early_trigger: bool = False  # TRTLLM_K3_KDA_DECODE_EARLY_TRIGGER
    kda_decode_fused_fab: bool = False  # TRTLLM_K3_KDA_DECODE_FUSED_FAB
    kda_fab_multi_tile: bool = False  # TRTLLM_K3_KDA_FAB_MULTI_TILE
    kda_l2_prefetch: bool = False  # TRTLLM_K3_KDA_L2_PREFETCH
    kda_qkvg_stage: bool = False  # TRTLLM_K3_KDA_QKVG_STAGE
    kda_h8_fused_gate: bool = False  # TRTLLM_KDA_H8_FUSED_GATE
    kda_h8_mx_pdl: bool = False  # TRTLLM_KDA_H8_MX_PDL
    # Push all-reduce (comm/).
    ar_push: bool = False  # TRTLLM_K3_AR_PUSH
    # The flat chain's own levers and diagnostics. Each default is the served value; the flat
    # chain's environment switch named in the comment is not read.
    # TLLM_K31_FLAT_DSA_CORE: "sparse" = the sparse DSA core for the rows it serves, "fmha" = the
    # FMHA path everywhere.
    dsa_core: str = "sparse"
    dsa_core_max_rows: int = 8  # TLLM_K31_FLAT_DSA_CORE_MAX_ROWS: widest step on the DSA core
    # Kernel choices a Rubin A/B forces (DSA tactics; no environment switch). "" = the
    # device's table, which is the served value.
    # dsa_inproj_tactic: the DSA in-projection's warp split-K at up to 8 rows (token tile 8),
    # "output_tile:token_tile:k_tile:stages:b_loader_warps" (``DsaInProjection``); the ring may be
    # deeper than vLLM's 16-stage tactic check, up to ``WARP_RING_STAGES_MAX``.
    dsa_inproj_tactic: str = ""
    # selector_gate: the DFlash2 selector's hidden projection (``selector_gate_linear``): "cublas"
    # or one kernel choice "<kind>:<param>:..." (a ``bf16_low_m_gemm`` tactic or "warp:..."), at
    # every row count up to 32.
    selector_gate: str = ""
    # The DFlash2 drafter's side-chain projections at up to one request's verify block (8 rows),
    # ``draft_linear``: draft_kv_stack_gemm, the five layers' stacked history kv_a [2880, 7168];
    # draft_fc_gemm, fc's rows of this rank [448, 43008]. "cublas", or one kernel choice
    # "<kind>:<param>:...": "splitk:..." (a ``bf16_low_m_gemm`` tactic) or
    # "proj:m_tile:n_tile:split_k:stages" (MR8's split-K projection, any split dividing K / 128).
    draft_kv_stack_gemm: str = ""
    draft_fc_gemm: str = ""
    # The fused select-accept's metadata save (``_prepare_attn_metadata_for_dflash``) makes its
    # copy of the device sequence lengths on a side stream forked after the target head, joined
    # after ``select_accept`` (the copy's first reader is the drafter's prepare). Without it the
    # copy is a 4-byte graph memcpy node on the main stream between the target head and
    # ``_local_argmax``. The same bytes either way. Enable this with model_kwargs for diagnostics.
    glue_h1a: bool = False
    # Diagnostics, off on the served path (their code paths are then absent).
    # TLLM_K31_FLAT_DSA_CORE_CHECK: a device check of the core's index operands before each core
    # launch (``kernels/attention/rubin_ull_dsa_core_check.py``).
    dsa_core_check: bool = False
    # TRTLLM_DSA_KEY_SHARD_STATS: every that many host steps, the DSA core logs its mailbox poll
    # timeouts (``[dsa core] rank r: ...``; one device sync); 0 = never.
    dsa_stats_every: int = 0
    # headtp: the first ``RING_FILL_CHECKS`` ring fills of the process also compare the ring's
    # window against every CP rank's owner pages (one more all-gather and a host read each) and
    # log ``check=ok`` or raise; off, a fill logs its K31_RING_FILL line without the check.
    ring_fill_check: bool = False

    def __post_init__(self):
        for name, values in MODE_VALUES.items():
            if getattr(self, name) not in values:
                raise ValueError(
                    f"Kimi K3.1 features: {name}={getattr(self, name)!r} is not one of {values}"
                )
        if self.dsa_core_max_rows < 0 or self.dsa_stats_every < 0:
            raise ValueError(
                "Kimi K3.1 features: dsa_core_max_rows and dsa_stats_every must be >= 0"
            )
        tactic = self.dsa_inproj_tactic.split(":") if self.dsa_inproj_tactic else []
        if tactic and (len(tactic) != 5 or not all(p.isdigit() for p in tactic)):
            raise ValueError(
                f"Kimi K3.1 features: dsa_inproj_tactic={self.dsa_inproj_tactic!r} is not "
                "'output_tile:token_tile:k_tile:stages:b_loader_warps'"
            )
        gate = self.selector_gate.split(":") if self.selector_gate else []
        if gate and gate != ["cublas"] and not all(p.isdigit() for p in gate[1:]):
            raise ValueError(
                f"Kimi K3.1 features: selector_gate={self.selector_gate!r} is not 'cublas' or "
                "'<kind>:<param>:...'"
            )
        for name in ("draft_kv_stack_gemm", "draft_fc_gemm"):
            spec = getattr(self, name)
            parts = spec.split(":") if spec else []
            if (
                parts
                and parts != ["cublas"]
                and not (
                    parts[0] in ("splitk", "proj")
                    and len(parts) == 5
                    and all(p.isdigit() for p in parts[1:])
                )
            ):
                raise ValueError(
                    f"Kimi K3.1 features: {name}={spec!r} is not 'cublas', "
                    "'splitk:mma_m:mma_n:split_k:stages' or 'proj:m_tile:n_tile:split_k:stages'"
                )

    def changed(self) -> Dict[str, object]:
        """The fields that differ from the plain path, by name."""
        base = K31Features()
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != getattr(base, f.name)
        }


@dataclasses.dataclass(frozen=True)
class ProfileKey:
    """The single ULL generation deployment."""

    topology: str
    role: str
    sub: str = ""

    def __post_init__(self):
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"Kimi K3.1 profile topology {self.topology!r}: one of {TOPOLOGIES}")
        if self.role not in ROLES:
            raise ValueError(f"Kimi K3.1 profile role {self.role!r}: one of {ROLES}")
        if self.sub not in ULL_SUBS:
            raise ValueError(f"Kimi K3.1 profile sub {self.sub!r}: one of {ULL_SUBS}")

    def __str__(self) -> str:
        return "/".join(part for part in (self.topology, self.role, self.sub) if part)


#: Fields a deployment sets through its model config: ``LlmArgs.model_kwargs`` key
#: ``kimi_k31_<field>``, which the model loader sets as an attribute of the pretrained config on
#: every rank (:func:`apply_model_config`). Each accepted field defines its value type here.
MODEL_CONFIG_FIELDS: Dict[str, Union[type, Tuple[type, ...]]] = {
    "dsa_core_check": bool,
    "dsa_stats_every": int,
    "ring_fill_check": bool,
    "dsa_inproj_tactic": str,
    "selector_gate": str,
    "draft_kv_stack_gemm": str,
    "draft_fc_gemm": str,
    "glue_h1a": bool,
}
MODEL_CONFIG_PREFIX = "kimi_k31_"

# Package modules that may be imported before select(): they read no feature at import.
_IMPORTABLE_BEFORE_SELECT = ("features",)

_ACTIVE = K31Features()
_SELECTED: Optional[ProfileKey] = None
_PACKAGE = __name__.rpartition(".ops.")[0] + ".kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
#: The target packages whose modules read the profile at import (:func:`select` checks them):
#: the ULL target (``_PACKAGE``) and its wide-batch fork.
_PACKAGES = (
    _PACKAGE,
    __name__.rpartition(".ops.")[0] + ".kimi_k31_pro_nvfp4__sm_107__helix16_wide",
)


def _profiles():
    """This module: PROFILES and SWITCHES per key."""
    return sys.modules[__name__]


def _switchboard():
    """The shared runtime's switchboard, ``tensorrt_llm._torch.decode_switches``."""
    return importlib.import_module("tensorrt_llm._torch.decode_switches")


def active() -> K31Features:
    """The selected deployment's settings (``K31Features()`` before :func:`select`)."""
    return _ACTIVE


def selected() -> Optional[ProfileKey]:
    """The selected deployment, or None."""
    return _SELECTED


def select(key: ProfileKey) -> K31Features:
    """Make ``key``'s profile the active one; once per process, before the package's other modules.

    Also configures the switchboard with the profile's switches. Selecting the same key again
    returns the active settings. Raises if another key was selected, or if a package module that
    reads features was imported first (it would have read the plain path's settings).
    """
    global _ACTIVE, _SELECTED
    if _SELECTED is not None:
        if key != _SELECTED:
            raise RuntimeError(
                f"Kimi K3.1 features: {_SELECTED} is selected in this process, cannot select {key}"
            )
        return _ACTIVE
    exempt = {f"{package}.{name}" for package in _PACKAGES for name in _IMPORTABLE_BEFORE_SELECT}
    early = sorted(
        m
        for m in sys.modules
        if any(m.startswith(package + ".") for package in _PACKAGES) and m not in exempt
    )
    if early:
        raise RuntimeError(
            f"Kimi K3.1 features: package modules were imported before the {key} profile was "
            f"selected, with the plain path's settings: {', '.join(early)}"
        )
    profiles = _profiles()
    if key not in profiles.PROFILES:
        raise KeyError(f"no Kimi K3.1 profile {key}; have {sorted(map(str, profiles.PROFILES))}")
    switchboard = _switchboard()
    # warn_retired=False: the switchboard's retired-variable warning reads the environment.
    switchboard.configure(profiles.SWITCHES[key], source=str(key), warn_retired=False)
    _ACTIVE, _SELECTED = profiles.PROFILES[key], key

    from tensorrt_llm.logger import logger

    changed = _ACTIVE.changed()
    logger.info(
        f"Kimi K3.1 features: {key} ({len(changed)} set): "
        + ", ".join(f"{name}={value}" for name, value in sorted(changed.items()))
    )
    return _ACTIVE


def apply_model_config(pretrained_config) -> K31Features:
    """Apply the :data:`MODEL_CONFIG_FIELDS` that ``pretrained_config`` (the source config, as the
    model loader built it, ``model_kwargs`` applied) sets as ``kimi_k31_<field>`` attributes to
    the active settings; the target calls it once, at construction, before it builds anything
    that reads them. Logs the fields it set; sets and logs nothing when the config sets none."""
    global _ACTIVE
    found = {}
    for name, kind in MODEL_CONFIG_FIELDS.items():
        attr = MODEL_CONFIG_PREFIX + name
        if not hasattr(pretrained_config, attr):
            continue
        value = getattr(pretrained_config, attr)
        kinds = kind if isinstance(kind, tuple) else (kind,)
        if type(value) not in kinds:
            raise TypeError(
                f"Kimi K3.1 features: model config {attr}={value!r} is not a "
                + " or ".join(k.__name__ for k in kinds)
            )
        found[name] = value
    if not found:
        return _ACTIVE
    _ACTIVE = dataclasses.replace(_ACTIVE, **found)
    from tensorrt_llm.logger import logger

    logger.info(
        "Kimi K3.1 features: set by the model config: "
        + ", ".join(f"{name}={value}" for name, value in sorted(found.items()))
    )
    return _ACTIVE


@contextlib.contextmanager
def override(**fields) -> Iterator[K31Features]:
    """Tests: replace some fields of the active settings inside the ``with`` block, then restore."""
    global _ACTIVE
    saved = _ACTIVE
    _ACTIVE = dataclasses.replace(saved, **fields)
    try:
        yield _ACTIVE
    finally:
        _ACTIVE = saved


__all__ = [
    "ULL_SUBS",
    "K31Features",
    "MODEL_CONFIG_FIELDS",
    "MODEL_CONFIG_PREFIX",
    "MODE_VALUES",
    "ProfileKey",
    "ROLES",
    "TOPOLOGIES",
    "active",
    "apply_model_config",
    "override",
    "select",
    "selected",
]


ULL16_GENERATION = ProfileKey("ull16", "generation")
PROFILES = {ULL16_GENERATION: K31Features(router_bf16=True)}
SWITCHES = {
    ULL16_GENERATION: DecodeSwitches(
        kv_v2_zero_gpu_pools=True, dsa_helix_key_shard=True, dsa_helix_fused_meta=True
    )
}


def fc2_input_global(key: ProfileKey, moe_backend: str) -> int:
    """Keep the loader's uncalibrated global scale for this bring-up profile."""
    if key not in PROFILES:
        raise KeyError(f"Unknown ULL profile: {key}")
    return 1


#: Topology -> target (``models/kimi_k31/<target>/``).
TARGETS = {"ull16": "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"}
#: Topology -> its target's wide-batch fork (``routing.py``, under ``TLLM_K31_HELIX16_WIDE``).
WIDE_TARGETS = {"ull16": "kimi_k31_pro_nvfp4__sm_107__helix16_wide"}
TOPOLOGY_ROLES = {"ull16": ("generation",)}
INDEXER_K_DTYPE = "fp8"
#: The engine's KV-cache algorithm (``engine_kv_cache_quant_algo``), by name.
KV_CACHE_QUANT_ALGO = "FP8"
#: CuTe DSL MoE backends: CUTEDSL (context), MEGAMOE_CUTEDSL (DEP16 generation), CUTEDSL_FC12
#: (helix8 and TEP8 generation).
MOE_BACKENDS = ("CUTEDSL",)
#: Speculative decoding mode name of DFlash (``SpeculativeDecodingMode.DFLASH.name``): one-engine
#: DFlash with the DFlash2 MLA drafter.
DFLASH = "DFLASH"
#: The draft length the target verifies (``max_draft_len``): a block of 8 rows per request.
DRAFT_LEN = 7
#: The draft lengths the wide-batch fork (``WIDE_TARGETS``) verifies, one per deployment: k7,
#: k5, k4 and k3 (up to 16 / 21 / 25 / 32 requests in 128 rows); the ULL target k7 only.
WIDE_DRAFT_LENS = (3, 4, 5, DRAFT_LEN)
#: The context servers in front of the ULL generation server.
CONTEXT_TARGETS = ("kimi_k31_pro_nvfp4__sm_107__dep16", "kimi_k31_pro_nvfp4__sm_107__tep8")

RECIPE = (
    "Kimi K3.1 ULL: sm_107, generation, Helix CP16 with attention TP1 and MoE TP16/EP1, "
    "FP8 index keys, FP8 latent KV, native ModelOpt NVFP4/MXFP8 weights, "
    "DFlash2 (one-engine DFlash, the Kimi-K3.1-DFlash2 MLA drafter) at max_draft_len 7, "
    "and a CuTe DSL MoE backend; experimental"
)


@dataclasses.dataclass(frozen=True)
class Layout:
    """The parallel mapping the target is built under, as plain values."""

    world_size: int
    tp_size: int
    pp_size: int
    cp_size: int
    helix: bool  # Helix context parallelism (Mapping.has_cp_helix())
    attention_dp: bool
    moe_tp_size: int
    moe_ep_size: int


@dataclasses.dataclass(frozen=True)
class Decision:
    """A target (``topology`` and ``target``), or the ``reason`` none fits."""

    topology: Optional[str] = None
    target: Optional[str] = None
    reason: Optional[str] = None


def topology(layout: Layout) -> Optional[str]:
    """The exclusive ULL layout: Helix CP16, attention TP1, MoE TP16/EP1."""
    if (
        layout.world_size == 16
        and layout.tp_size == 1
        and layout.pp_size == 1
        and layout.cp_size == 16
        and layout.helix
        and not layout.attention_dp
        and (layout.moe_tp_size, layout.moe_ep_size) == (16, 1)
    ):
        return "ull16"
    return None


def _describe(layout: Layout) -> str:
    return (
        f"world_size={layout.world_size} tp={layout.tp_size} pp={layout.pp_size} "
        f"cp={layout.cp_size} helix={layout.helix} attention_dp={layout.attention_dp} "
        f"moe_tp={layout.moe_tp_size} moe_ep={layout.moe_ep_size}"
    )


def decide(
    *,
    indexer_k_dtype: Optional[str],
    kv_cache_quant_algo: Optional[str],
    spec_mode: Optional[str],
    max_draft_len: Optional[int],
    moe_backend: str,
    layout: Layout,
    role: str,
) -> Decision:
    """The target of a Kimi K3.1 deployment on sm_107 that asks for this recipe.

    ``kv_cache_quant_algo`` is the name of the engine's KV-cache algorithm (``NVFP4``, ``FP8``,
    None for BF16, or the raw dtype string the engine does not map); ``spec_mode`` the
    speculative decoding mode name, None without; ``max_draft_len`` the configured draft length.
    The drafter checkpoint itself (DFlash2, MLA attention) is routing's to check.
    """
    return _decide(
        indexer_k_dtype=indexer_k_dtype,
        kv_cache_quant_algo=kv_cache_quant_algo,
        spec_mode=spec_mode,
        max_draft_len=max_draft_len,
        moe_backend=moe_backend,
        layout=layout,
        role=role,
        draft_lens=(DRAFT_LEN,),
        verifier="ULL",
    )


def decide_wide(
    *,
    indexer_k_dtype: Optional[str],
    kv_cache_quant_algo: Optional[str],
    spec_mode: Optional[str],
    max_draft_len: Optional[int],
    moe_backend: str,
    layout: Layout,
    role: str,
) -> Decision:
    """:func:`decide` for the wide-batch fork (``WIDE_TARGETS``, ``TLLM_K31_HELIX16_WIDE``): the
    same checks with the draft lengths of ``WIDE_DRAFT_LENS``."""
    return _decide(
        indexer_k_dtype=indexer_k_dtype,
        kv_cache_quant_algo=kv_cache_quant_algo,
        spec_mode=spec_mode,
        max_draft_len=max_draft_len,
        moe_backend=moe_backend,
        layout=layout,
        role=role,
        draft_lens=WIDE_DRAFT_LENS,
        verifier="the ULL wide-batch target",
    )


def _decide(
    *,
    indexer_k_dtype: Optional[str],
    kv_cache_quant_algo: Optional[str],
    spec_mode: Optional[str],
    max_draft_len: Optional[int],
    moe_backend: str,
    layout: Layout,
    role: str,
    draft_lens: Tuple[int, ...],
    verifier: str,
) -> Decision:
    """The checks of :func:`decide` (the same body but the draft lengths): ``draft_lens``
    the ``max_draft_len`` values ``verifier`` (the refusal's subject) takes."""
    if indexer_k_dtype != INDEXER_K_DTYPE:
        reason = f"indexer_k_dtype {indexer_k_dtype!r}: ULL requires FP8 index keys"
    elif kv_cache_quant_algo != KV_CACHE_QUANT_ALGO:
        reason = f"KV cache {kv_cache_quant_algo!r}: ULL requires FP8 KV cache"
    elif spec_mode != DFLASH:
        reason = (
            f"speculative decoding {spec_mode}: ULL drafts with DFlash2 (one-engine DFlash with "
            "the DFlash2 MLA drafter)"
        )
    elif max_draft_len not in draft_lens:
        rows = " or ".join(str(k + 1) for k in draft_lens)
        lens = " or ".join(str(k) for k in draft_lens)
        reason = (
            f"max_draft_len {max_draft_len}: {verifier} verifies blocks of {rows} rows "
            f"(max_draft_len {lens})"
        )
    elif str(moe_backend).upper() not in MOE_BACKENDS:
        reason = f"MoE backend {moe_backend}: the targets run {', '.join(MOE_BACKENDS)}"
    elif (name := topology(layout)) is None:
        reason = f"layout {_describe(layout)} is not ULL Helix CP16 / MoE TP16"
    elif role not in TOPOLOGY_ROLES[name]:
        reason = (
            f"{name} serves {' and '.join(TOPOLOGY_ROLES[name])} servers, not a {role} server; "
            f"its context servers run the DFlash2 context targets {' or '.join(CONTEXT_TARGETS)}"
        )
    else:
        return Decision(topology=name, target=TARGETS[name])
    return Decision(reason=f"{reason} ({RECIPE})")


def profile_key(
    topology: str, role: str, max_batch_size: Optional[int], max_draft_len: Optional[int]
) -> ProfileKey:
    """ULL has one fixed profile; batch size does not select code."""
    return ProfileKey(topology, role)


__all__ = [
    "CONTEXT_TARGETS",
    "DFLASH",
    "DRAFT_LEN",
    "Decision",
    "Layout",
    "RECIPE",
    "TARGETS",
    "WIDE_DRAFT_LENS",
    "WIDE_TARGETS",
    "decide",
    "decide_wide",
    "profile_key",
    "topology",
]


def validate_trunk_weight_dtype(source_dtype, parameter_dtype, name: str) -> None:
    """Permit native storage or exact BF16 promotion into an FP32 compute buffer."""
    import torch

    if source_dtype == parameter_dtype or (
        source_dtype == torch.bfloat16 and parameter_dtype == torch.float32
    ):
        return
    raise TypeError(
        f"{name}: checkpoint dtype {source_dtype} != runtime {parameter_dtype}; "
        "ULL forbids implicit weight precision changes except exact BF16-to-FP32 promotion"
    )


def validate_native_config(config, *, check_kv: bool = True) -> None:
    """Reject load-time conversion and unsupported precision before model allocation."""
    pretrained = config.pretrained_config
    source = getattr(pretrained, "quantization_config", {}) or {}
    if source.get("quant_method") != "modelopt_mixed":
        raise ValueError("K3.1 ULL requires the native ModelOpt NVFP4 checkpoint")
    quant = config.quant_config
    if quant is None or quant.quant_algo != QuantAlgo.MIXED_PRECISION:
        raise ValueError("K3.1 ULL requires checkpoint-declared MIXED_PRECISION")
    declarations = config.quant_config_dict or {}
    if not declarations or not any(q.quant_algo == QuantAlgo.MXFP8 for q in declarations.values()):
        raise ValueError("K3.1 ULL requires the checkpoint's native MXFP8 declarations")
    experts = [q for name, q in declarations.items() if name.endswith(".experts")]
    if not experts or any(q.quant_algo != QuantAlgo.NVFP4 for q in experts):
        raise ValueError("K3.1 ULL requires native NVFP4 expert banks")
    if check_kv and quant.kv_cache_quant_algo != QuantAlgo.FP8:
        raise ValueError("K3.1 ULL requires FP8 KV cache")


def is_native_mxfp8(module) -> bool:
    """Whether a Linear owns native E4M3 codes and 1x32 UE8M0 scales."""
    return getattr(getattr(module, "quant_config", None), "quant_algo", None) == QuantAlgo.MXFP8


def shared_quant_config(config, layer: int):
    """Resolve all three shared projections; refuse a recipe the fused MLP cannot represent."""
    declarations = config.quant_config_dict or {}
    matches = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        suffix = f"layers.{layer}.block_sparse_moe.shared_experts.{projection}"
        names = [prefix + suffix for prefix in ("language_model.model.", "model.", "")]
        if any(config.quant_config.is_module_excluded_from_quantization(n) for n in names):
            raise ValueError(f"Native ULL shared projection is excluded: {suffix}")
        entries = [declarations[n] for n in names if n in declarations]
        if not entries or any(q.quant_algo != QuantAlgo.MXFP8 for q in entries):
            raise ValueError(f"Native ULL shared projection must be declared MXFP8: {suffix}")
        matches.append(entries[0])
    if any(q.model_dump() != matches[0].model_dump() for q in matches[1:]):
        raise ValueError("Shared gate/up/down projections have inconsistent quantization")
    return matches[0]
