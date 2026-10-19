# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a NemotronHForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in NemotronH implementation; in ``require`` it raises that
error.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..._router_index import NoTarget

# The block kinds the Ultra target implements. A NemotronH checkpoint with a
# dense-MLP block ("mlp") is a different forward, not a different depth.
_ULTRA_BLOCK_KINDS = frozenset({"mamba", "attention", "moe"})


def _block_kinds(pretrained_config: Any) -> frozenset:
    """The set of block kinds the config stacks, from either spelling the config carries."""
    kinds = getattr(pretrained_config, "layers_block_type", None)
    if kinds:
        return frozenset(kinds)
    pattern = getattr(pretrained_config, "hybrid_override_pattern", "") or ""
    names = {"M": "mamba", "*": "attention", "E": "moe", "-": "mlp"}
    return frozenset(names.get(ch, ch) for ch in pattern)


def _quantization(pretrained_config: Any) -> Tuple[Optional[str], Optional[int]]:
    """(quantization algorithm, kv-cache bits) as the checkpoint's config.json declares them.

    The algorithm is the engine's own `quant_algo` key (what the NVIDIA checkpoints' quantization
    config carries) or, failing that, a Hugging Face native `quant_method`: any declared
    quantization is something other than (None, None), so an unquantized target never claims it.
    """
    qc = getattr(pretrained_config, "quantization_config", None)
    if qc is None:
        return None, None
    get = qc.get if isinstance(qc, dict) else lambda k, d=None: getattr(qc, k, d)
    kv = get("kv_cache_scheme", None) or {}
    kv_bits = kv.get("num_bits") if isinstance(kv, dict) else getattr(kv, "num_bits", None)
    return get("quant_algo", None) or get("quant_method", None), kv_bits


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    cfg = pretrained_config
    # The checkpoint is recognized by its per-layer geometry, the upstream
    # idiom (``is_mla``, ``is_nemotron_hybrid``). Depth is deliberately not in
    # the fingerprint: every weight shape the target declares is per layer, and
    # the stack is built from the config's own block list, so a checkpoint that
    # keeps a prefix of the layers runs the same code. The accuracy gate is what
    # pins the identity of the full checkpoint.
    shape = (
        cfg.hidden_size,
        getattr(cfg, "mamba_num_heads", None),
        getattr(cfg, "mamba_head_dim", None),
        getattr(cfg, "n_groups", None),
        getattr(cfg, "ssm_state_size", None),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        getattr(cfg, "head_dim", None),
        getattr(cfg, "n_routed_experts", None),
        getattr(cfg, "num_experts_per_tok", None),
        getattr(cfg, "moe_intermediate_size", None),
        getattr(cfg, "moe_latent_size", None),
        getattr(cfg, "moe_shared_expert_intermediate_size", None),
    )
    kinds = _block_kinds(cfg)
    quant = _quantization(cfg)

    # The NVFP4 targets put 4 ranks on one checkpoint and differ in where the
    # non-expert weights live -- a weight-layout question, which is why each
    # selects a target rather than a runtime branch. tep4: attention and mamba
    # heads tensor-parallel over the 4 ranks. dep4: attention data-parallel,
    # everything but the routed experts whole on every rank. Both put the routed
    # experts expert-parallel over the same 4 ranks. The BF16 checkpoint is 1.1 TB
    # (the NVFP4 one 329 GB), so it takes 8 ranks, and only the tensor-parallel
    # layout: replicating the non-expert weights (dep) would put about 63 GiB of
    # them on every rank beside 120 GiB of experts: 183 of a 185 GiB GPU before the
    # unsharded fp32 recurrent state (384 MiB per sequence), the KV cache or any activation.
    four_rank_ep = (
        mapping.world_size == 4
        and mapping.tp_size == 4
        and mapping.pp_size == 1
        and mapping.moe_ep_size == 4
        and mapping.moe_tp_size == 1
    )
    eight_rank_ep = (
        mapping.world_size == 8
        and mapping.tp_size == 8
        and mapping.pp_size == 1
        and mapping.moe_ep_size == 8
        and mapping.moe_tp_size == 1
    )
    ultra_shape = (
        shape == (8192, 256, 64, 8, 128, 64, 2, 128, 512, 22, 5120, 2048, 10240)
        and kinds == _ULTRA_BLOCK_KINDS
        # The targets implement no draft path (the checkpoint's MTP module is not
        # loaded), so claiming such a deployment would fail in the model instead
        # of falling back.
        and spec_config is None
    )
    ultra = ultra_shape and compute_capability == (10, 7) and quant == ("MIXED_PRECISION", 8)
    # The BF16 release declares no quantization at all.
    ultra_bf16 = ultra_shape and compute_capability == (10, 0) and quant == (None, None)

    # Nemotron-3 Ultra 550B-A55B NVFP4 on 4 sm_107 GPUs, tep4.
    if ultra and four_rank_ep and not mapping.enable_attention_dp:
        # fmt: off
        # isort: off
        from .nemotron3_ultra_nvfp4__sm_107__tep4.modeling import ModelingV2Nemotron3UltraNvfp4Sm107Tep4
        # isort: on
        # fmt: on

        return ModelingV2Nemotron3UltraNvfp4Sm107Tep4

    # Nemotron-3 Ultra 550B-A55B NVFP4 on 4 sm_107 GPUs, dep4.
    if ultra and four_rank_ep and mapping.enable_attention_dp:
        # fmt: off
        # isort: off
        from .nemotron3_ultra_nvfp4__sm_107__dep4.modeling import ModelingV2Nemotron3UltraNvfp4Sm107Dep4
        # isort: on
        # fmt: on

        return ModelingV2Nemotron3UltraNvfp4Sm107Dep4

    # Nemotron-3 Ultra 550B-A55B BF16 on 8 sm_100 GPUs, tep8.
    if ultra_bf16 and eight_rank_ep and not mapping.enable_attention_dp:
        # fmt: off
        # isort: off
        from .nemotron3_ultra_bf16__sm_100__tep8.modeling import ModelingV2Nemotron3UltraBf16Sm100Tep8
        # isort: on
        # fmt: on

        return ModelingV2Nemotron3UltraBf16Sm100Tep8

    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"block kinds {sorted(kinds)}, quantization {quant}, "
        f"world_size={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"moe_ep={mapping.moe_ep_size} moe_tp={mapping.moe_tp_size} "
        f"attention_dp={mapping.enable_attention_dp}, speculative decoding "
        f"{type(spec_config).__name__ if spec_config is not None else 'off'}; "
        "the nemotron_h targets are Nemotron-3 Ultra 550B-A55B without speculative "
        "decoding: NVFP4 (MIXED_PRECISION, FP8 KV) on sm_107, 4 ranks with expert-parallel "
        "MoE, either tep4 (tensor-parallel attention and mamba) or dep4 (attention data "
        "parallel); and BF16 (no quantization) on sm_100, 8 ranks with expert-parallel MoE "
        "and tensor-parallel attention and mamba (tep8)"
    )
