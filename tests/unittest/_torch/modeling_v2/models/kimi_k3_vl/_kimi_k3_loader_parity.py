# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 targets' checkpoint load (``weights.py``) against the built-in model and loader, each run
by the target's test directory on its own modules.

On one GPU, a scaled-down text model (4 layers: KDA + dense MLP, then MLA, KDA and MLA over the latent MoE; K3's MLA
and KDA geometry, 8 experts) is built twice at one rank's share of a TP group (``Mapping(tp_size=4, rank=r)``):

* the target's own ``KimiLinearModel`` behind a shell of the target class, its all-reduces replaced by a local
  stand-in (``K3AllReduce`` builds a collective workspace at construction), loaded by the target's ``weights.load``;
* the built-in ``KimiLinearForCausalLM`` (``_torch/models/modeling_kimi_linear.py``) with NCCL all-reduces (which
  allocate nothing across ranks at construction), loaded by its own ``load_weights`` with the language-model prefix
  stripped.

Both read the same synthetic safetensors shards through the engine's lazy container, with a
``model.safetensors.index.json`` (the per-shard-file expert stream; one expert split across two files) or without
(the shared lazy fallback). After the loads, every parameter and buffer the two models hold under the same name has
the same bytes, and each fused projection parameter of the target holds the built-in's separate weights in its row
ranges, pad rows zero (``compare_fused``); then both run their post-load (the engine's walk: every module's
``post_load_weights``; the target's root hook is left out, it builds only decode workspaces and collective state) and
the same holds again, together with the derived weights the two hold under different names or as plain tensors
(``compare_derived``). Every name only one side holds must be one of the explained families (``ONLY_BUILTIN``,
``ONLY_TARGET``).

The host-side checks run the target's rejections on stand-in models, and the fused rows' per-key cuts and row ranges on
a small checkpoint (``check_fused_rows``).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import types
import warnings

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

LANG = "language_model."

#: (tp_size, rank, moe_tp, moe_ep, gpus_per_node, routed intermediate): one rank holding everything; the last rank of
#: a TP4 group with its experts split moe_tp 2 x moe_ep 2 (A's kind of layout); and a TP4 rank with every expert split
#: 4 ways to 192 rows that the relayout pads to 256 (B's kind), whose dense MLP the node narrows to a group of 2.
LAYOUTS = {
    "tp1": (1, 0, 1, 1, 8, 512),
    "tp4_moetp2ep2_r3": (4, 3, 2, 2, 8, 512),
    "tp4_moetp4ep1_r2_dense2": (4, 2, 4, 1, 2, 768),
}

#: Names only the built-in model holds, by family, with why the target has none under that name.
ONLY_BUILTIN = {
    # Parameters of the built-in MLA, written by its loader; the target's MLA derives them in post_load_weights as
    # plain tensors (compare_derived).
    r"\.self_attn\.mixer\.(k_b_proj_trans|v_b_proj)$": "MLA absorption weights, derived by the target's MLA",
    # The built-in fused MoE's state beside the four expert buffers (ConfigurableMoE / TRTLLMGenFusedMoE).
    r"\.block_sparse_moe\.routed_experts\.": "built-in fused-MoE internals beside the four expert buffers",
    # The built-in MLA's attention backends (TrtllmAttention mha / mqa) and their RoPE tables.
    r"\.self_attn\.mixer\.(mha|mqa|rotary_emb)\.": "built-in MLA attention-backend state",
    # The built-in's separate projections; the target holds them as row ranges of one parameter (compare_fused).
    r"\.linear_attn\.(q_proj|k_proj|v_proj|g_proj|f_a_proj|b_proj)\.weight$": "KDA rows of k3_proj_weight",
    r"\.self_attn\.mixer\.(kv_a_proj_with_mqa|g_proj)\.weight$": "MLA rows of k3_ag_weight",
}
#: Configuration both models hold under the same name but neither loader writes: the routed experts' SiTU caps per
#: expert. The target's are buffers filled at construction and again by their post-load; the built-in backend's are
#: parameters (TRTLLM-Gen promotes them), so the sentinel overwrites them and its cache_derived_state refills them in
#: the post-load. Compared after the post-load only.
NOT_LOADED = (r"\.block_sparse_moe\.routed_experts\.backend\.act_(alpha|beta)$",)

#: Names only the target holds, by family, with why the built-in model has none under that name.
ONLY_TARGET = {
    r"\.linear_attn\.k3_proj_weight$": "KDA's six input projections in one parameter (compare_fused)",
    r"\.self_attn\.mixer\.k3_ag_weight$": "MLA's [q_a ; kv_a ; g] in one parameter (compare_fused)",
}

#: The fused parameters' sources: the built-in's separate weights, in the fused parameter's row order.
KDA_FUSED = ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj")
MLA_FUSED = ("kv_a_proj_with_mqa", "g_proj")
#: KDA's k3_proj_weight pads its rows with zeros to a multiple of this.
KDA_ROW_MULTIPLE = 8

_SENTINEL_FLOAT = 3.0
_SENTINEL_INT = 0x5A


def sm_100() -> bool:
    """The routed-expert relayout and the KDA derived weights need SM 10.x."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def text_config(moe_intermediate: int):
    """A 4-layer Kimi K3 text config at K3's attention geometry (MLA: 24 heads, 6 per rank at TP4, latent 512, rope
    64, q_lora 1536, nope and v 128, which the target's MLA requires; KDA: head dim 128, conv 4) with a small hidden
    size, vocabulary and expert count. Layer 0 is KDA with the dense MLP; layers 1-3 MLA, KDA, MLA over the MoE."""
    from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig

    return KimiLinearConfig(
        vocab_size=1002,  # not a multiple of 4: the last vocab-parallel lm_head shard is padded
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=24,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_output_gate=True,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        linear_attn_config={
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
            "num_heads": 8,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=moe_intermediate,
        routed_expert_hidden_size=512,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        routed_scaling_factor=2.5,
        moe_router_activation_func="sigmoid",
        moe_renormalize=True,
        num_expert_group=1,
        topk_group=1,
        attn_res_block_size=2,
        latent_moe_use_norm=True,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        torch_dtype=torch.bfloat16,
        tie_word_embeddings=False,
    )


# --- the synthetic checkpoint, written from the config alone ---


def checkpoint_tensors(cfg, seed: int = 0) -> dict[str, torch.Tensor]:
    """Every language-model tensor of the checkpoint at full (unsplit) shape, keys without ``LANG``, plus a key of a
    layer past the config (left unread by both loaders)."""
    g = torch.Generator().manual_seed(seed)
    bf16 = torch.bfloat16
    h, v = cfg.hidden_size, cfg.vocab_size

    def w(*shape, dtype=bf16):
        return (0.05 * torch.randn(shape, generator=g)).to(dtype)

    def u8(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, dtype=torch.uint8)

    lin = cfg.linear_attn_config
    nh, hd, conv = lin["num_heads"], lin["head_dim"], lin["short_conv_kernel_size"]
    proj = nh * hd
    mh = cfg.num_attention_heads
    nope, rope, vd = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    latent, inter = cfg.routed_expert_hidden_size, cfg.moe_intermediate_size
    t = {
        "model.embed_tokens.weight": w(v, h),
        "model.norm.weight": w(h),
        "model.output_attn_res_norm.weight": w(h),
        "model.output_attn_res_proj.weight": w(1, h),
        "lm_head.weight": w(v, h),
        # A layer past num_hidden_layers: both loaders leave it.
        f"model.layers.{cfg.num_hidden_layers}.input_layernorm.weight": w(h),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attention_res_norm",
            "mlp_res_norm",
        ):
            t[p + name + ".weight"] = w(h)
        for name in ("self_attention_res_proj", "mlp_res_proj"):
            t[p + name + ".weight"] = w(1, h)
        a = p + "self_attn."
        if i + 1 in lin["kda_layers"]:
            for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
                t[a + name + ".weight"] = w(proj, h)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                t[a + name + ".weight"] = w(proj, 1, conv)
            # Padded from the head count to the head dim, the padding zero; fp32, cast to the parameter on load.
            a_log = torch.zeros(hd)
            a_log[:nh] = torch.rand(nh, generator=g) + 0.5
            t[a + "A_log"] = a_log
            t[a + "dt_bias"] = w(proj, dtype=torch.float32)
            t[a + "f_a_proj.weight"] = w(hd, h)
            t[a + "f_b_proj.weight"] = w(proj, hd)
            t[a + "b_proj.weight"] = w(nh, h)
            t[a + "o_norm.weight"] = w(hd)
            t[a + "o_proj.weight"] = w(h, proj)
        else:
            t[a + "q_a_proj.weight"] = w(cfg.q_lora_rank, h)
            t[a + "kv_a_proj_with_mqa.weight"] = w(cfg.kv_lora_rank + rope, h)
            t[a + "q_a_layernorm.weight"] = w(cfg.q_lora_rank)
            t[a + "kv_a_layernorm.weight"] = w(cfg.kv_lora_rank)
            t[a + "q_b_proj.weight"] = w(mh * (nope + rope), cfg.q_lora_rank)
            t[a + "kv_b_proj.weight"] = w(mh * (nope + vd), cfg.kv_lora_rank)
            t[a + "g_proj.weight"] = w(mh * vd, h)
            t[a + "o_proj.weight"] = w(h, mh * vd)
        if i < cfg.first_k_dense_replace:
            m = p + "mlp."
            t[m + "gate_proj.weight"] = w(cfg.intermediate_size, h)
            t[m + "up_proj.weight"] = w(cfg.intermediate_size, h)
            t[m + "down_proj.weight"] = w(h, cfg.intermediate_size)
            continue
        m = p + "block_sparse_moe."
        shared = inter * cfg.num_shared_experts
        t[m + "gate.weight"] = w(cfg.num_experts, h)
        t[m + "gate.e_score_correction_bias"] = w(cfg.num_experts, dtype=torch.float32)
        t[m + "shared_experts.gate_proj.weight"] = w(shared, h)
        t[m + "shared_experts.up_proj.weight"] = w(shared, h)
        t[m + "shared_experts.down_proj.weight"] = w(h, shared)
        t[m + "routed_expert_down_proj.weight"] = w(latent, h)
        t[m + "routed_expert_up_proj.weight"] = w(h, latent)
        t[m + "routed_expert_norm.weight"] = w(latent)
        for e in range(cfg.num_experts):
            x = f"{m}experts.{e}."
            for proj_name, rows, cols in (
                ("w1", inter, latent),
                ("w3", inter, latent),
                ("w2", latent, inter),
            ):
                t[x + proj_name + ".weight_packed"] = u8(rows, cols // 2)
                t[x + proj_name + ".weight_scale"] = u8(rows, cols // 32, lo=118, hi=124)
    return t


def write_checkpoint(directory: str, cfg, with_index: bool) -> None:
    """The checkpoint as safetensors shards: the trunk, the vision tower and one file per MoE layer's experts, with
    one expert's w2 in the next layer's file (an expert split across files). With ``with_index``, also the index."""
    tensors = {LANG + k: v for k, v in checkpoint_tensors(cfg).items()}
    tensors["vision_tower.patch_embed.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    tensors["mm_projector.proj.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    files: dict[str, dict[str, torch.Tensor]] = {}
    moe_layers = list(range(cfg.first_k_dense_replace, cfg.num_hidden_layers))
    for key, value in tensors.items():
        name = "model-trunk.safetensors"
        if ".block_sparse_moe.experts." in key:
            layer = int(key.split("model.layers.")[1].split(".")[0])
            if layer == moe_layers[0] and ".experts.1.w2." in key:
                layer = moe_layers[1]
            name = f"model-experts-{layer}.safetensors"
        files.setdefault(name, {})[key] = value.contiguous()
    weight_map = {}
    for name, part in files.items():
        save_file(part, os.path.join(directory, name))
        weight_map.update({key: name for key in part})
    if with_index:
        with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)


def lazy_weights(directory: str):
    """The checkpoint as the engine hands it to ``load_weights``: lazy safetensors slices with ``checkpoint_dir``."""
    from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader

    return HfWeightLoader()._load_lazy_safetensors(directory)


# --- the two models ---


class LocalAllReduce(nn.Module):
    """Stands in for ``K3AllReduce`` in one process: no workspace, no MNNVL, the identity. Loading runs no
    collective."""

    def __init__(self, mapping=None, strategy=None, dtype=None):
        super().__init__()
        self.mapping = mapping
        self.mnnvl_allreduce = None

    def forward(self, x, *args, **kwargs):
        return x


def _mapping(layout: str):
    from tensorrt_llm.mapping import Mapping

    tp, rank, moe_tp, moe_ep, gpus_per_node, _ = LAYOUTS[layout]
    return Mapping(
        world_size=tp,
        tp_size=tp,
        rank=rank,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        gpus_per_node=gpus_per_node,
    )


def _model_config(layout: str, strategy):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig

    return ModelConfig(
        pretrained_config=text_config(LAYOUTS[layout][5]),
        mapping=_mapping(layout),
        quant_config=QuantConfig(),
        skip_create_weights_in_init=False,
        allreduce_strategy=strategy,
        moe_backend="TRTLLM",
    )


def _sentinel(model: nn.Module) -> None:
    """Every parameter holds a sentinel first, so a tensor one loader writes and the other does not shows."""
    with torch.no_grad():
        for p in model.parameters():
            if p.numel():
                p.fill_(_SENTINEL_FLOAT if p.is_floating_point() else _SENTINEL_INT)


def build_target(target_pkg: str, target_cls, layout: str, monkeypatch):
    """The target's text model at one rank of ``layout`` behind a shell of the target class (built without its
    ``__init__``, whose checks need 16 ranks of SM 10.0) holding the stock lm_head the shell builds, with
    ``K3AllReduce`` replaced by ``LocalAllReduce`` in every module of the target that builds one."""
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm.functional import AllReduceStrategy

    for name in ("modeling", "mla", "kda", "moe"):
        module = importlib.import_module(f"{target_pkg}.{name}")
        if hasattr(module, "K3AllReduce"):
            monkeypatch.setattr(module, "K3AllReduce", LocalAllReduce)
    modeling = importlib.import_module(f"{target_pkg}.modeling")
    config = _model_config(layout, AllReduceStrategy.AUTO)
    cfg = config.pretrained_config
    with torch.device("cuda"):
        text = modeling.KimiLinearModel(config)
        # The stock LMHead the shell builds; NCCL so its all-reduce allocates nothing across ranks.
        lm_head = LMHead(
            cfg.vocab_size,
            cfg.hidden_size,
            dtype=torch.bfloat16,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            allreduce_strategy=AllReduceStrategy.NCCL,
        )
    shell = target_cls.__new__(target_cls)
    nn.Module.__init__(shell)
    shell.model = text
    shell.lm_head = lm_head
    shell.model_config = config
    _sentinel(shell)
    return shell


def build_builtin(layout: str):
    """The built-in Kimi K3 causal LM at one rank of ``layout``, NCCL all-reduces."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
    from tensorrt_llm.functional import AllReduceStrategy

    with torch.device("cuda"):
        model = KimiLinearForCausalLM(_model_config(layout, AllReduceStrategy.NCCL))
    _sentinel(model)
    return model


def builtin_load(model, weights) -> None:
    """The built-in loader as the target called it before: the language-model prefix stripped."""
    from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
    from tensorrt_llm._torch.models.modeling_utils import filter_weights

    lm_weights = ConsumableWeightsDict(filter_weights(LANG[:-1], weights))
    checkpoint_dir = getattr(weights, "checkpoint_dir", None)
    if checkpoint_dir is not None:
        lm_weights.checkpoint_dir = checkpoint_dir
    lm_weights.checkpoint_prefix = LANG
    model.load_weights(lm_weights)


def post_load(model, include_root: bool) -> None:
    """The engine's post-load walk: every module's ``post_load_weights`` in ``modules()`` order (the root first)."""
    for module in model.modules():
        if module is model and not include_root:
            continue
        hook = getattr(module, "post_load_weights", None)
        if hook is not None:
            hook()
    torch.cuda.synchronize()


# --- the comparison ---


def _raw(t: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1).contiguous().view(torch.uint8)


def _same(x, y) -> str | None:
    """None if the two hold the same bytes, else why not."""
    if x is None or y is None:
        return f"missing ({'built-in' if x is None else 'target'} is None)"
    if x.shape != y.shape or x.dtype != y.dtype:
        return f"{tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}"
    if x.numel() and not torch.equal(_raw(x), _raw(y)):
        if x.numel() <= 16:
            return f"bytes differ: built-in {x.flatten().tolist()} target {y.flatten().tolist()}"
        return "bytes differ"
    return None


def _named(model) -> dict[str, torch.Tensor]:
    out = {n: p for n, p in model.named_parameters()}
    out.update({n: b for n, b in model.named_buffers()})
    return out


def _unexplained(names, families: dict) -> list:
    return sorted(n for n in names if not any(re.search(rule, n) for rule in families))


def compare_common(builtin, target, stage: str) -> int:
    """Every name both models hold has the same bytes; every name one side holds alone is an explained family.
    Returns the number of tensors compared."""
    a, b = _named(builtin), _named(target)
    only_builtin = _unexplained(set(a) - set(b), ONLY_BUILTIN)
    only_target = _unexplained(set(b) - set(a), ONLY_TARGET)
    assert not only_builtin and not only_target, (
        f"{stage}: unexplained names only the built-in model holds {only_builtin[:12]}, "
        f"only the target holds {only_target[:12]}"
    )
    common = sorted(set(a) & set(b))
    if stage == "after the load":
        common = [n for n in common if not any(re.search(rule, n) for rule in NOT_LOADED)]
    diffs = [(n, why) for n in common if (why := _same(a[n], b[n])) is not None]
    assert not diffs, f"{stage}: {len(diffs)} of {len(common)} common tensors differ: {diffs[:10]}"
    return len(common)


def _shares_storage(view: torch.Tensor, base: torch.Tensor) -> bool:
    return view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


def _is_rows_view(view: torch.Tensor, base: torch.Tensor, start: int, stop: int) -> bool:
    """``view`` is rows ``[start, stop)`` of ``base``, sharing its storage (no copy)."""
    return (
        _shares_storage(view, base)
        and view.data_ptr() == base[start:].data_ptr()
        and tuple(view.shape) == (stop - start, base.shape[1])
    )


def _padded_cat(weights: list[torch.Tensor], multiple: int) -> torch.Tensor:
    pad = (-sum(w.shape[0] for w in weights)) % multiple
    return torch.cat(weights + [weights[0].new_zeros(pad, weights[0].shape[1])])


def as_target_names(names) -> set[str]:
    """The built-in model's parameter names as the target names them: each separate projection that the target
    holds in a fused parameter becomes that parameter's name."""
    out = set()
    for name in names:
        name = re.sub(
            r"\.linear_attn\.(%s)\.weight$" % "|".join(KDA_FUSED),
            ".linear_attn.k3_proj_weight",
            name,
        )
        name = re.sub(
            r"\.self_attn\.mixer\.(%s)\.weight$" % "|".join(MLA_FUSED),
            ".self_attn.mixer.k3_ag_weight",
            name,
        )
        out.add(name)
    return out


def compare_fused(builtin, target, stage: str) -> int:
    """Each fused projection parameter of the target holds, byte for byte, the built-in's separate weights in its row
    ranges: KDA's ``k3_proj_weight`` = ``[q | k | v | g | f_a | b]`` and zero rows up to a multiple of 8, MLA's
    ``k3_ag_weight`` = ``[kv_a_proj_with_mqa ; g_proj]`` (the built-in's kv_a_proj_with_mqa is ``[q_a ; kv_a]``).
    Returns the number of tensors compared."""
    diffs, count = [], 0
    for i, (old, new) in enumerate(zip(builtin.model.layers, target.model.layers)):
        if new.is_kda:
            o, fused = old.linear_attn, new.linear_attn.k3_proj_weight
            parts = [getattr(o, name).weight.detach() for name in KDA_FUSED]
            expected = _padded_cat(parts, KDA_ROW_MULTIPLE)
            rows = sum(w.shape[0] for w in parts)
            if fused.shape[0] > rows and fused[rows:].abs().amax().item() != 0:
                diffs.append((f"{i}.linear_attn.k3_proj_weight", "pad rows not zero"))
            name = f"{i}.linear_attn.k3_proj_weight"
        else:
            o, fused = old.self_attn.mixer, new.self_attn.mixer.k3_ag_weight
            expected = torch.cat([getattr(o, name).weight.detach() for name in MLA_FUSED])
            name = f"{i}.self_attn.mixer.k3_ag_weight"
        count += 1
        why = _same(expected, fused)
        if why is not None:
            diffs.append((name, why))
    assert not diffs, f"{stage}: {len(diffs)} of {count} fused parameters differ: {diffs[:10]}"
    return count


def compare_derived(builtin, target) -> int:
    """After both post-loads: the derived weights, which the two hold under different names or as plain tensors.

    MLA: ``k_b_proj_trans`` and ``v_b_proj`` (built-in parameters, target tensors; ``v_b_proj`` a view of kv_b), and
    the target's generic-path weights ``a_proj_weight`` / ``g_proj_weight`` (the built-in's ``kv_a_proj_with_mqa`` /
    ``g_proj``), row views of ``k3_ag_weight``. KDA: the fused-projection rows and decode constants under the same
    attribute names, ``_qkvg_proj_weight`` and ``_bfa_proj_weight`` row views of the target's ``k3_proj_weight``
    (which equals the built-in's ``[_qkvg_proj_weight ; _bfa_proj_weight]``), and its fp32 conv taps
    (``_conv_weights_f32``, the built-in ``_mtp_conv_weights``)."""
    diffs, count = [], 0

    def check(name, x, y):
        nonlocal count
        count += 1
        why = _same(x, y)
        if why is not None:
            diffs.append((name, why))

    for i, (old, new) in enumerate(zip(builtin.model.layers, target.model.layers)):
        if new.is_kda:
            o, n = old.linear_attn, new.linear_attn
            for attr in (
                "_qkvg_proj_weight",
                "_bfa_proj_weight",
                "_w_q_t",
                "_w_k_t",
                "_w_v_t",
                "_A_log_f32",
                "_dt_bias_f32",
                "_onorm_w_f32",
                "_packed_conv_weight",
            ):
                check(f"{i}.linear_attn.{attr}", getattr(o, attr), getattr(n, attr))
            fused = (
                torch.cat([o._qkvg_proj_weight, o._bfa_proj_weight])
                if o._qkvg_proj_weight is not None
                else None
            )
            check(f"{i}.linear_attn.k3_proj_weight", fused, n.k3_proj_weight)
            for k, (x, y) in enumerate(zip(o._mtp_conv_weights, n._conv_weights_f32)):
                check(f"{i}.linear_attn._conv_weights_f32[{k}]", x, y)
            qkvg_rows = n._qkvg_proj_weight.shape[0]
            for attr, start, stop in (
                ("_qkvg_proj_weight", 0, qkvg_rows),
                ("_bfa_proj_weight", qkvg_rows, n.k3_proj_weight.shape[0]),
            ):
                if not _is_rows_view(getattr(n, attr), n.k3_proj_weight, start, stop):
                    diffs.append((f"{i}.linear_attn.{attr}", "not a row view of k3_proj_weight"))
        else:
            o, n = old.self_attn.mixer, new.self_attn.mixer
            check(f"{i}.k_b_proj_trans", o.k_b_proj_trans, n.k_b_proj_trans)
            check(f"{i}.v_b_proj", o.v_b_proj, n.v_b_proj)
            check(
                f"{i}.k3_ag_weight",
                torch.cat([o.kv_a_proj_with_mqa.weight, o.g_proj.weight]),
                n.k3_ag_weight,
            )
            a_rows = o.kv_a_proj_with_mqa.weight.shape[0]
            for attr, built_in, start, stop in (
                ("a_proj_weight", o.kv_a_proj_with_mqa.weight, 0, a_rows),
                ("g_proj_weight", o.g_proj.weight, a_rows, n.k3_ag_weight.shape[0]),
            ):
                check(f"{i}.{attr}", built_in, getattr(n, attr))
                if not _is_rows_view(getattr(n, attr), n.k3_ag_weight, start, stop):
                    diffs.append((f"{i}.{attr}", "not a row view of k3_ag_weight"))
            if not _shares_storage(n.v_b_proj, n.kv_b_proj.weight):
                diffs.append((f"{i}.v_b_proj", "not a view of kv_b_proj.weight"))
    assert not diffs, f"{len(diffs)} of {count} derived tensors differ: {diffs[:10]}"
    return count


def check_load_matches_builtin(
    target_pkg: str, weights_mod, target_cls, layout: str, with_index: bool, tmp_path, monkeypatch
) -> None:
    """The target's model loaded by its ``weights.load`` and the built-in model loaded by the built-in loader, on the
    same synthetic checkpoint at the same rank, hold the same bytes after the load and after their post-loads, and
    read the same checkpoint keys into the same parameters."""
    monkeypatch.delenv("KIMI_K3_FP8_WEIGHT_READ_MOE_MLP", raising=False)
    tp, _, _, _, gpus_per_node, inter = LAYOUTS[layout]
    write_checkpoint(str(tmp_path), text_config(inter), with_index)
    weights = lazy_weights(str(tmp_path))
    builtin = build_builtin(layout)
    target = build_target(target_pkg, target_cls, layout, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builtin_load(builtin, weights)
        weights_mod.load(target, weights)

    # The same plan: every parameter the built-in loader maps, read from the same keys.
    name_map, expected, _ = builtin.checkpoint_name_plan("")
    d = weights_mod.dims(target)
    rows = weights_mod.manifest(target, d)
    assert {row.param for row in rows} == as_target_names(name_map)
    assert weights_mod.expected_keys(target, d, rows) == expected
    # The dense MLP's share follows the layer's TP group.
    assert target.model.layers[0].mlp_tp_size == (2 if gpus_per_node == 2 else tp)

    assert compare_common(builtin, target, "after the load") > 0
    assert compare_fused(builtin, target, "after the load") == len(target.model.layers)
    post_load(builtin, include_root=True)
    post_load(target, include_root=False)
    assert compare_common(builtin, target, "after the post-load") > 0
    assert compare_fused(builtin, target, "after the post-load") == len(target.model.layers)
    assert compare_derived(builtin, target) > 0


def check_load_rejects_missing_and_unknown_keys(
    target_pkg: str, weights_mod, target_cls, tmp_path, monkeypatch
) -> None:
    """A missing language-model key fails naming it; a key outside the language model and the vision tower fails; an
    FP8 tensor fails."""
    cfg = text_config(512)
    write_checkpoint(str(tmp_path), cfg, with_index=True)
    weights = lazy_weights(str(tmp_path))
    model = build_target(target_pkg, target_cls, "tp1", monkeypatch)
    missing = dict(weights)
    del missing[LANG + "model.layers.1.self_attn.kv_b_proj.weight"]
    with pytest.raises(KeyError, match="kv_b_proj"):
        weights_mod.load(model, missing)
    unknown = dict(weights)
    unknown["audio_tower.weight"] = torch.zeros(1)
    with pytest.raises(AssertionError, match="audio_tower"):
        weights_mod.load(model, unknown)
    fp8 = dict(weights)
    fp8[LANG + "model.layers.0.self_attn.q_proj.weight"] = torch.zeros(
        1024, 256, dtype=torch.float8_e4m3fn
    )
    with pytest.raises(NotImplementedError, match="float8"):
        weights_mod.load(model, fp8)


# --- host side: the rejections, on stand-in models ---


def _stand_in(
    cp_size=1,
    attention_dp=False,
    pp_size=1,
    fp8_param=False,
    quant_config_dict=None,
):
    mapping = types.SimpleNamespace(
        cp_size=cp_size, enable_attention_dp=attention_dp, pp_size=pp_size
    )
    model = types.SimpleNamespace(
        model_config=types.SimpleNamespace(mapping=mapping, quant_config_dict=quant_config_dict),
        model=types.SimpleNamespace(layers=[]),
    )
    dtype = torch.float8_e4m3fn if fp8_param else torch.bfloat16
    params = {"model.layers.1.self_attn.mixer.q_b_proj.weight": torch.empty(1, dtype=dtype)}
    return model, params


REJECTIONS = {
    "helix": (dict(cp_size=2), NotImplementedError, "helix"),
    "attention_dp": (dict(attention_dp=True), NotImplementedError, "attention DP"),
    "pipeline": (dict(pp_size=2), NotImplementedError, "pipeline"),
    "fp8_attention_param": (dict(fp8_param=True), NotImplementedError, "FP8 attention"),
    "fp8_attention_declared": (
        dict(quant_config_dict={"q_b_proj": object()}),
        NotImplementedError,
        "FP8 attention",
    ),
}


def check_rejects(weights_mod, case: str) -> None:
    kwargs, error, match = REJECTIONS[case]
    model, params = _stand_in(**kwargs)
    d = types.SimpleNamespace(moe_layers=[], expert_offset=0)
    with pytest.raises(error, match=match):
        weights_mod._check_supported(model, {}, d, params)


def check_rejects_nvfp4_experts(weights_mod) -> None:
    """An expert stored as NVFP4 (``weight``, ``weight_scale_2``, ...) instead of packed MXFP4 fails naming the key."""
    model, params = _stand_in()
    model.model.layers = [types.SimpleNamespace() for _ in range(2)]
    d = types.SimpleNamespace(moe_layers=[1], expert_offset=4)
    nvfp4 = {LANG + "model.layers.1.block_sparse_moe.experts.4.w1.weight": None}
    with pytest.raises(NotImplementedError, match="experts.4.w1.weight_packed"):
        weights_mod._check_supported(model, nvfp4, d, params)


# --- host side: the fused rows' cuts and row ranges ---


def check_fused_rows(weights_mod) -> None:
    """The KDA and MLA fused rows on a small checkpoint at rank 1 of TP 2: every source key lands in its row range of
    the fused parameter with this rank's cut (q / k / v / g and b by head, f_a, q_a and kv_a whole, MLA's g by head),
    and KDA's pad rows are zero. Every checkpoint row holds a distinct value, so a wrong cut or order shows."""
    tp, rank, hidden = 2, 1, 3
    kda_heads, head_dim = 3, 4  # 4 * 12 + 4 + 3 = 55 rows: one pad row
    mla_heads, v_dim, q_lora, kv_a_rows = 2, 4, 5, 6
    d = types.SimpleNamespace(
        tp=tp,
        rank=rank,
        kda_heads=kda_heads,
        kda_proj=kda_heads * head_dim,
        mla_heads=mla_heads,
        qk_dim=6,
        nope=4,
        v_dim=v_dim,
        kv_lora=4,
    )
    full_rows = {
        "q_proj": tp * d.kda_proj,
        "k_proj": tp * d.kda_proj,
        "v_proj": tp * d.kda_proj,
        "g_proj": tp * d.kda_proj,
        "f_a_proj": head_dim,
        "b_proj": tp * kda_heads,
    }
    kda_cuts = {name: rank * rows // tp for name, rows in full_rows.items() if name != "f_a_proj"}
    kda_cuts["f_a_proj"] = None
    mla_rows = {
        "q_a_proj": q_lora,
        "kv_a_proj_with_mqa": kv_a_rows,
        "g_proj": tp * mla_heads * v_dim,
    }
    mla_cuts = {"q_a_proj": None, "kv_a_proj_with_mqa": None, "g_proj": rank * mla_heads * v_dim}

    def tensor(tag: int, rows: int) -> torch.Tensor:
        return (
            (tag * 1000 + torch.arange(rows, dtype=torch.float32))[:, None]
            .expand(rows, hidden)
            .contiguous()
        )

    kda, mla = "model.layers.0.self_attn.", "model.layers.1.self_attn."
    ckpt = {}
    for tag, (name, rows) in enumerate(full_rows.items(), start=1):
        ckpt[LANG + kda + name + ".weight"] = tensor(tag, rows)
    for tag, (name, rows) in enumerate(mla_rows.items(), start=11):
        ckpt[LANG + mla + name + ".weight"] = tensor(tag, rows)

    def expected(prefix: str, rows: dict, cuts: dict, local: dict) -> list[torch.Tensor]:
        out = []
        for name in rows:
            t = ckpt[LANG + prefix + name + ".weight"]
            start = cuts[name]
            out.append(t if start is None else t[start : start + local[name]])
        return out

    kda_local = {name: rows // tp for name, rows in full_rows.items()}
    kda_local["f_a_proj"] = head_dim
    mla_local = {"q_a_proj": q_lora, "kv_a_proj_with_mqa": kv_a_rows, "g_proj": mla_heads * v_dim}

    for rows, param, prefix, sources, cuts, local, multiple in (
        (
            weights_mod._kda_rows(0, d),
            "model.layers.0.linear_attn.k3_proj_weight",
            kda,
            full_rows,
            kda_cuts,
            kda_local,
            KDA_ROW_MULTIPLE,
        ),
        (
            weights_mod._mla_rows(1, d),
            "model.layers.1.self_attn.mixer.k3_ag_weight",
            mla,
            mla_rows,
            mla_cuts,
            mla_local,
            1,
        ),
    ):
        (row,) = [r for r in rows if r.param == param]
        assert row.keys == tuple(prefix + name + ".weight" for name in sources)
        parts = expected(prefix, sources, cuts, local)
        want = _padded_cat(parts, multiple)
        out = torch.full_like(want, -1.0)
        weights_mod._fill(out, row, ckpt, d)
        assert torch.equal(out, want), f"{param}: rows differ from the cut, concatenated sources"
        start = 0
        for name, part in zip(sources, parts):
            assert torch.equal(out[start : start + part.shape[0]], part), (
                f"{param}: {name}'s row range"
            )
            start += part.shape[0]
        assert out[start:].shape[0] == (-start) % multiple and not out[start:].any(), (
            f"{param}: pad rows"
        )
        # No other row reads a fused source key.
        others = {key for r in rows if r is not row for key in r.keys}
        assert not others & set(row.keys), f"{param}: a source key is also another row's"


def check_row_cuts(weights_mod) -> None:
    """A row's cut per key: one `cut` applies to every key, `cuts` gives one per key, and a row may not give both."""
    cut = (slice(2, 4),)
    assert weights_mod.Row("p", ("a", "b"), cut).key_cuts() == (cut, cut)
    assert weights_mod.Row("p", ("a",)).key_cuts() == (None,)
    assert weights_mod.Row("p", ("a", "b"), cuts=(None, cut)).key_cuts() == (None, cut)
    with pytest.raises(AssertionError, match="one cut per key"):
        weights_mod.Row("p", ("a", "b"), cut, cuts=(None, cut)).key_cuts()
    with pytest.raises(AssertionError, match="one cut per key"):
        weights_mod.Row("p", ("a", "b"), cuts=(cut,)).key_cuts()
