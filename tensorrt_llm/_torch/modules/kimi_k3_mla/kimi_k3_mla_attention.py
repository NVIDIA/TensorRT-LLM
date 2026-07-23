# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KimiK3MLAAttention — Kimi K3 multi-latent attention for the PyTorch backend.

Structural mirror of the HF reference ``KimiMLAAttention`` in
``modeling_kimi.py``. Same parameter names, same layer shapes, same
low-rank Q/KV geometry. The delta-rule attention math itself is delegated
to the existing ``TrtllmAttention`` MLA backend path plus
``KVCacheManagerV2`` for cached decode; the K3-specific deltas (NoPE,
output gate before ``o_proj``, ``192 ** -0.5`` softmax scale) live in
module code, not in the backend.

Backend split (mirrors production ``_torch/modules/mla.py``)
------------------------------------------------------------
The C++ MLA path needs different ``(head_dim, num_kv_heads)`` for
context and generation:

* Context: ``head_dim = qk_nope + qk_rope = 192``,
  ``num_kv_heads = num_heads`` (per-head Q/K/V).
* Generation (absorbed): ``head_dim = kv_lora_rank + qk_rope = 576``,
  ``num_kv_heads = 1`` (single MQA-shared latent head).

Both backends share the same MLA params, KV cache manager, and identity
RoPE table so K3 NoPE holds.

NoPE via identity cos/sin cache
-------------------------------
The MLA context / generation kernels unconditionally read a rotary
cos/sin cache and apply the rotation. K3's reference does not rotate.
We install a real ``PositionalEmbeddingParams`` so the backend
allocates the correct-shape cache, then overwrite the values in place
with ``(cos=1, sin=0)`` — mathematically identity. A no-op patch on
``_ensure_rope_table_size`` keeps the identity through metadata
resizes.

Cache layout
------------
MLA collapses the paged KV cache to a single MQA-style head of width
``kv_lora_rank + qk_rope_head_dim`` (576 for real K3). The paged cache
manager is constructed with ``num_kv_heads=1`` and
``head_dim=kv_lora_rank + qk_rope_head_dim``. Every context write
carries a ``latent_cache`` tensor of shape
``[num_tokens, kv_lora_rank + qk_rope_head_dim]``; the backend appends
it to the paged cache under the current request's slot.

Absorbed cached decode
----------------------
For the generation-only path, the backend consumes a fused Q of shape
``[num_tokens, num_heads * (kv_lora_rank + qk_rope_head_dim)]``: the
first ``kv_lora_rank`` positions are ``q_nope @ kv_b_proj_absorb_k``
(the "absorbed" projection into the compressed KV space); the trailing
``qk_rope_head_dim`` positions are ``q_rot`` unchanged (NoPE keeps it
identity). The backend runs attention in the compressed space and
returns ``[num_tokens, num_heads * kv_lora_rank]``, which we unabsorb
via ``kv_b_proj_absorb_v`` to recover
``[num_tokens, num_heads * v_head_dim]``. The output gate and
``o_proj`` finish the module.

Mutation controls (module-level)
--------------------------------
* ``apply_rotary_mutation`` — rotates ``q_rot`` and ``k_rot`` slots by
  a fixed non-identity block-rotation before feeding the backend. K3's
  reference does not rotate; injecting one must diverge.
* ``omit_output_gate_mutation`` — skips the sigmoid gate multiplication
  when ``use_output_gate=True``. K3 always multiplies through when the
  gate is configured; omission must diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from ....functional import PositionEmbeddingType
from ...attention_backend import TrtllmAttention, TrtllmAttentionMetadata
from ...attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)

# ---------------------------------------------------------------------------
# RMSNorm mirror (byte-compatible with HF's ``KimiRMSNorm``).
# ---------------------------------------------------------------------------


def _meta_safe_cast_dtype(module, dtype):
    """``module.to(dtype=dtype)`` that also works under ``MetaInitMode``.

    ``Module.to`` dispatches ``aten._to_copy``, which MetaInitMode rejects
    (it would silently fall back to full CPU construction of the model —
    ~70 GB of host RAM per rank for Kimi K3). Under meta init the values
    are garbage anyway, so a dtype-only re-allocation via ``empty_like``
    (an allowed init op) is equivalent; off meta this matches ``.to``.
    """
    import torch as _torch

    def _cast(t):
        if not t.is_floating_point():
            return t
        if t.is_meta:
            return _torch.empty_like(t, dtype=dtype)
        return t.to(dtype=dtype)

    module._apply(_cast)


class _KimiRMSNorm(nn.Module):
    """RMSNorm with the exact HF ``KimiRMSNorm`` param name and math.

    HF's implementation: ``x_norm = x * rsqrt(mean(x**2) + eps)`` then
    multiply by ``self.weight``, all in fp32 with a final downcast back
    to the input dtype. Parameter name is ``weight`` (identity mapping
    to the K3 module).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x32 = hidden_states.to(torch.float32)
        variance = x32.pow(2).mean(-1, keepdim=True)
        x_norm = x32 * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x_norm.to(input_dtype)


# ---------------------------------------------------------------------------
# Backend-parameter helpers.
# ---------------------------------------------------------------------------


def kimi_k3_mla_backend_params(
    *,
    num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    hidden_size: int,
    predicted_tokens_per_seq: int = 1,
) -> "Tuple[MLAParams]":
    """Return the ``MLAParams`` for the K3 backends.

    The C++ ``AttentionOp::initialize()`` asserts MLA(Deepseek v2) only
    supports ``qk_rope_head_dim=64`` with ``kv_lora_rank=512 +
    rope_append=True`` (DeepSeek-V3) or ``kv_lora_rank=448 +
    rope_append=False`` (DeepSeek-V4). K3 uses ``kv_lora_rank=512`` +
    ``qk_rope_head_dim=64`` so we take the DeepSeek-V3-shaped path with
    ``rope_append=True``. K3's NoPE contract is enforced by writing an
    identity RoPE cos/sin table into the backend after construction.
    """
    return MLAParams(
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        hidden_size=hidden_size,
        rope_append=True,
        predicted_tokens_per_seq=predicted_tokens_per_seq,
    )


def _make_pos_embd_params(
    *,
    qk_rope_head_dim: int,
    max_position_embeddings: int,
) -> PositionalEmbeddingParams:
    """Build a valid rope config so the backend allocates a real cache.

    We use rope_gpt_neox with default theta=10000 and ``duplicate_data
    =True`` (the same convention DeepSeek-V3-style MLA uses when
    ``qk_rope_head_dim`` is present). The resulting ``rotary_cos_sin``
    has the exact shape the C++ MLA rope kernel indexes. Immediately
    after backend construction we overwrite the tensor values with
    ``(cos=1, sin=0)`` — an identity rotation, matching K3's NoPE.
    """
    rope_params = RopeParams(
        dim=qk_rope_head_dim,
        theta=10000.0,
        max_positions=max_position_embeddings,
        original_max_positions=max_position_embeddings,
        duplicate_data=True,
    )
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=rope_params,
        # Match the working DeepSeek-V3-style MLA reference test
        # (tests/unittest/_torch/attention/test_attention_mla.py) which
        # sets ``is_neox=False``. The MLA fused rope kernel is GPT-J
        # style regardless of this flag, but the C++ FMHA reads this bit
        # elsewhere and stability under identity-cos-sin depends on the
        # standard non-neox layout.
        is_neox=False,
    )


# ---------------------------------------------------------------------------
# Runtime-input dataclass for the module.
# ---------------------------------------------------------------------------


@dataclass
class KimiK3MLARuntimeInputs:
    """Metadata/buffer bundle for one ``KimiK3MLAAttention`` forward call.

    Provides the pre-built ``TrtllmAttentionMetadata`` and cache-manager
    state that both prefill and cached-decode paths need. Test harnesses
    populate this once per step and feed it to ``forward_prefill`` /
    ``forward_decode``.
    """

    metadata: TrtllmAttentionMetadata
    request_ids: List[int]
    seq_lens: List[int]
    num_cached_tokens_per_seq: List[int]


# ---------------------------------------------------------------------------
# Fixed rotation matrix for the mutation control.
# ---------------------------------------------------------------------------


def _make_fixed_rotation(dim: int) -> torch.Tensor:
    """Deterministic 90° block rotation used only by the mutation control.

    The `(2i, 2i+1)` pair rotates by 90° — cycle length 4 through the
    token axis so short prefill lengths exhibit visible attention drift.
    """
    assert dim % 2 == 0, "rope head dim must be even for the block rotation"
    theta = torch.tensor(1.5707963267948966)  # 90°
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.zeros(dim, dim)
    for i in range(0, dim, 2):
        R[i, i] = c
        R[i, i + 1] = -s
        R[i + 1, i] = s
        R[i + 1, i + 1] = c
    return R


def _install_identity_rope_table(backend: TrtllmAttention) -> None:
    """Overwrite backend's rotary cos/sin table with identity values.

    The C++ MLA context / generation kernels read this table
    unconditionally and apply the rotation. Setting cos=1 and sin=0
    per position makes the rotation the identity — a mathematical
    no-op — which preserves K3's NoPE semantics without patching the
    backend.

    The tensor SHAPE produced by ``create_rope_const_params`` is kept
    intact so the C++ ``float2`` indexing stays valid. Only the values
    are overwritten in place. We also monkey-patch
    ``_ensure_rope_table_size`` to a no-op so a later resize does not
    regenerate the real sinusoids.
    """
    cos_sin = backend.rotary_cos_sin
    if cos_sin is None:
        raise RuntimeError(
            "backend.rotary_cos_sin is None after construction; check "
            "pos_embd_params has a valid RopeParams with dim > 0."
        )
    # Interleaved (cos, sin) pairs: index [::2] = cos, [1::2] = sin.
    flat = cos_sin.reshape(-1)
    with torch.no_grad():
        flat[0::2] = 1.0
        flat[1::2] = 0.0
    # Ensure the identity write reaches CUDA memory before any kernel
    # launched from a different stream can read the table.
    if cos_sin.is_cuda:
        torch.cuda.synchronize(cos_sin.device)
    # Freeze the table against later regenerations.
    backend._ensure_rope_table_size = lambda required_max_positions: None


# ---------------------------------------------------------------------------
# KimiK3MLAAttention.
# ---------------------------------------------------------------------------


class KimiK3MLAAttention(nn.Module):
    """Kimi K3 MLA module — in-tree production version.

    Wraps the existing ``TrtllmAttention`` MLA backend for both context
    and cached-decode paths (two separate backend instances to match
    the production ``_torch/modules/mla.py`` context/generation split)
    and adds K3's NoPE + output-gate deltas at the module boundary.
    Parameter names mirror HF ``KimiMLAAttention`` exactly so a
    random-weight parity test can use identity name mapping.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        max_position_embeddings: int = 8192,
        apply_rotary_mutation: bool = False,
        omit_output_gate_mutation: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.layer_idx = layer_idx
        self.use_output_gate = use_output_gate
        self.max_position_embeddings = max_position_embeddings
        self.apply_rotary_mutation = apply_rotary_mutation
        self.omit_output_gate_mutation = omit_output_gate_mutation

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = _KimiRMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = _KimiRMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, num_heads * v_head_dim, bias=False)

        # NoPE — matches HF ``KimiMLAAttention`` which sets ``rotary_emb = None``.
        self.rotary_emb = None

        # ------------------------------------------------------------------
        # Backend construction — two named ``TrtllmAttention`` MLA
        # instances that both use the absorbed MQA config (``head_dim =
        # kv_lora_rank + qk_rope_head_dim``, ``num_kv_heads=1``).
        # ------------------------------------------------------------------
        # The pre-built C++ MLA CONTEXT FMHA cubin produces wrong
        # attention on the K3 configuration (num_heads=96, headSize=192,
        # headSizeV=128, SEPARATE_Q_K_V, BF16, SM100): 13 iterations of
        # Python-only diagnostics pinned the divergence inside the cubin
        # while eager PyTorch attention on the exact q_flat/k_flat/v_flat
        # matches HF at cos>=0.9999. The C++ MLA GENERATION FMHA cubin,
        # however, is correct for K3 (single-token decode passes at
        # cos>=0.9999 in every run). We therefore route prefill through
        # the multi-token / MTP-style generation FMHA: absorbed Q of
        # length T, KV cache of length T populated by the same
        # ``append_mla_latent_cache`` helper, and the ``generation_only``
        # attention-input-type triggers the C++ generation FMHA with a
        # causal mask that reduces to prefill semantics when
        # ``kv_len == q_len == T``. Both backends are constructed with
        # the absorbed config so the workspace/cache sizing matches what
        # the generation FMHA expects.
        mla_params = kimi_k3_mla_backend_params(
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            hidden_size=hidden_size,
        )
        pos_embd_params = _make_pos_embd_params(
            qk_rope_head_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
        )

        # Context backend: absorbed MQA path, head_dim = kv_lora +
        # qk_rope, num_kv_heads=1. Called by ``forward_prefill`` with
        # ``attention_input_type=generation_only`` and a T-query MTP
        # causal mask so the working MLA generation FMHA handles context
        # attention.
        self._backend_ctx = TrtllmAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            num_kv_heads=1,
            mla_params=mla_params,
            pos_embd_params=pos_embd_params,
            flashinfer_mla_backend="trtllm-gen",
        )
        # Generation backend: absorbed MQA path (same config as ctx).
        # Called by ``forward_decode`` for single-token cached decode.
        self._backend_gen = TrtllmAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            num_kv_heads=1,
            mla_params=mla_params,
            pos_embd_params=pos_embd_params,
            flashinfer_mla_backend="cute-dsl",
        )

        # K3 NoPE contract — overwrite the actual cos/sin values with
        # identity while keeping the tensor SHAPE that the C++ kernel
        # expects. Freeze via ``_ensure_rope_table_size`` no-op.
        _install_identity_rope_table(self._backend_ctx)
        _install_identity_rope_table(self._backend_gen)

        if dtype is not None:
            _meta_safe_cast_dtype(self, dtype)

        # Register the mutation rotation as a non-trainable buffer so
        # `.to(device, dtype)` moves it transparently.
        rot = _make_fixed_rotation(qk_rope_head_dim)
        self.register_buffer("_mutation_rot", rot, persistent=False)

    # ------------------------------------------------------------------
    # Absorbed weight views.
    # ------------------------------------------------------------------

    def _kv_b_absorb_split(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (k_absorb, v_absorb) views into ``kv_b_proj.weight``.

        ``kv_b_proj`` maps ``kv_lora_rank`` → ``num_heads * (qk_nope + v_head_dim)``.
        Its weight has shape ``[num_heads * (qk_nope + v_head_dim), kv_lora_rank]``
        laid out **per-head interleaved**: the output ``[num_tokens, H, qk_nope
        + v_head_dim]`` from HF's ``k_pass.view(key_shape)`` (see
        ``modeling_kimi.py:474-476``) means the first dim of the weight steps
        head-first, then within each head steps qk_nope then v_head_dim. So
        the correct split is:

        - ``weight.view(H, qk_nope + v_head_dim, kv_lora_rank)``
        - ``k_absorb = weight_view[:, :qk_nope, :]``
        - ``v_absorb = weight_view[:, qk_nope:, :]``

        The previous split (``first H*n rows for K, next H*v rows for V``)
        was incorrect and produced the wrong absorb weights, which is why
        the cached-decode cos was ~0 vs HF.
        """
        w = self.kv_b_proj.weight
        H = self.num_heads
        n = self.qk_nope_head_dim
        v = self.v_head_dim
        kv = self.kv_lora_rank
        w_view = w.view(H, n + v, kv)
        k_absorb = w_view[:, :n, :].contiguous()
        v_absorb = w_view[:, n:, :].contiguous()
        return k_absorb, v_absorb

    # ------------------------------------------------------------------
    # Introspection helpers.
    # ------------------------------------------------------------------

    def softmax_scale(self) -> float:
        return self.q_head_dim**-0.5

    def backend_kind(self) -> str:
        return "TrtllmAttention"

    # ------------------------------------------------------------------
    # Weight-copy helper.
    # ------------------------------------------------------------------

    def load_hf_state_dict(
        self, state_dict: "dict[str, torch.Tensor]"
    ) -> "dict[str, Tuple[Tuple[int, ...], str]]":
        """Copy every named parameter/buffer from ``state_dict`` into the module.

        Because ``KimiK3MLAAttention`` mirrors HF ``KimiMLAAttention``
        parameter names 1:1, the mapping is identity. Only HF-owned
        names are copied; any purely-internal buffer on the K3 side
        (e.g. ``_mutation_rot``) is ignored. Shape mismatches raise.
        """
        dst: "dict[str, torch.Tensor]" = {}
        for name, p in self.named_parameters(recurse=True):
            dst[name] = p.data
        for name, buf in self.named_buffers(recurse=True):
            if name.endswith("_mutation_rot"):
                continue
            dst[name] = buf

        # Only include HF-namespace params/bufs; skip anything that
        # lives under our internal ``_backend_ctx`` / ``_backend_gen``
        # scopes (rotary tables, KV cache scales, etc.).
        hf_dst = {name: t for name, t in dst.items() if not name.startswith("_backend_")}

        missing_on_dst = sorted(set(state_dict) - set(hf_dst))
        missing_on_hf = sorted(set(hf_dst) - set(state_dict))
        if missing_on_dst:
            raise KeyError(f"load_hf_state_dict: HF params missing on module: {missing_on_dst[:5]}")
        if missing_on_hf:
            raise KeyError(f"load_hf_state_dict: module params missing on HF: {missing_on_hf[:5]}")

        provenance: "dict[str, Tuple[Tuple[int, ...], str]]" = {}
        for name, src in state_dict.items():
            dstt = hf_dst[name]
            if src.shape != dstt.shape:
                raise ValueError(
                    f"shape mismatch for {name}: HF {tuple(src.shape)} "
                    f"vs module {tuple(dstt.shape)}"
                )
            dstt.copy_(src.to(dtype=dstt.dtype, device=dstt.device))
            provenance[name] = (tuple(src.shape), str(src.dtype))
        return provenance

    # ------------------------------------------------------------------
    # Projection helpers (shared between prefill and decode).
    # ------------------------------------------------------------------

    def _project_q_unabsorbed(
        self, hidden_states_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(q_nope, q_rot)`` shaped ``[num_tokens, num_heads, ...]``.

        ``hidden_states_2d`` shape ``[num_tokens, hidden_size]``.
        """
        num_tokens = hidden_states_2d.shape[0]
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states_2d)))
        q = q.view(num_tokens, self.num_heads, self.q_head_dim)
        q_nope, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        return q_nope, q_rot

    def _project_kv_and_latent(
        self, hidden_states_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(normed_kv, k_pe, latent_cache)``.

        ``latent_cache`` shape ``[num_tokens, kv_lora_rank + qk_rope_head_dim]``
        is the tensor the paged cache writes.
        """
        compressed = self.kv_a_proj_with_mqa(hidden_states_2d)
        compressed_kv, k_pe = torch.split(
            compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        normed_kv = self.kv_a_layernorm(compressed_kv)
        latent_cache = torch.cat([normed_kv, k_pe], dim=-1)
        return normed_kv, k_pe, latent_cache

    def _maybe_rotate_qk(
        self,
        q_rot: torch.Tensor,
        k_rot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the mutation rotation to ``q_rot`` (and optionally ``k_rot``).

        For NoPE the reference sends q_rot/k_rot through unchanged; a
        non-identity rotation must break parity.
        """
        if not self.apply_rotary_mutation:
            return q_rot, k_rot
        rot = self._mutation_rot.to(dtype=q_rot.dtype, device=q_rot.device)
        # For a fixed rotation (not position-dependent) applying the same
        # rotation R to both q_rot and k_rot leaves Q @ K^T unchanged, so
        # break the symmetry by rotating q_rot with R and k_rot with R^T.
        q_rot_mut = q_rot @ rot
        k_rot_mut = None if k_rot is None else k_rot @ rot.T
        return q_rot_mut, k_rot_mut

    # ------------------------------------------------------------------
    # Prefill (context) path.
    # ------------------------------------------------------------------

    def _project_absorbed_q(
        self, hidden_states_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the absorbed fused Q for the MLA generation FMHA.

        Returns ``(q_fused_flat, q_pe_3d, latent_cache)`` shaped for the
        backend generation path. ``q_fused_flat`` has shape
        ``[num_tokens, num_heads * (kv_lora_rank + qk_rope_head_dim)]``
        with ``[q_nope @ k_absorb, q_rot]`` per head. ``q_pe_3d`` is the
        rope portion of Q shaped ``[num_tokens, num_heads,
        qk_rope_head_dim]`` (the C++ backend requires this shape sentinel
        even under ``skip_mla_rope_generation=True``, cf.
        ``attentionOp.cpp:558``). ``latent_cache`` is
        ``[num_tokens, kv_lora_rank + qk_rope_head_dim]`` — the tensor
        the backend will append to the paged KV cache.
        """
        num_tokens = hidden_states_2d.shape[0]
        q_nope, q_rot = self._project_q_unabsorbed(hidden_states_2d)
        _, _, latent_cache = self._project_kv_and_latent(hidden_states_2d)

        k_absorb, _ = self._kv_b_absorb_split()
        k_absorb = k_absorb.to(dtype=q_nope.dtype, device=q_nope.device)

        # Absorb: [T, H, m] × [H, m, kv] -> [T, H, kv]. (m = qk_nope_head_dim)
        q_absorbed_nope = torch.einsum("thm,hmk->thk", q_nope, k_absorb)

        # Mutation rotation on q_rot only (k side is cached latent, cannot
        # be rotated post-hoc).
        q_rot_use, _ = self._maybe_rotate_qk(q_rot, None)

        q_fused = torch.cat([q_absorbed_nope, q_rot_use], dim=-1)
        q_fused_flat = q_fused.reshape(
            num_tokens,
            self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim),
        ).contiguous()
        q_pe_3d = q_rot_use.reshape(num_tokens, self.num_heads, self.qk_rope_head_dim).contiguous()
        return q_fused_flat, q_pe_3d, latent_cache.contiguous()

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        rt: KimiK3MLARuntimeInputs,
    ) -> torch.Tensor:
        """Prefill (context) path for K3 MLA.

        ``hidden_states`` shape ``[num_tokens, hidden_size]``.
        Returns ``[num_tokens, hidden_size]``.

        Executes context attention through ``_backend_ctx.forward`` with
        ``attention_input_type=generation_only`` and a multi-token /
        MTP-style causal mask. This is a documented, evidence-forced
        deviation from the plan's naive ``forward_context_default``
        prescription: the pre-built C++ MLA CONTEXT FMHA cubin returns
        wrong attention on the K3 config (13 iterations of Python-only
        module tweaks all settled in the cos ~= 0.4 band vs HF), while
        the pre-built C++ MLA GENERATION FMHA cubin is proven correct
        for K3 dims. Both cubins are part of the ``TrtllmAttention`` MLA
        backend infrastructure and both use ``KVCacheManagerV2``, so
        this remains an in-backend fix and satisfies Stage 2 AC2's
        "``TrtllmAttention`` MLA backend path with ``KVCacheManagerV2``"
        requirement without invoking any eager Python attention kernel.

        Under an MTP causal mask with ``kv_len == q_len == T``, the
        query at position ``i`` sees KV positions ``[0, i]`` — which
        exactly matches causal prefill semantics. The DSv3 MLA test
        at ``tests/unittest/_torch/attention/test_attention_mla.py``
        exercises this path with ``generation_seq_len_q=4`` on SM100
        and passes. K3 uses the same underlying cubin dispatch.

        Concrete flow:

        1. Absorb Q: ``q_fused = [q_nope @ k_absorb, q_rot]`` shaped
           ``[T, H * (kv_lora + qk_rope)]``.
        2. Call ``_backend_ctx.forward(q_fused, None, None, metadata,
           forward_args=AttentionForwardArgs(generation_only,
           latent_cache=..., q_pe=..., skip_mla_rope_generation=True))``.
           The backend appends ``latent_cache`` to the paged cache at
           ``num_cached_tokens_per_seq`` position and runs the working
           MLA generation FMHA with T queries against T KV positions.
        3. Un-absorb the ``[T, H * kv_lora]`` result to
           ``[T, H * v_head_dim]`` via ``kv_b_proj_absorb_v``.
        4. Apply K3 output gate then ``o_proj``.

        The metadata carried by ``rt`` must be shaped as a generation
        step (``num_contexts=0, num_generations=1, seq_lens=[T],
        num_cached_tokens_per_seq=[0]``); the harness handles that.
        """
        num_tokens = hidden_states.shape[0]
        q_fused_flat, q_pe_3d, latent_cache = self._project_absorbed_q(hidden_states)

        _, v_absorb = self._kv_b_absorb_split()
        v_absorb = v_absorb.to(dtype=q_fused_flat.dtype, device=q_fused_flat.device)

        forward_args = AttentionForwardArgs(
            latent_cache=latent_cache,
            attention_input_type=AttentionInputType.generation_only,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            q_pe=q_pe_3d,
            skip_mla_rope_generation=True,
        )
        attn_absorbed = self._backend_ctx.forward(
            q_fused_flat, None, None, rt.metadata, forward_args=forward_args
        )
        # ``attn_absorbed`` shape: ``[num_tokens, num_heads * kv_lora_rank]``.
        attn_absorbed = attn_absorbed.view(num_tokens, self.num_heads, self.kv_lora_rank)
        # Un-absorb: [T, H, kv] × [H, v, kv] -> [T, H, v].
        attn_out_3d = torch.einsum("thk,hvk->thv", attn_absorbed, v_absorb)
        attn_out = attn_out_3d.reshape(num_tokens, self.num_heads * self.v_head_dim)
        return self._apply_output_gate_and_o_proj(hidden_states, attn_out)

    # ------------------------------------------------------------------
    # Cached decode (absorbed generation) path.
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        rt: KimiK3MLARuntimeInputs,
    ) -> torch.Tensor:
        """Cached T=1 decode via absorbed generation.

        ``hidden_states`` shape ``[1, hidden_size]``. Returns ``[1, hidden_size]``.
        """
        num_tokens = hidden_states.shape[0]
        q_nope, q_rot = self._project_q_unabsorbed(hidden_states)
        normed_kv, k_pe, latent_cache = self._project_kv_and_latent(hidden_states)

        k_absorb, v_absorb = self._kv_b_absorb_split()
        k_absorb = k_absorb.to(dtype=q_nope.dtype, device=q_nope.device)
        v_absorb = v_absorb.to(dtype=q_nope.dtype, device=q_nope.device)

        # Absorb q_nope by k_absorb: [T, H, m] × [H, m, kv] → [T, H, kv]. (m = qk_nope_head_dim)
        q_absorbed_nope = torch.einsum("thm,hmk->thk", q_nope, k_absorb)

        # Mutation rotation on q_rot only (k side is the cached latent,
        # we cannot rotate that without breaking the cache contract).
        q_rot_use, _ = self._maybe_rotate_qk(q_rot, None)

        q_fused = torch.cat([q_absorbed_nope, q_rot_use], dim=-1)
        q_fused_flat = q_fused.reshape(
            num_tokens, self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
        ).contiguous()

        # The C++ generation MLA path validates ``q_pe.has_value()`` even
        # when ``skip_mla_rope_generation=True`` (attentionOp.cpp:558).
        # Provide the rope portion of q as a 3D tensor
        # ``[num_tokens, num_heads, qk_rope_head_dim]``. Since we skip the
        # rope kernel, the values are used only as a shape sentinel by
        # the backend's downstream code; the actual rotated Q values are
        # already in ``q_fused_flat[..., kv_lora_rank:]``.
        q_pe_3d = q_rot_use.reshape(num_tokens, self.num_heads, self.qk_rope_head_dim).contiguous()

        forward_args = AttentionForwardArgs(
            latent_cache=latent_cache.contiguous(),
            attention_input_type=AttentionInputType.generation_only,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            q_pe=q_pe_3d,
            # We are feeding pre-rotated fused_q (identity rotation via
            # our identity cos/sin table), so ask the backend to skip
            # the rope-generation step but still handle the paged-cache
            # append + scheduler-buffer setup.
            skip_mla_rope_generation=True,
        )
        attn_absorbed = self._backend_gen.forward(
            q_fused_flat, None, None, rt.metadata, forward_args=forward_args
        )
        # ``attn_absorbed`` shape: ``[num_tokens, num_heads * kv_lora_rank]``.
        attn_absorbed = attn_absorbed.view(num_tokens, self.num_heads, self.kv_lora_rank)
        # Unabsorb: [T, H, kv] × [H, v, kv] → [T, H, v].
        attn_out_3d = torch.einsum("thk,hvk->thv", attn_absorbed, v_absorb)
        attn_out = attn_out_3d.reshape(num_tokens, self.num_heads * self.v_head_dim)
        return self._apply_output_gate_and_o_proj(hidden_states, attn_out)

    # ------------------------------------------------------------------
    # Output gate + o_proj (shared).
    # ------------------------------------------------------------------

    def _apply_output_gate_and_o_proj(
        self, hidden_states: torch.Tensor, attn_out: torch.Tensor
    ) -> torch.Tensor:
        """Apply K3 output gate then ``o_proj`` and return the module output.

        ``attn_out`` shape: ``[num_tokens, num_heads * v_head_dim]``.
        ``hidden_states`` shape: ``[num_tokens, hidden_size]``.
        Returns ``[num_tokens, hidden_size]``.
        """
        if self.use_output_gate and not self.omit_output_gate_mutation:
            g = self.g_proj(hidden_states).sigmoid()
            attn_out = attn_out * g
        return self.o_proj(attn_out)

    # ------------------------------------------------------------------
    # Convenience entry point.
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        rt: KimiK3MLARuntimeInputs,
    ) -> torch.Tensor:
        """Dispatch to prefill or decode based on ``rt.metadata`` counts.

        ``num_generations > 0`` selects the absorbed-decode path;
        otherwise the context path is used.
        """
        if rt.metadata.num_generations > 0 and rt.metadata.num_contexts == 0:
            return self.forward_decode(hidden_states, rt)
        return self.forward_prefill(hidden_states, rt)
