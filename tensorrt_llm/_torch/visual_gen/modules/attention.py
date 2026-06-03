from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn

from tensorrt_llm.llmapi.llm_args import SkipSoftmaxAttentionConfig

from ...modules.linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from ..attention_backend.interface import AttentionTensorLayout
from ..attention_backend.parallel import wrap_parallel_attention
from ..attention_backend.utils import create_attention
from ..config import DiffusionModelConfig, SkipSoftmaxConfig
from ..modules.rms_norm import RMSNormTPAware


class QKVMode(str, Enum):
    FUSE_QKV = "fuse_qkv"
    FUSE_KV = "fuse_kv"
    SEPARATE_QKV = "separate"


# TODO: torch compile
def apply_rotary_emb(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    freqs_cos = freqs_cos.to(x.dtype)
    freqs_sin = freqs_sin.to(x.dtype)
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)  # [B, S, H, D/2]

    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]

    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    """Attention module for visual generation models."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        qkv_mode: QKVMode = QKVMode.FUSE_QKV,
        qk_norm: bool = True,
        qk_norm_mode: str = "full",
        eps: float = 1e-6,
        bias: bool = True,
        interleave: bool = True,
        fuse_qk_norm_rope: Optional[bool] = None,
        config: Optional[DiffusionModelConfig] = None,
        layer_idx: Optional[int] = None,
        enable_sequence_parallel: bool = True,
    ):
        super().__init__()

        config = config or DiffusionModelConfig()
        self.dtype = config.torch_dtype
        self.quant_config = config.quant_config
        self.skip_create_weights_in_init = config.skip_create_weights_in_init
        self.force_dynamic_quantization = config.force_dynamic_quantization
        self.mapping = config.mapping
        self.allreduce_strategy = config.allreduce_strategy

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.qkv_mode = QKVMode(qkv_mode) if isinstance(qkv_mode, str) else qkv_mode
        self.bias = bias

        self.tp_size = self.mapping.tp_size if self.mapping else 1
        assert (
            self.num_attention_heads % self.tp_size == 0
            and self.num_key_value_heads % self.tp_size == 0
        ), "TP size must divide the number of Query and KV Heads"

        # Fused QK Norm + RoPE: each model class opts in via fuse_qk_norm_rope.
        # Backed by torch.ops.trtllm.fused_dit_qk_norm_rope which auto-dispatches:
        #   - per-head template (FLUX/Cosmos):   q/k_weight.shape == [head_dim]
        #   - full-dim template (LTX-2, WAN):    q/k_weight.shape == [num_heads * head_dim]
        # Full-dim template envelope: num_heads <= 64, head_dim in {64, 128}.
        self.fuse_qk_norm_rope = fuse_qk_norm_rope if fuse_qk_norm_rope is not None else False
        assert not (self.fuse_qk_norm_rope and self.tp_size > 1 and qk_norm_mode == "full"), (
            "fuse_qk_norm_rope + qk_norm_mode='full' + TP>1: fused kernel lacks cross-rank "
            "all-reduce for cross-head RMSNorm variance. Disable fuse_qk_norm_rope for TP>1."
        )
        self.interleave = interleave

        # Select compute backend (orthogonal to parallelism)
        vgm = config.visual_gen_mapping
        ulysses_size = vgm.ulysses_size if vgm else 1
        attn2d_size = (vgm.attn2d_row_size * vgm.attn2d_col_size) if vgm else 1
        base_backend = config.attention.backend
        _sa_cfg = config.attention.sparse_attention_config
        _is_vsa = (
            base_backend == "CUTEDSL"
            and _sa_cfg is not None
            and getattr(_sa_cfg, "algorithm", None) == "vsa"
        )

        # Cross-attention fallback: TRTLLM and CUTEDSL VSA are self-attn only.
        if self.qkv_mode == QKVMode.SEPARATE_QKV and (base_backend == "TRTLLM" or _is_vsa):
            backend_name = "VANILLA"
        else:
            backend_name = base_backend

        if _is_vsa and attn2d_size > 1:
            raise ValueError(
                f"VSA needs the full token sequence per rank, so it is incompatible "
                f"with Attention2D (attn2d_size={attn2d_size}). Use ulysses or cfg "
                f"parallelism instead."
            )
        self.attn_backend = backend_name
        self.qk_norm = qk_norm
        self.qk_norm_mode = qk_norm_mode
        self.layer_idx = layer_idx if layer_idx is not None else 0
        self.eps = eps

        self.q_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim

        self.local_num_attention_heads = self.num_attention_heads // self.tp_size
        self.local_num_key_value_heads = self.num_key_value_heads // self.tp_size
        self.local_q_dim = self.local_num_attention_heads * self.head_dim
        self.local_kv_dim = self.local_num_key_value_heads * self.head_dim

        self._init_qkv_proj()

        attention_metadata_state = getattr(config, "attention_metadata_state", None)

        if self.qk_norm:
            # "full": norm over all heads combined (e.g. WAN, dim=q_dim)
            # "per_head": norm over each head independently (e.g. FLUX, dim=head_dim)

            q_norm_dim = self.head_dim if qk_norm_mode == "per_head" else self.q_dim
            k_norm_dim = self.head_dim if qk_norm_mode == "per_head" else self.kv_dim
            enable_tp_rms = self.tp_size > 1 and qk_norm_mode == "full"
            self.norm_q = RMSNormTPAware(
                hidden_size=q_norm_dim,
                eps=self.eps,
                dtype=self.dtype,
                has_weights=True,
                enable_tp=enable_tp_rms,
                mapping=self.mapping,
            )
            self.norm_k = RMSNormTPAware(
                hidden_size=k_norm_dim,
                eps=self.eps,
                dtype=self.dtype,
                has_weights=True,
                enable_tp=enable_tp_rms,
                mapping=self.mapping,
            )

        # TODO: Use weight mapper to create just a Linear module
        self.to_out = nn.ModuleList(
            [
                Linear(
                    self.q_dim,
                    self.hidden_size,
                    bias=self.bias,
                    dtype=self.dtype,
                    mapping=self.mapping,
                    quant_config=self.quant_config,
                    skip_create_weights_in_init=self.skip_create_weights_in_init,
                    force_dynamic_quantization=self.force_dynamic_quantization,
                    tensor_parallel_mode=TensorParallelMode.ROW if self.tp_size > 1 else None,
                    reduce_output=(self.tp_size > 1),
                    allreduce_strategy=self.allreduce_strategy,
                )
            ]
        )

        # Ulysses shards heads across workers; inner backend sees sharded head count.
        # Attention2D gathers sequence (not heads); see wrap_parallel_attention for nesting.
        use_ulysses = ulysses_size > 1 and enable_sequence_parallel
        if use_ulysses:
            backend_num_heads = self.local_num_attention_heads // ulysses_size
            backend_num_kv_heads = self.local_num_key_value_heads // ulysses_size
        else:
            backend_num_heads = self.local_num_attention_heads
            backend_num_kv_heads = self.local_num_key_value_heads

        # Resolve sparse attention config for TRTLLM backend
        sparse_attention_config = None
        ss_cfg = config.attention.sparse_attention_config
        if isinstance(ss_cfg, SkipSoftmaxConfig) and backend_name == "TRTLLM":
            # Cache the resolved scalar on a private attr (idempotent across
            # all Attention modules); does NOT mutate the source-of-truth
            # `threshold_scale_factor` / `target_sparsity` fields. Subsequent
            # callers — including `apply_skip_softmax_overrides` — read the
            # cached value via `resolve_threshold(module_name)`.
            threshold = ss_cfg.get_or_resolve_threshold()

            if threshold is not None and threshold > 0:
                sparse_attention_config = SkipSoftmaxAttentionConfig(
                    threshold_scale_factor={"prefill": threshold, "decode": 0}
                )

        # Create compute backend
        self.attn = create_attention(
            backend=backend_name,
            layer_idx=self.layer_idx,
            num_heads=backend_num_heads,
            head_dim=self.head_dim,
            num_kv_heads=backend_num_kv_heads,
            quant_config=self.quant_config,
            dtype=self.dtype,
            attention_config=config.attention,
            attention_metadata_state=attention_metadata_state,
            sparse_attention_config=sparse_attention_config,
        )

        if enable_sequence_parallel and self.qkv_mode == QKVMode.SEPARATE_QKV and vgm is not None:
            ring_size = vgm.ring_size
            attn2d_size = vgm.attn2d_row_size * vgm.attn2d_col_size
            if ring_size > 1 or attn2d_size > 1:
                raise ValueError(
                    "SEPARATE_QKV cross-attention does not support Ring or Attention2D "
                    "sequence parallelism; use enable_sequence_parallel=False or Ulysses-only "
                    f"(ring_size={ring_size}, attn2d_size={attn2d_size})."
                )

        self.attn = wrap_parallel_attention(
            self.attn,
            visual_gen_mapping=vgm,
            enable_sequence_parallel=enable_sequence_parallel,
        )

    def _init_qkv_proj(self) -> None:
        tp_mode = TensorParallelMode.COLUMN if self.tp_size > 1 else None

        if self.qkv_mode == QKVMode.FUSE_QKV:
            qkv_out_dim = self.q_dim + 2 * self.kv_dim

            # Input / Output dims are the full tensor sizes
            # fused_weight_shard_indices_mapping want indexes for just _this_ shard
            self.qkv_proj = Linear(
                self.hidden_size,
                qkv_out_dim,
                bias=self.bias,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_QKV_LINEAR
                ),
                fused_weight_shard_indices_mapping={
                    "q": (0, self.local_q_dim),
                    "k": (self.local_q_dim, self.local_kv_dim),
                    "v": (self.local_q_dim + self.local_kv_dim, self.local_kv_dim),
                },
                tensor_parallel_mode=tp_mode,
                reduce_output=False,
            )
        else:
            self.to_q = Linear(
                self.hidden_size,
                self.q_dim,
                bias=self.bias,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
                tensor_parallel_mode=tp_mode,
                reduce_output=False,
            )
            self.to_k = Linear(
                self.hidden_size,
                self.kv_dim,
                bias=self.bias,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
                tensor_parallel_mode=tp_mode,
                reduce_output=False,
            )
            self.to_v = Linear(
                self.hidden_size,
                self.kv_dim,
                bias=self.bias,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
                tensor_parallel_mode=tp_mode,
                reduce_output=False,
            )

    def get_qkv(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.qkv_mode == QKVMode.FUSE_QKV:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.local_q_dim, self.local_kv_dim, self.local_kv_dim], dim=-1)
        else:
            kv_source = (
                encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            )
            q = self.to_q(hidden_states)
            k = self.to_k(kv_source)
            v = self.to_v(kv_source)
        return q, k, v

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
        return q, k

    def apply_packed_qk_norm_rope(
        self,
        qkv: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        num_txt_tokens: int = -1,
        q_add_weight: Optional[torch.Tensor] = None,
        k_add_weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Apply fused QK Norm + RoPE in-place on packed QKV tensor (FUSE_QKV self-attn).

        The op auto-dispatches by tensor shapes/dtypes:
          - q_weight.shape = [head_dim]              → per-head norm (FLUX/Cosmos, w/ dual-stream)
          - q_weight.shape = [num_heads * head_dim]  → full-dim norm (LTX-2, WAN, ≤64 heads)
          - cos last dim = head_dim                  → per-token shared cos (FLUX, WAN)
          - cos last dim = num_heads * head_dim      → per-token per-head cos (LTX-2 3D RoPE)
          - cos rows < num_tokens                    → kernel broadcasts via cos_seq_per_batch

        Caller passes raw freqs_cos / freqs_sin (any rank); the op reshapes internally.
        """
        B, S, D = qkv.shape
        tokens_per_batch = S if num_txt_tokens > 0 else 0
        assert self.tp_size == 1, "fused_dit_split_norm_rope does not support TP"
        torch.ops.trtllm.fused_dit_qk_norm_rope(
            qkv.view(B * S, D),
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.eps,
            self.norm_q.weight,
            self.norm_k.weight,
            q_add_weight,
            k_add_weight,
            freqs_cos,
            freqs_sin,
            num_txt_tokens,
            self.interleave,
            tokens_per_batch,
        )

    def apply_split_norm_rope(
        self,
        tensor: torch.Tensor,
        weight: torch.Tensor,
        num_heads: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> None:
        """In-place fused RMSNorm + RoPE on a single Q or K tensor [B, T, H*D] (SEPARATE_QKV cross-attn).

        The op auto-dispatches by shapes/dtypes (same as `apply_packed_qk_norm_rope`):
          - cos last dim = head_dim                  → per-token shared cos
          - cos last dim = num_heads * head_dim      → per-token per-head cos (LTX-2 3D RoPE)
          - cos rows < num_tokens                    → kernel broadcasts via cos_seq_per_batch
          - cos dtype bf16 or fp32                   → kernel upcasts bf16 to fp32 in registers
        Caller passes raw cos/sin (any rank); the op reshapes internally.
        """
        B, T, _ = tensor.shape
        assert self.tp_size == 1, "fused_dit_split_norm_rope does not support TP"
        torch.ops.trtllm.fused_dit_split_norm_rope(
            tensor.view(B * T, -1),
            num_heads,
            self.head_dim,
            self.eps,
            weight,
            cos,
            sin,
            self.interleave,
        )

    def apply_split_norm(
        self,
        tensor: torch.Tensor,
        weight: torch.Tensor,
        num_heads: int,
    ) -> None:
        """In-place fused full-dim RMSNorm only (no RoPE) on a single Q or K tensor [B, T, H*D].

        Calls trtllm.fused_dit_split_norm. Used by paths that need norm but
        no RoPE -- e.g. LTX-2 text cross-attn (Q-norm with pe=None).
        """
        B, T, _ = tensor.shape
        assert self.tp_size == 1, "fused_dit_split_norm does not support TP"
        torch.ops.trtllm.fused_dit_split_norm(
            tensor.view(B * T, -1),
            num_heads,
            self.head_dim,
            self.eps,
            weight,
        )

    def apply_split_norm_or_norm_rope(
        self,
        tensor: torch.Tensor,
        weight: torch.Tensor,
        num_heads: int,
        pe: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        """Dispatcher: in-place norm-only when pe is None, else norm + RoPE.

        Used to dispatch all SEPARATE_QKV cross-attn norm paths through the
        split-fuse kernels regardless of whether the path needs RoPE.
        """
        if pe is None:
            self.apply_split_norm(tensor, weight, num_heads)
        else:
            cos, sin = pe
            self.apply_split_norm_rope(tensor, weight, num_heads, cos, sin)

    def _attn_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Call attention backend with appropriate tensor layout.

        Dimensions are derived from tensor shapes. Extra **kwargs are
        forwarded to the backend. Backend-specific tensors that share
        Q/K/V's [B, S, H*D] layout (e.g. VSA's gate_compress /
        gate_fine) are reshaped here to the backend's 4-D layout.

        Two layout paths:
        1. HND backends (VANILLA): [B, S, H*D] -> [B, H, S, D]
        2. NHD backends (TRTLLM, UlyssesAttention, Attention2DAttention): [B, S, H*D] -> [B, S, H, D]
        """
        backend_layout = getattr(self.attn, "preferred_layout", AttentionTensorLayout.NHD)

        batch_size = q.shape[0]
        seq_len = q.shape[1]
        seq_len_kv = k.shape[1] if k is not None else seq_len

        def _reshape_gate(gate: torch.Tensor) -> torch.Tensor:
            gate = gate.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)
            if backend_layout == AttentionTensorLayout.HND:
                gate = gate.transpose(1, 2)
            return gate

        # Reshape inputs: [B, S, H*D] -> backend's preferred 4D layout
        if backend_layout == AttentionTensorLayout.HND:
            q = q.view(batch_size, -1, self.local_num_attention_heads, self.head_dim).transpose(
                1, 2
            )
            k = k.view(batch_size, -1, self.local_num_key_value_heads, self.head_dim).transpose(
                1, 2
            )
            v = v.view(batch_size, -1, self.local_num_key_value_heads, self.head_dim).transpose(
                1, 2
            )
        else:
            q = q.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)
            k = k.view(batch_size, -1, self.local_num_key_value_heads, self.head_dim)
            v = v.view(batch_size, -1, self.local_num_key_value_heads, self.head_dim)

        kwargs.update(
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "seq_len_kv": seq_len_kv,
            }
        )
        for gate_key in ("gate_compress", "gate_fine"):
            if kwargs.get(gate_key) is not None:
                kwargs[gate_key] = _reshape_gate(kwargs[gate_key])

        out = self.attn.forward(q=q, k=k, v=v, **kwargs)

        # Flatten back to [B, S, H*D]
        if backend_layout == AttentionTensorLayout.HND:
            return out.transpose(1, 2).flatten(2)
        else:
            return out.flatten(2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert hidden_states.ndim == 3, "hidden_states must be a 3D tensor"
        batch_size, seq_len = hidden_states.shape[:2]
        kv_seq_len = (
            encoder_hidden_states.shape[1] if encoder_hidden_states is not None else seq_len
        )

        # Fused path: QKV projection → fused QK norm + RoPE → attention
        if (
            self.fuse_qk_norm_rope
            and freqs is not None
            and self.qkv_mode == QKVMode.FUSE_QKV
            and self.qk_norm
        ):
            qkv = self.qkv_proj(hidden_states)
            freqs_cos, freqs_sin = freqs
            self.apply_packed_qk_norm_rope(qkv, freqs_cos, freqs_sin)
            q, k, v = qkv.split([self.local_q_dim, self.local_kv_dim, self.local_kv_dim], dim=-1)
            out = self._attn_impl(q, k, v, **kwargs)
            return self.to_out[0](out)

        # Unfused path: separate QK norm → separate RoPE → attention
        q, k, v = self.get_qkv(hidden_states, encoder_hidden_states)
        q, k = self.apply_qk_norm(q, k)

        # Apply RoPE if provided (model handles RoPE, not attention backend)
        if freqs is not None:
            freqs_cos, freqs_sin = freqs
            q = q.view(
                batch_size, seq_len, self.local_num_attention_heads, self.head_dim
            )  # [B, S, H, D]
            k = k.view(batch_size, kv_seq_len, self.local_num_key_value_heads, self.head_dim)
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
            q = q.flatten(2)
            k = k.flatten(2)

        out = self._attn_impl(q, k, v, **kwargs)
        out = self.to_out[0](out)
        return out
