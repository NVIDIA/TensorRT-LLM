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
"""The MiniMax-H3 transformer of the ``minimax_h3_nvfp4__sm_107__u1_u4`` scenario.

Three subclasses of the family's modules, each replacing one forward with the scenario's path and
leaving the family's construction, weight loading and everything else untouched:

* :class:`ScenarioAttention`: the fused QK-norm + RoPE + FP8 cast producer (one grouped Q / K cast with
  V's amax folded in, static Q / K scales carrying the softmax scale), trtllm-gen's FP16-softmax dense
  FP8 FMHA through the cuDNN backend's ``fp8_dense_fmha`` option (deferred rescale threshold 1), the
  packed FP8 Ulysses exchange and the NVFP4 output quantize for the static-NVFP4 ``to_out``.
* :class:`ScenarioBlock`: the AdaLN norm + modulation and the SwiGLU producers write the Linears'
  NVFP4 inputs directly.
* :class:`ScenarioTransformer`: the family transformer built from the two classes above, with the
  cuDNN backend options of the scenario and its Linears' NVFP4 GEMMs restricted to the scenario's
  backend tuple.

Every switch comes from :mod:`.config`; the family's environment-free defaults are what an unmatched
configuration runs.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen.attention.attention import QKVMode
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm.logger import logger

from ..fused_adaln import h3_gate_res_norm_mod, h3_norm_mod
from ..transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
)
from .config import SCENARIO
from .kernels.fused_fp8_qk import fusion_mode, qk_static_descale, set_reduce_group, static_descale
from .kernels.fused_out_a2a import OUT_A2A_FP4_ENABLED, h3_out_a2a_fp4
from .kernels.fused_out_quant import OUT_QUANT_ENABLED, h3_out_quant_fp4
from .kernels.fused_out_quant import static_nvfp4_input as _static_nvfp4_input
from .kernels.fused_quant import (
    FUSED_QUANT_ENABLED,
    h3_gate_res_norm_mod_fp4,
    h3_norm_mod_fp4,
    h3_swiglu_fp4,
    static_nvfp4_input,
)


class ScenarioAttention(MiniMaxH3Attention):
    """Full non-causal attention on pre-quantized FP8 Q / K / V, NVFP4 out.

    An attention whose configuration carries no ``quant_attention_config`` (a module the checkpoint
    keeps on BF16 operands) is built without the FP8 backend options and runs the family path."""

    def __init__(
        self, *, model_config: DiffusionModelConfig, backend_kwargs=None, **kwargs
    ) -> None:
        if model_config.attention.quant_attention_config is None and backend_kwargs:
            backend_kwargs = {
                key: value for key, value in backend_kwargs.items() if not key.startswith("fp8_")
            }
        super().__init__(model_config=model_config, backend_kwargs=backend_kwargs or None, **kwargs)

    def on_fp8_path(self) -> bool:
        """Whether the backend takes pre-quantized FP8 Q / K / V, the only path that accepts the
        fused NVFP4 producer's codes."""
        backend = getattr(self.attn, "inner_backend", self.attn)
        return getattr(backend, "quant_dtype", None) == "fp8"

    def _use_fused_qk_norm_rope(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:
        """The family's check, with a pre-quantized activation (``Fp4QuantizedTensor`` from the fused
        NVFP4 producer) read through its packed codes: the projection output is BF16 either way."""
        if not self.fuse_qk_norm_rope or not self.qk_norm or self.qkv_mode != QKVMode.FUSE_QKV:
            return False
        if self.local_num_attention_heads != self.local_num_key_value_heads or self.head_dim != 128:
            return False
        for norm in (self.norm_q, self.norm_k):
            if norm.allreduce is not None or norm.use_gemma or norm.weight.dtype != torch.bfloat16:
                return False
        cos, sin = rotary_emb
        if isinstance(hidden_states, Fp4QuantizedTensor):
            probe, is_bf16 = hidden_states.fp4_tensor, True
        else:
            probe, is_bf16 = hidden_states, hidden_states.dtype == torch.bfloat16
        if (
            not probe.is_cuda
            or not is_bf16
            or probe.ndim != 3
            or cos.shape != (probe.shape[1], cos.shape[-1])
            or sin.shape != cos.shape
            or cos.shape[-1] % 32
            or cos.shape[-1] > self.head_dim
        ):
            return False
        if torch.is_grad_enabled() and any(
            t.requires_grad for t in (probe, cos, sin, self.norm_q.weight, self.norm_k.weight)
        ):
            return False
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        valid_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        if self.use_async_ulysses:
            raise NotImplementedError(
                "the scenario serves the packed FP8 Ulysses exchange, not async_ulysses"
            )
        batch_size, sequence_length = hidden_states.shape[:2]
        if rotary_emb is not None and self._use_fused_fp8_qk(hidden_states, rotary_emb):
            # One pass normalises and rotates Q and K from the packed projection output and casts
            # Q, K and V to FP8 with their per-tensor scales; the backend receives FP8 inputs and
            # the descales.
            qkv = self.qkv_proj(hidden_states)
            cos, sin = rotary_emb
            fp8_device = getattr(hidden_states, "fp4_tensor", hidden_states).device
            if fusion_mode() == "static":
                descale_in = static_descale(self.layer_idx, fp8_device)
            elif fusion_mode() == "qkstatic":
                descale_in = qk_static_descale(self.head_dim, fp8_device)
            else:
                descale_in = None
            valid_len = None
            if key_padding_mask is not None:
                # Trailing padding; the mask spans the full sequence, this rank holds rows
                # [rank * S_local, (rank + 1) * S_local). Padded rows must not set the scale.
                lengths = key_padding_mask.sum(dim=1) - self._ulysses_rank * sequence_length
                valid_len = lengths.clamp(0, sequence_length).to(torch.int32).contiguous()
            if self._use_packed_a2a(batch_size, key_padding_mask):
                # The cast writes the Ulysses send buffer directly; one all-to-all, no permutes.
                ulysses = int(self.attn.world_size)
                send, descales = torch.ops.h3_lever8.qk_norm_rope_fp8_packed(
                    qkv,
                    self.norm_q.weight,
                    self.norm_k.weight,
                    cos,
                    sin,
                    descale_in,
                    float(self.norm_q.variance_epsilon),
                    self.local_num_attention_heads,
                    self.head_dim,
                    int(self.layer_idx),
                    valid_len,
                    ulysses,
                )
                out_linear = self.to_out[0]
                fp4_exchange = (
                    OUT_A2A_FP4_ENABLED and OUT_QUANT_ENABLED and _static_nvfp4_input(out_linear)
                )
                attn_out = self.attn.forward_packed_fp8(
                    send,
                    fp8_descales=descales,
                    return_pre_a2a=fp4_exchange,
                    discard_padded_queries=True,
                    timestep=timestep,
                )
                if fp4_exchange:
                    codes, scales = h3_out_a2a_fp4(
                        attn_out, out_linear.input_scale, self.attn.process_group
                    )
                    return out_linear(Fp4QuantizedTensor(codes, scales))
                return self._out_proj(attn_out.flatten(2))
            query, key, value, descales = torch.ops.h3_lever8.qk_norm_rope_fp8(
                qkv,
                self.norm_q.weight,
                self.norm_k.weight,
                cos,
                sin,
                descale_in,
                float(self.norm_q.variance_epsilon),
                self.local_num_attention_heads,
                self.head_dim,
                int(self.layer_idx),
                valid_len,
            )
            hidden_states = self._attn_impl(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                discard_padded_queries=True,
                timestep=timestep,
                fp8_descales=descales,
            )
            return self._out_proj(hidden_states)
        # Not the FP8 recipe (BF16 Q / K / V): the family path, its to_out replaced by _out_proj.
        if isinstance(hidden_states, Fp4QuantizedTensor):
            raise RuntimeError(
                "the fused NVFP4 producer handed codes to an attention that is not on the FP8 path"
            )
        out = super().forward(
            hidden_states,
            rotary_emb,
            key_padding_mask=key_padding_mask,
            timestep=timestep,
            valid_seq_len=valid_seq_len,
        )
        return out

    def _out_proj(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``to_out`` on the attention output; with a static-NVFP4 ``to_out`` the BF16 output is
        quantized by the tiled kernel  instead of the Linear's ``fp4_quantize``."""
        out = self.to_out[0]
        if (
            OUT_QUANT_ENABLED
            and torch.is_tensor(hidden_states)
            and hidden_states.dtype == torch.bfloat16
            and _static_nvfp4_input(out)
        ):
            codes, scales = h3_out_quant_fp4(hidden_states, out.input_scale)
            return out(Fp4QuantizedTensor(codes, scales))
        return out(hidden_states)

    @property
    def _ulysses_rank(self) -> int:
        """Ulysses rank of this attention's sequence shard (0 without Ulysses); read once and cached."""
        rank = self.__dict__.get("_ulysses_rank_cache")
        if rank is None:
            rank = (
                torch.distributed.get_rank(group=self.attn.process_group)
                if getattr(self.attn, "world_size", 1) > 1
                else 0
            )
            self.__dict__["_ulysses_rank_cache"] = rank
        return rank

    def _use_packed_a2a(self, batch_size: int, key_padding_mask: Optional[torch.Tensor]) -> bool:
        """Packed FP8 Ulysses exchange: one sequence, no padding (the padded path keeps the per-tensor
        exchange and the cuDNN prefix plan), a Ulysses wrapper with ``forward_packed_fp8`` and an inner
        backend that takes explicit strides (cuDNN)."""
        if not SCENARIO.packed_a2a or batch_size != 1 or key_padding_mask is not None:
            return False
        attn = self.attn
        if getattr(attn, "world_size", 1) <= 1 or not hasattr(attn, "forward_packed_fp8"):
            return False
        inner = getattr(attn, "inner_backend", None)
        return inner is not None and getattr(inner, "quant_dtype", None) == "fp8"

    def _use_fused_fp8_qk(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]
    ) -> bool:
        """Select the fused QK-norm + RoPE + FP8 cast for the per-tensor FP8 recipe on a packed BF16
        projection with equal Q / K head counts."""
        if fusion_mode() == "off" or self.qkv_mode != QKVMode.FUSE_QKV or not self.qk_norm:
            return False
        backend = getattr(self.attn, "inner_backend", self.attn)
        if getattr(backend, "quant_dtype", None) != "fp8":
            return False
        # Ulysses: the ranks hold shards of one sequence; the dynamic amax must be reduced over
        # them so the codes that travel through the all-to-all share one scale per tensor.
        if getattr(self.attn, "world_size", 1) > 1:
            set_reduce_group(self.attn.process_group)
        if self.local_num_attention_heads != self.local_num_key_value_heads:
            return False
        for norm in (self.norm_q, self.norm_k):
            if (
                getattr(norm, "allreduce", None) is not None
                or getattr(norm, "use_gemma", False)
                or norm.weight.dtype != torch.bfloat16
                or norm.weight.numel() != self.head_dim
            ):
                return False
        cos, sin = rotary_emb
        rot = cos.shape[-1]
        # The NVFP4 producer hands an Fp4QuantizedTensor (codes + scales): read device, rank and
        # sequence length from the codes; the projection output is BF16 either way.
        codes = getattr(hidden_states, "fp4_tensor", None)
        x = codes if codes is not None else hidden_states
        return (
            x.is_cuda
            and (codes is not None or hidden_states.dtype == torch.bfloat16)
            and x.ndim == 3
            and cos.shape == (x.shape[1], rot)
            and sin.shape == cos.shape
            and rot % 2 == 0
            and rot <= self.head_dim
            and (self.head_dim & (self.head_dim - 1)) == 0
            and not torch.is_grad_enabled()
        )


class ScenarioBlock(MiniMaxH3TransformerBlock):
    """The family block with the norm and SwiGLU producers emitting the NVFP4 inputs of the Linears
    that take them; under a MIXED_PRECISION checkpoint this is decided per consumer Linear."""

    attention_cls = ScenarioAttention

    def _forward_fused_adaln(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor],
        valid_seq_len: Optional[int],
    ) -> torch.Tensor:
        fuse_qkv, fuse_gate_up, fuse_down = self._nvfp4_producers()
        if not (fuse_qkv or fuse_gate_up or fuse_down):
            return super()._forward_fused_adaln(
                hidden_states,
                temb,
                adaln_indices,
                rotary_emb,
                key_padding_mask,
                timestep,
                valid_seq_len,
            )
        hidden_size = hidden_states.shape[-1]
        mod = self.adaln_proj.table(temb)
        if fuse_qkv:
            codes, scales = h3_norm_mod_fp4(
                hidden_states,
                self.norm1.weight,
                mod,
                adaln_indices,
                self.attn.qkv_proj.input_scale,
                self.norm1.variance_epsilon,
                1,
                0,
            )
            attn_input = Fp4QuantizedTensor(codes, scales)
        else:
            attn_input = h3_norm_mod(
                hidden_states,
                self.norm1.weight,
                mod,
                adaln_indices,
                self.norm1.variance_epsilon,
                1,
                0,
            )
        attn_output = self.attn(attn_input, rotary_emb, key_padding_mask, timestep, valid_seq_len)
        if fuse_gate_up:
            hidden_states, codes, scales = h3_gate_res_norm_mod_fp4(
                hidden_states,
                attn_output,
                self.norm2.weight,
                mod,
                adaln_indices,
                self.ff.gate_up_proj.input_scale,
                self.norm2.variance_epsilon,
                2,
                4,
                3,
            )
            ff_input = Fp4QuantizedTensor(codes.reshape(-1, codes.shape[-1]), scales)
        else:
            hidden_states, norm_hidden_states = h3_gate_res_norm_mod(
                hidden_states,
                attn_output,
                self.norm2.weight,
                mod,
                adaln_indices,
                self.norm2.variance_epsilon,
                2,
                4,
                3,
            )
            ff_input = norm_hidden_states.reshape(-1, hidden_size)
        if fuse_gate_up or fuse_down:
            gate_up = self.ff.gate_up_proj(ff_input)
            if fuse_down:
                codes, scales = h3_swiglu_fp4(gate_up, self.ff.down_proj.input_scale)
                activation = Fp4QuantizedTensor(codes, scales)
            else:
                # GatedMLP's own SwiGLU; emits FP8 when down_proj is a static-FP8 Linear.
                activation = self.ff._apply_activation(gate_up)
            ff_output = self.ff.down_proj(activation, layer_idx=self.ff.layer_idx)
        else:
            ff_output = self.ff(ff_input)
        ff_output = ff_output.reshape_as(hidden_states)
        gate_mlp = mod[:, 5 * hidden_size : 6 * hidden_size]
        return hidden_states + gate_mlp.index_select(0, adaln_indices) * ff_output

    def _nvfp4_producers(self) -> tuple[bool, bool, bool]:
        """Which producers quantize straight into their consumer Linear: (norm1 -> qkv_proj,
        gated residual + norm2 -> gate_up_proj, SwiGLU -> down_proj).

        Each is decided per Linear (``fused_quant.static_nvfp4_input``: the plain static-NVFP4
        method with its calibrated input scale), so under a MIXED_PRECISION checkpoint an FP8
        Linear gets the BF16 producer and quantizes its input itself while its NVFP4 neighbours
        keep the fused producer. The norm1 producer also needs the attention on the FP8 path: an
        attention on BF16 operands takes the BF16 norm output and its ``qkv_proj`` quantizes it.
        Needs the fused-QKV attention and the fused gate / up MLP; the scenario's ``fused_quant``
        switch off keeps every producer unfused.
        """
        if not FUSED_QUANT_ENABLED or getattr(self.ff, "split_gate_up", False):
            return False, False, False
        qkv_proj = getattr(self.attn, "qkv_proj", None)
        return (
            qkv_proj is not None and static_nvfp4_input(qkv_proj) and self.attn.on_fp8_path(),
            static_nvfp4_input(self.ff.gate_up_proj),
            static_nvfp4_input(self.ff.down_proj),
        )


class ScenarioTransformer(MiniMaxH3Transformer3DModel):
    """The family transformer built from :class:`ScenarioBlock`, attention on the cuDNN backend's
    trtllm-gen route with the scenario's options."""

    block_cls = ScenarioBlock

    def __init__(self, model_config: DiffusionModelConfig) -> None:
        super().__init__(model_config)
        if self._attn_backend.upper() != "CUDNN":
            raise RuntimeError(
                "the minimax_h3_nvfp4__sm_107__u1_u4 scenario runs the CUDNN attention backend, "
                f"got {self._attn_backend}"
            )
        self._restrict_nvfp4_backends()
        logger.info(
            "MiniMax-H3 scenario transformer: fp8_dense_fmha=%s skip_corr_threshold=%s strided_receive=%s "
            "fp8_scale_fusion=%s grouped_cast=%s vamax_fold=%s amax_allreduce=%s fused_quant=%s out_quant=%s "
            "nvfp4_gemm_backends=%s",
            SCENARIO.attention_fmha,
            SCENARIO.skip_corr_threshold,
            SCENARIO.strided_receive,
            SCENARIO.fp8_scale_fusion,
            SCENARIO.grouped_cast,
            SCENARIO.vamax_fold,
            SCENARIO.amax_allreduce,
            SCENARIO.fused_quant,
            SCENARIO.out_quant,
            SCENARIO.nvfp4_gemm_backends,
        )

    def _restrict_nvfp4_backends(self) -> None:
        """Hand the scenario's backend tuple to every Linear of the transformer. The unified NVFP4 GEMM
        reads ``nvfp4_allowed_backends`` at each call and keys its AutoTuner entries on the tuple, so a
        single backend gives one output per shape; the family's construction stays untouched."""
        backends = list(SCENARIO.nvfp4_gemm_backends)
        for module in self.modules():
            if isinstance(module, Linear):
                module.nvfp4_allowed_backends = list(backends)

    def _attention_backend_kwargs(self) -> Optional[dict]:
        if self._attn_backend.upper() != "CUDNN":
            return None
        return {
            "fp8_dense_fmha": SCENARIO.attention_fmha,
            "fp8_skip_corr_threshold": SCENARIO.skip_corr_threshold,
            "fp8_strided_views": SCENARIO.strided_receive,
        }
