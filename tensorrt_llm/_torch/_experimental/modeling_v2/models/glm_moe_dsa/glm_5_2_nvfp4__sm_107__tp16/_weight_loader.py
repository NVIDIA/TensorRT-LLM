# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# --------------------------------------------------
# Portions of this code were derived from DeepSeek‑V3:
#   https://github.com/deepseek-ai/DeepSeek-V3
#
# MIT License

# Copyright (c) 2023 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# --------------------------------------------------

"""Checkpoint loading and weight layout conversion owned by the GLM-5.2 target."""

from typing import Dict, List

import torch
import triton
import triton.language as tl
from torch import nn
from tqdm import tqdm

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.modeling_utils import filter_weights
from tensorrt_llm._torch.moe.fused_moe import is_moe_weight_owner


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"])

    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


class Glm52WeightLoader:
    def __init__(self, model, is_draft_model: bool = False):
        self.model = model
        self.config = model.config
        self.model_config = model.model_config
        self.is_draft_model = is_draft_model

    def load_weights(self, weights: ConsumableWeightsDict, skip_modules: List[str] = []):
        def requantize_weight_with_new_scale(
            weight, weight_scale, old_scale_2, new_scale_2, device
        ):
            """
            Dequantize FP4 weights and requantize with a new scale.

            Args:
                weight: FP4 quantized weight tensor 2D [,]
                weight_scale: FP8 per-block scaling factors
                old_scale_2: original global scale (amax/(448*6))
                new_scale_2: new global scale (amax/(448*6))
                device: target device for computation

            Returns:
                (requantized_weight, new_weight_scale)
            """
            # Remember original dtype of weight_scale
            original_scale_dtype = weight_scale.dtype
            original_scale_shape = weight_scale.shape

            # Dequantize
            dequant_shape = (weight.shape[0], weight.shape[1] * 2)
            weight_dequant = (
                torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
                    weight.contiguous(),
                    weight_scale.flatten().view(fp4_utils.float4_sf_dtype).contiguous(),
                    old_scale_2,
                    16,
                    1,
                    True,
                )
                .to(dtype=torch.bfloat16)
                .reshape(dequant_shape)
            )

            # Requantize using the new_scale_2
            weight_requant, weight_scale_requant = torch.ops.trtllm.fp4_quantize(
                weight_dequant.to(device),
                1.0 / new_scale_2.to(device),
                16,  # scaling_vector_size
                False,
            )

            # Ensure the returned scale has the same dtype as the input scale
            return weight_requant.cpu(), weight_scale_requant.reshape(original_scale_shape).view(
                original_scale_dtype
            ).cpu()

        def rename_moe_weight(weights: Dict, rename_rules: Dict):
            result = {}
            for key, value in weights.items():
                new_key = key
                for old, new in rename_rules.items():
                    new_key = new_key.replace(old, new)
                result[new_key] = value
            return result

        ## Prepare weights for TP
        def split(v, tp_size, idx, dim=0):
            if tp_size == 1:
                return v
            if len(v.shape) == 1:
                return torch.chunk(v, tp_size)[idx].contiguous()
            return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()

        def split_matrix_tp(v, tensor_parallel, rank, dim):
            return split(v, tensor_parallel, rank, dim=dim)

        def load_kv_b_proj_and_k_b_proj_trans(module_name: str, is_scale: bool) -> torch.Tensor:
            weight_name = "weight" if not is_scale else "weight_scale_inv"
            local_qk_nope_head_dim = qk_nope_head_dim if not is_scale else qk_nope_head_dim // 128
            local_v_head_dim = v_head_dim if not is_scale else v_head_dim // 128
            local_kv_lora_rank = kv_lora_rank if not is_scale else kv_lora_rank // 128

            kv_b_proj = weights[f"{module_name}.{weight_name}"][:].unflatten(
                0,
                [
                    num_heads,
                    local_qk_nope_head_dim + local_v_head_dim,
                ],
            )

            if not self.model_config.mapping.enable_attention_dp:
                kv_b_proj = split_matrix_tp(kv_b_proj, tp_size, tp_rank, 0)
            k_nope_weight, v_weight = kv_b_proj.split(
                [local_qk_nope_head_dim, local_v_head_dim],
                dim=1,
            )
            weight_divisor = 1 if self.model_config.mapping.enable_attention_dp else tp_size
            local_num_heads = num_heads // weight_divisor

            k_nope_weight_trans = k_nope_weight.transpose(2, 1).contiguous()

            kv_b_proj = torch.concat(
                [
                    k_nope_weight.reshape(
                        local_num_heads * local_qk_nope_head_dim, local_kv_lora_rank
                    ),
                    v_weight.reshape(local_num_heads * local_v_head_dim, local_kv_lora_rank),
                ],
                dim=0,
            )

            return kv_b_proj, k_nope_weight_trans

        def load_kv_b_proj_and_k_b_proj_trans_dequant(module_name: str) -> torch.Tensor:
            weight_name = "weight"
            local_qk_nope_head_dim = qk_nope_head_dim
            local_v_head_dim = v_head_dim
            local_kv_lora_rank = kv_lora_rank

            kv_b_proj = weights[f"{module_name}.{weight_name}"][:].cuda()

            weight_name = "weight_scale_inv"
            kv_b_proj_scale = weights[f"{module_name}.{weight_name}"][:].cuda()

            kv_b_proj = weight_dequant(kv_b_proj, kv_b_proj_scale)
            kv_b_proj = kv_b_proj.unflatten(
                0,
                [
                    num_heads,
                    local_qk_nope_head_dim + local_v_head_dim,
                ],
            )
            if not self.model_config.mapping.enable_attention_dp:
                kv_b_proj = split_matrix_tp(kv_b_proj, tp_size, tp_rank, 0)
            k_nope_weight, v_weight = kv_b_proj.split(
                [local_qk_nope_head_dim, local_v_head_dim],
                dim=1,
            )
            weight_divisor = 1 if self.model_config.mapping.enable_attention_dp else tp_size
            local_num_heads = num_heads // weight_divisor

            k_nope_weight_trans = k_nope_weight.transpose(2, 1).contiguous()

            kv_b_proj = torch.concat(
                [
                    k_nope_weight.reshape(
                        local_num_heads * local_qk_nope_head_dim, local_kv_lora_rank
                    ),
                    v_weight.reshape(local_num_heads * local_v_head_dim, local_kv_lora_rank),
                ],
                dim=0,
            )

            return kv_b_proj, k_nope_weight_trans

        def split_kv_b_proj(kv_b_proj: torch.Tensor, is_scale: bool) -> torch.Tensor:
            local_qk_nope_head_dim = qk_nope_head_dim if not is_scale else qk_nope_head_dim // 128
            local_v_head_dim = v_head_dim if not is_scale else v_head_dim // 128

            weight_divisor = 1 if self.model_config.mapping.enable_attention_dp else tp_size
            local_num_heads = num_heads // weight_divisor

            k_b_proj, v_b_proj = kv_b_proj.split(
                [local_num_heads * local_qk_nope_head_dim, local_num_heads * local_v_head_dim],
                dim=0,
            )
            k_b_proj = k_b_proj.view([local_num_heads, local_qk_nope_head_dim, -1])
            v_b_proj = v_b_proj.view([local_num_heads, local_v_head_dim, -1])

            if cp_size > 1:
                local_cp_heads = local_num_heads // cp_size
                k_b_proj = k_b_proj[cp_rank * local_cp_heads : (cp_rank + 1) * local_cp_heads]
                v_b_proj = v_b_proj[cp_rank * local_cp_heads : (cp_rank + 1) * local_cp_heads]

            return k_b_proj, v_b_proj

        is_lite = self.config.q_lora_rank is None
        num_heads = self.config.num_attention_heads
        qk_nope_head_dim = self.config.qk_nope_head_dim
        v_head_dim = self.config.v_head_dim
        kv_lora_rank = self.config.kv_lora_rank

        tp_rank = self.model_config.mapping.tp_rank
        tp_size = self.model_config.mapping.tp_size
        cp_rank = self.model_config.mapping.cp_rank
        cp_size = self.model_config.mapping.cp_size

        params_map = {"gate_up_proj": ["gate_proj", "up_proj"]}
        all_named_modules = dict(self.model.named_modules())

        # Check if weights supports mark_consumed (ConsumableWeightsDict)
        can_mark_consumed = hasattr(weights, "mark_consumed")

        # The pretrained_config's num_nextn_predict_layers may have been expanded
        # by ModelLoader._load_and_validate_config to match user max_draft_len.
        # The original checkpoint MTP layer count (used for mod-indexing) is
        # preserved as `_ckpt_num_nextn_predict_layers`.
        ckpt_num_nextn_predict_layers = getattr(
            self.config, "_ckpt_num_nextn_predict_layers", None
        ) or getattr(self.config, "num_nextn_predict_layers", None)

        def detect_shared_mtp_weights() -> bool:
            # Detect if MTP layers share checkpoint weights (model has more MTP
            # layer instances than the checkpoint provides). In this case,
            # multiple model MTP layers map to the same checkpoint layer via
            # modulo, and mark_consumed must be skipped to avoid deleting
            # weights that later MTP layers still need.
            model_nextn = getattr(self.config, "num_nextn_predict_layers", None) or 0
            return model_nextn > (ckpt_num_nextn_predict_layers or 0) > 0

        has_shared_mtp_weights = detect_shared_mtp_weights()

        for name, module in tqdm(all_named_modules.items(), desc="Loading weights"):
            if len(module._parameters) <= 0 or name.startswith("draft_model"):
                continue
            elif any(skip_module in name for skip_module in skip_modules):
                continue
            else:
                names = name.split(".")
                parent_module_name = ".".join(names[:-1])
                is_shared_mtp_layer = False
                if "model.layers" in name and int(names[2]) >= self.config.num_hidden_layers:
                    is_shared_mtp_layer = has_shared_mtp_weights
                    mtp_layer_idx = int(names[2]) - self.config.num_hidden_layers
                    names[2] = str(
                        mtp_layer_idx % ckpt_num_nextn_predict_layers
                        + self.config.num_hidden_layers
                    )
                    name = ".".join(names)
                mark_consumed = can_mark_consumed and not is_shared_mtp_layer
                if names[-1] == "kv_b_proj":
                    # TODO: remove weight_dequant after enabling fp8_bmm
                    dequant_kv_b_proj = (
                        self.model_config.quant_config.is_module_excluded_from_quantization(
                            names[-1]
                        )
                    )
                    if dequant_kv_b_proj:
                        kv_b_proj, k_b_proj_trans = load_kv_b_proj_and_k_b_proj_trans_dequant(name)
                    else:
                        kv_b_proj, k_b_proj_trans = load_kv_b_proj_and_k_b_proj_trans(
                            name, is_scale=False
                        )
                    module.weight.data.copy_(kv_b_proj.reshape(module.weight.shape))

                    attn_module = all_named_modules[parent_module_name]
                    _, v_b_proj = split_kv_b_proj(module.weight.data, is_scale=False)
                    attn_module._bind_v_b_proj_weight(v_b_proj)

                    attn_module.k_b_proj_trans.data.copy_(
                        k_b_proj_trans.reshape(attn_module.k_b_proj_trans.shape)
                    )

                    if getattr(module, "weight_scale", None) is not None and not dequant_kv_b_proj:
                        kv_b_proj_scale, k_b_proj_trans_scale = load_kv_b_proj_and_k_b_proj_trans(
                            name, is_scale=True
                        )
                        module.weight_scale.copy_(
                            kv_b_proj_scale.reshape(module.weight_scale.shape)
                        )
                        attn_module.k_b_proj_trans_scale.copy_(
                            k_b_proj_trans_scale.reshape(attn_module.k_b_proj_trans_scale.shape)
                        )

                        _, v_b_proj_scale = split_kv_b_proj(module.weight_scale.data, is_scale=True)
                        attn_module.v_b_proj_scale = nn.Parameter(
                            v_b_proj_scale, requires_grad=False
                        )

                        if attn_module.k_b_proj_trans_dequant is not None:
                            attn_module.k_b_proj_trans_dequant.data.copy_(
                                weight_dequant(
                                    k_b_proj_trans.view(-1, k_b_proj_trans.shape[-1]).cuda(),
                                    k_b_proj_trans_scale.view(
                                        -1, k_b_proj_trans_scale.shape[-1]
                                    ).cuda(),
                                )
                                .view(*attn_module.k_b_proj_trans_dequant.shape)
                                .to(attn_module.k_b_proj_trans_dequant.dtype)
                            )
                        if attn_module.v_b_proj_dequant is not None:
                            attn_module.v_b_proj_dequant.data.copy_(
                                weight_dequant(
                                    v_b_proj.view(-1, v_b_proj.shape[-1]).cuda(),
                                    v_b_proj_scale.view(-1, v_b_proj_scale.shape[-1]).cuda(),
                                )
                                .view(*attn_module.v_b_proj_dequant.shape)
                                .to(attn_module.v_b_proj_dequant.dtype)
                            )
                    # Mark consumed kv_b_proj weights
                    if mark_consumed:
                        weights.mark_consumed(name)
                elif names[-1] == "kv_a_proj_with_mqa":
                    nvfp4_fused_a = (
                        self.model_config.get_quant_config().layer_quant_mode.has_nvfp4()
                        and weights[f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight"].dtype
                        == fp4_utils.float4_e2m1x2
                    )
                    # Non-lite models (V3, R1, V3.2) fuse q_a_proj into
                    # kv_a_proj_with_mqa, so both must be NVFP4 for the fused
                    # path. Lite models (V3-Lite) have no q_a_proj.
                    if not is_lite and nvfp4_fused_a:
                        nvfp4_fused_a = (
                            weights[f"{'.'.join(names[:-1])}.q_a_proj.weight"].dtype
                            == fp4_utils.float4_e2m1x2
                        )
                    if nvfp4_fused_a:
                        ########### input_scale
                        kv_a_proj_with_mqa_input_scale = weights[
                            f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.input_scale"
                        ]
                        if not is_lite:
                            q_a_proj_input_scale = weights[
                                f"{'.'.join(names[:-1])}.q_a_proj.input_scale"
                            ]
                            assert kv_a_proj_with_mqa_input_scale == q_a_proj_input_scale, (
                                "kv_a_proj_with_mqa.input_scale and q_a_proj.input_scale should be the same"
                            )
                        # modelopt ckpt stores amax/(448*6), convert to (448*6)/amax
                        shared_input_scale = kv_a_proj_with_mqa_input_scale
                        module.input_scale.data.copy_(1.0 / shared_input_scale)
                        E2M1_MAX = 6.0
                        module.inv_input_scale.data.copy_(module.input_scale / E2M1_MAX)
                        ########### weight_scale_2
                        need_requant_kv_a_proj_with_mqa = False
                        need_requant_q_a_proj = False
                        kv_a_proj_with_mqa_scale_2 = weights[
                            f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight_scale_2"
                        ]
                        shared_weight_scale_2 = kv_a_proj_with_mqa_scale_2
                        if not is_lite:
                            q_a_proj_scale_2 = weights[
                                f"{'.'.join(names[:-1])}.q_a_proj.weight_scale_2"
                            ]
                            if kv_a_proj_with_mqa_scale_2 < q_a_proj_scale_2:
                                shared_weight_scale_2 = q_a_proj_scale_2
                                need_requant_kv_a_proj_with_mqa = True
                            elif q_a_proj_scale_2 < kv_a_proj_with_mqa_scale_2:
                                need_requant_q_a_proj = True

                        ########### alpha
                        alpha = shared_input_scale.float() * shared_weight_scale_2.float()
                        module.alpha.data.copy_(alpha)
                        module.scalar_alpha = alpha.item()

                        ########### weights
                        kv_a_proj_with_mqa = weights[
                            f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight"
                        ][:]

                        if not is_lite:
                            q_a_proj = weights[f"{'.'.join(names[:-1])}.q_a_proj.weight"][:]

                        ########### weight_scale
                        kv_a_proj_with_mqa_scale = weights[
                            f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight_scale"
                        ][:]
                        kv_a_proj_with_mqa_scale = torch.ops.trtllm.block_scale_interleave(
                            kv_a_proj_with_mqa_scale.view(fp4_utils.float4_sf_dtype)
                        )
                        if not is_lite:
                            q_a_proj_scale = weights[
                                f"{'.'.join(names[:-1])}.q_a_proj.weight_scale"
                            ][:]
                            q_a_proj_scale = torch.ops.trtllm.block_scale_interleave(
                                q_a_proj_scale.view(fp4_utils.float4_sf_dtype)
                            )

                        ########### requantize
                        if need_requant_kv_a_proj_with_mqa:
                            # requant kv_a_proj_with_mqa
                            kv_a_proj_with_mqa, kv_a_proj_with_mqa_scale = (
                                requantize_weight_with_new_scale(
                                    kv_a_proj_with_mqa,
                                    kv_a_proj_with_mqa_scale,
                                    kv_a_proj_with_mqa_scale_2,
                                    shared_weight_scale_2,
                                    device=module.weight.device,
                                )
                            )
                        if need_requant_q_a_proj:
                            # requant q_a_proj
                            q_a_proj, q_a_proj_scale = requantize_weight_with_new_scale(
                                q_a_proj,
                                q_a_proj_scale,
                                q_a_proj_scale_2,
                                shared_weight_scale_2,
                                device=module.weight.device,
                            )

                        ########### fuse and load weights
                        if not is_lite:
                            fused_a = torch.cat([q_a_proj, kv_a_proj_with_mqa], dim=0)
                        else:
                            fused_a = kv_a_proj_with_mqa

                        module.weight.data[0 : fused_a.shape[0]].copy_(fused_a)

                        ########### fuse weight_scale
                        if not is_lite:
                            fused_a_scale = torch.cat(
                                [q_a_proj_scale, kv_a_proj_with_mqa_scale], dim=0
                            )
                        else:
                            fused_a_scale = kv_a_proj_with_mqa_scale
                        module.weight_scale.data[0 : fused_a_scale.shape[0]].copy_(fused_a_scale)
                    else:
                        fused_a = weights[f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight"][:]
                        if not is_lite:
                            q_a_proj = weights[f"{'.'.join(names[:-1])}.q_a_proj.weight"][:]
                            fused_a = torch.cat([q_a_proj, fused_a], dim=0)

                        if f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight_scale_inv" in weights:
                            fused_a_scale = weights[
                                f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight_scale_inv"
                            ]
                            if not is_lite:
                                q_a_proj_scale = weights[
                                    f"{'.'.join(names[:-1])}.q_a_proj.weight_scale_inv"
                                ][:]
                                fused_a_scale = torch.cat([q_a_proj_scale, fused_a_scale], dim=0)

                            module.weight_scale.data[0 : fused_a_scale.shape[0]].copy_(
                                fused_a_scale
                            )
                        module.weight.data[0 : fused_a.shape[0]].copy_(fused_a)
                    # Mark consumed kv_a_proj_with_mqa and q_a_proj weights
                    if mark_consumed:
                        parent_prefix = ".".join(names[:-1])
                        weights.mark_consumed(f"{parent_prefix}.kv_a_proj_with_mqa")
                        if not is_lite:
                            weights.mark_consumed(f"{parent_prefix}.q_a_proj")
                elif names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        module_weights.append(
                            filter_weights(".".join(names[:-1] + [new_name]), weights)
                        )
                    module.load_weights(weights=module_weights)
                    # Mark consumed source weights (e.g., gate_proj, up_proj)
                    if mark_consumed:
                        for src_name in params_map[names[-1]]:
                            weights.mark_consumed(".".join(names[:-1] + [src_name]))
                elif names[-1] == "experts":
                    module_weights = filter_weights(name, weights)
                    module_weights = rename_moe_weight(
                        module_weights,
                        {
                            "down_proj": "w2",
                            "up_proj": "w3",
                            "gate_proj": "w1",
                        },
                    )
                    module.load_weights(weights=[module_weights])
                    # Mark consumed experts weights
                    if mark_consumed:
                        weights.mark_consumed(name)
                elif names[-1] == "backend" and is_moe_weight_owner(module):
                    # Special case: ConfigurableMoE.backend (TRTLLMGenFusedMoE)
                    # Currently saved MoE weights don't include 'backend' in their names.
                    # After MoE refactoring, ConfigurableMoE now has a backend submodule,
                    # and weights loading is done in the backend, so module name includes '.backend'.
                    # We need to use parent module name (without .backend) to match saved weight names.
                    # After MoE refactoring is fully complete, all paths will follow this branch.
                    parent_name = ".".join(names[:-1])
                    module_weights = filter_weights(parent_name, weights)
                    module_weights = rename_moe_weight(
                        module_weights,
                        {
                            "down_proj": "w2",
                            "up_proj": "w3",
                            "gate_proj": "w1",
                        },
                    )
                    module.load_weights(weights=[module_weights])
                    # Mark consumed MoE weights using parent name
                    if mark_consumed:
                        weights.mark_consumed(parent_name)
                elif names[-1] == "self_attn":
                    continue
                elif names[-1] == "next_layer_layernorm":
                    continue
                else:
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, "load_weights"):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module.named_parameters():
                            p.data.copy_(module_weights[n][:])
                    # Mark consumed weights
                    if mark_consumed:
                        weights.mark_consumed(name)
