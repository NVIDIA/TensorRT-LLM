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
"""Shared lightweight fakes for SourceIdentity unit tests.

These expose only the attributes the fingerprint projection reads, so the
tests touch no real model, GPU, or weights. Used by both
``test_source_identity.py`` and ``test_mx_source_identity_gate.py``.
"""

from typing import Optional, Sequence

from tensorrt_llm._torch.weight_sharing import SourceIdentity

_UNSET = object()


class FakePretrainedConfig:
    """Minimal stand-in for a HF pretrained config."""

    def __init__(
        self,
        architectures: Sequence[str] = ("LlamaForCausalLM",),
        torch_dtype: str = "bf16",
        **attrs,
    ):
        self.architectures = list(architectures)
        self.torch_dtype = torch_dtype
        for name, value in attrs.items():
            setattr(self, name, value)

    def to_dict(self) -> dict:
        payload = {
            "architectures": self.architectures,
            "torch_dtype": self.torch_dtype,
        }
        payload.update(
            {name: value for name, value in self.__dict__.items() if name not in payload}
        )
        return payload


class FakeQuantConfig:
    """Minimal stand-in for a pydantic ``QuantConfig``."""

    def __init__(self, quant_algo: str = "FP8", kv_cache_quant_algo: Optional[str] = None):
        self.quant_algo = quant_algo
        self.kv_cache_quant_algo = kv_cache_quant_algo

    def model_dump(self, mode: str = "json") -> dict:
        return {
            "quant_algo": self.quant_algo,
            "kv_cache_quant_algo": self.kv_cache_quant_algo,
        }


class FakeQuantConfigWithPythonOnlyField(FakeQuantConfig):
    """Quant config whose JSON dump fails but Python dump succeeds."""

    def __init__(
        self,
        quant_algo: str = "FP8",
        kv_cache_quant_algo: Optional[str] = None,
        dtype: object = "torch.float16",
    ):
        super().__init__(quant_algo=quant_algo, kv_cache_quant_algo=kv_cache_quant_algo)
        self.dtype = dtype

    def model_dump(self, mode: str = "json") -> dict:
        if mode == "json":
            raise TypeError("Unable to serialize unknown type: torch.dtype")
        payload = super().model_dump(mode=mode)
        payload["dtype"] = self.dtype
        return payload


class FakeMapping:
    """Minimal stand-in for ``Mapping`` exposing layout attributes only.

    ``world_size`` defaults to ``tp_size * pp_size`` when not given.
    """

    def __init__(
        self,
        *,
        rank: int = 0,
        tp_size: int = 8,
        pp_size: int = 1,
        cp_size: int = 1,
        moe_tp_size: int = 1,
        moe_ep_size: int = 1,
        moe_cluster_size: int = -1,
        attn_tp_size: int = -1,
        attn_cp_size: int = -1,
        enable_attention_dp: bool = False,
        gpus_per_node: int = 8,
        world_size: Optional[int] = None,
        tp_rank: int = 0,
        pp_rank: int = 0,
        cp_rank: int = 0,
        moe_tp_rank: int = 0,
        moe_ep_rank: int = 0,
    ):
        self.rank = rank
        self.world_size = world_size if world_size is not None else tp_size * pp_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.cp_size = cp_size
        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        self.moe_cluster_size = moe_cluster_size
        self.attn_tp_size = attn_tp_size
        self.attn_cp_size = attn_cp_size
        self.enable_attention_dp = enable_attention_dp
        self.gpus_per_node = gpus_per_node
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.cp_rank = cp_rank
        self.moe_tp_rank = moe_tp_rank
        self.moe_ep_rank = moe_ep_rank


class FakeModelConfig:
    """Minimal stand-in for a torch-backend ``ModelConfig``.

    ``quant_config`` defaults to a :class:`FakeQuantConfig`; pass ``None``
    explicitly to fingerprint an unquantized model.
    """

    def __init__(
        self,
        *,
        mapping: Optional[FakeMapping] = None,
        pretrained_config: Optional[FakePretrainedConfig] = None,
        quant_config=_UNSET,
        attn_backend: str = "TRTLLM",
        moe_backend: str = "CUTLASS",
    ):
        self.mapping = mapping or FakeMapping()
        self.pretrained_config = (
            pretrained_config if pretrained_config is not None else FakePretrainedConfig()
        )
        self.quant_config = FakeQuantConfig() if quant_config is _UNSET else quant_config
        self.quant_config_dict = None
        self.force_dynamic_quantization = False
        self.attn_backend = attn_backend
        self.moe_backend = moe_backend
        self.nvfp4_gemm_allowed_backends = ["cutlass", "cublaslt", "cuda_core"]
        self.moe_disable_finalize_fusion = False
        self.use_low_precision_moe_combine = False
        self.enable_min_latency = False
        self.allreduce_strategy = "AUTO"
        self.use_cute_dsl_blockscaling_mm = False
        self.use_cute_dsl_blockscaling_bmm = False
        self.use_cute_dsl_bf16_bmm = False
        self.use_cute_dsl_bf16_gemm = False


class _FakeTensor:
    """Stand-in for a parameter/buffer exposing only ``shape`` and ``dtype``."""

    def __init__(self, shape: Sequence[int], dtype: str):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeModel:
    """Minimal stand-in for an ``nn.Module`` exposing named tensors.

    Parameter shapes are derived from the pretrained config so that
    layout-affecting fields (``hidden_size`` etc.) change the realized layout
    fingerprint, mirroring how real modules behave. ``dtype`` models a runtime
    compute-dtype override that a config-only projection would miss.
    """

    def __init__(self, pretrained_config: FakePretrainedConfig, *, dtype: str = "torch.bfloat16"):
        hidden = int(getattr(pretrained_config, "hidden_size", 4096))
        vocab = int(getattr(pretrained_config, "vocab_size", 32000))
        intermediate = int(getattr(pretrained_config, "intermediate_size", 11008))
        self._params = {
            "embed_tokens.weight": _FakeTensor((vocab, hidden), dtype),
            "mlp.gate_proj.weight": _FakeTensor((intermediate, hidden), dtype),
            "mlp.down_proj.weight": _FakeTensor((hidden, intermediate), dtype),
        }
        self._buffers = {
            "rotary.inv_freq": _FakeTensor((hidden // 2,), "torch.float32"),
        }

    def named_parameters(self):
        return list(self._params.items())

    def named_buffers(self):
        return list(self._buffers.items())


def identity_from(config: FakeModelConfig, *, model_name: Optional[str] = None) -> SourceIdentity:
    """Build a :class:`SourceIdentity` from a fake config and derived model."""
    return SourceIdentity.from_model_config(
        config, FakeModel(config.pretrained_config), model_name=model_name
    )


def make_identity(
    *, attn_backend: str = "TRTLLM", rank: int = 0, model_name: str = "m"
) -> SourceIdentity:
    """Build a :class:`SourceIdentity` from a fake config for ``rank``."""
    cfg = FakeModelConfig(mapping=FakeMapping(rank=rank, tp_rank=rank), attn_backend=attn_backend)
    return identity_from(cfg, model_name=model_name)
