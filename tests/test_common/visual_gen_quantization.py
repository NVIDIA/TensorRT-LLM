# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small random models and execution-path observations for quantization ignore tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Literal
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.modules.linear import Linear, LinearMethodBase, UnquantizedLinearMethod
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen.attention.backends.interface import (
    AttentionBackend,
    AttentionTensorLayout,
)
from tensorrt_llm._torch.visual_gen.attention.backends.parallel import (
    Attention2DAttention,
    RingAttention,
    UlyssesAttention,
)
from tensorrt_llm._torch.visual_gen.attention.backends.quant_dispatcher import (
    AttentionQuantDispatcher,
)
from tensorrt_llm._torch.visual_gen.attention.backends.utils import create_attention
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.quantization.quant_dispatcher import (
    LinearQuantDispatcher,
    install_linear_quant_dispatcher,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig, DiffusionQuantConfig, QuantAttentionConfig

Component = Literal["linear", "attention"]
QuantizationRoute = Literal["bf16", "quantized"]
RouteKey = tuple[int, str, Component]


def _input_dtypes(values: tuple[object, ...]) -> list[str]:
    dtypes = []
    for value in values:
        if isinstance(value, torch.Tensor):
            dtypes.append(str(value.dtype))
        elif isinstance(value, Fp4QuantizedTensor):
            dtypes.append(str(value.fp4_tensor.dtype))
        elif isinstance(value, tuple):
            dtypes.extend(_input_dtypes(value))
        else:
            raise TypeError(f"Unexpected Linear/attention input type: {type(value).__name__}")
    return dtypes


@dataclass
class QuantizationTrace:
    """Actual method entries, with optional counters that execute inside GPU graphs.

    Counter order is Linear BF16, Linear quantized, attention BF16, attention quantized.
    Set ``step`` from the denoise-loop observer; ignore flags never determine a route.
    """

    step: int | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    device_counts: torch.Tensor | None = None
    device_only: bool = False

    def reset(self) -> None:
        self.events.clear()
        if self.device_counts is not None:
            self.device_counts.zero_()


def _record_entry(
    trace: QuantizationTrace,
    name: str,
    component: Component,
    route: QuantizationRoute,
    backend: str,
    inputs: tuple[object, ...],
    weight: torch.Tensor | None = None,
    entry_point: str = "apply",
) -> None:
    if trace.device_counts is not None:
        index = (0 if component == "linear" else 2) + (route == "quantized")
        trace.device_counts[index].add_(1)
    if not trace.device_only:
        trace.events.append(
            {
                "step": trace.step,
                "module": name,
                "component": component,
                "route": route,
                "backend": backend,
                "entry_point": entry_point,
                "input_dtypes": _input_dtypes(inputs),
                "weight_dtype": str(weight.dtype) if weight is not None else None,
            }
        )


@contextmanager
def observe_quantization_paths(
    model: nn.Module, *, device_only: bool = False
) -> Iterator[QuantizationTrace]:
    """Observe the called methods without replacing their calculations.

    Python events are for eager execution. ``device_only`` instruments the same entries
    with tensor increments, so compiled execution and CUDA Graph replay remain observable.
    """
    trace = QuantizationTrace(device_only=device_only)
    if device_only:
        trace.device_counts = torch.zeros(
            4, dtype=torch.int64, device=next(model.parameters()).device
        )
    seen: set[int] = set()
    with ExitStack() as stack:

        def observe_linear(
            method: LinearMethodBase | _ToyQuantizedLinearMethod,
            name: str,
            route: QuantizationRoute,
        ) -> None:
            original = method.apply

            def apply(
                layer: Linear | _ToyLinear,
                inputs: torch.Tensor | tuple | Fp4QuantizedTensor,
                bias: torch.Tensor | None,
                **kwargs: object,
            ) -> torch.Tensor:
                _record_entry(
                    trace,
                    name,
                    "linear",
                    route,
                    type(method).__name__,
                    (inputs,),
                    kwargs.get("weight", layer.weight),
                )
                return original(layer, inputs, bias, **kwargs)

            stack.enter_context(patch.object(method, "apply", apply))

        def observe_attention(
            backend: AttentionBackend, name: str, route: QuantizationRoute | None = None
        ) -> None:
            if id(backend) in seen:
                return
            seen.add(id(backend))
            if isinstance(backend, AttentionQuantDispatcher):
                observe_attention(backend.unquantized, name, "bf16")
                observe_attention(backend.quantized, name, "quantized")
                return
            if isinstance(backend, RingAttention):
                observe_attention(backend.inner, name, route)
                return
            if isinstance(backend, (UlyssesAttention, Attention2DAttention)):
                observe_attention(backend.inner_backend, name, route)
                return
            route = route or (
                "quantized"
                if getattr(backend, "quant_attention_config", None) is not None
                else "bf16"
            )

            def observe_call(method_name: str) -> None:
                original = getattr(backend, method_name)

                def forward(
                    q: torch.Tensor,
                    k: torch.Tensor | None = None,
                    v: torch.Tensor | None = None,
                    **kwargs: object,
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                    _record_entry(
                        trace,
                        name,
                        "attention",
                        route,
                        type(backend).__name__,
                        tuple(value for value in (q, k, v) if value is not None),
                        entry_point=method_name,
                    )
                    return original(q=q, k=k, v=v, **kwargs)

                stack.enter_context(patch.object(backend, method_name, forward))

            observe_call("forward")
            if backend.support_lse():
                observe_call("forward_with_lse")

        for name, module in model.named_modules():
            method = getattr(module, "quant_method", None)
            if isinstance(method, LinearQuantDispatcher):
                observe_linear(method.quantized_method, name, "quantized")
                observe_linear(method.unquantized_method, name, "bf16")
            elif method is not None:
                route = "bf16" if type(method) is UnquantizedLinearMethod else "quantized"
                observe_linear(method, name, route)
            for attribute in ("attn", "backend"):
                backend = getattr(module, attribute, None)
                if isinstance(backend, AttentionBackend):
                    observe_attention(backend, name)
        yield trace


def assert_quantization_routes(
    trace: QuantizationTrace, expected: Mapping[RouteKey, QuantizationRoute]
) -> None:
    """Require every expected module/step to execute exclusively its expected method."""
    observed = {(event["step"], event["module"], event["component"]) for event in trace.events}
    assert observed == set(expected), {
        "missing": set(expected) - observed,
        "extra": observed - set(expected),
    }
    for event in trace.events:
        key = (event["step"], event["module"], event["component"])
        assert event["route"] == expected[key], event
        if event["route"] == "bf16":
            assert event["input_dtypes"] and set(event["input_dtypes"]) == {"torch.bfloat16"}, event
            if event["component"] == "linear":
                assert event["weight_dtype"] == "torch.bfloat16", event


class ToyAttentionBackend(AttentionBackend):
    """CPU attention kernel double; production factory and dispatch remain under test."""

    def __init__(
        self,
        *,
        num_heads: int = 1,
        head_dim: int = 128,
        quant_attention_config: QuantAttentionConfig | None = None,
        **kwargs: object,
    ) -> None:
        self.num_heads = self.num_kv_heads = num_heads
        self.head_dim = head_dim
        self.quant_attention_config = quant_attention_config

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    def support_lse(self) -> bool:
        return True

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(q, k, v, **kwargs), q.new_empty(q.shape[0], q.shape[2], q.shape[1])


class _ToyQuantizedLinearMethod:
    """CPU quantized-kernel double with FP8 storage; no kernel accuracy is asserted."""

    requires_unquantized_activation = False
    quantizes_nvfp4_activations = False

    def apply(
        self, module: _ToyLinear, inputs: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return F.linear(inputs, module.weight.to(inputs.dtype), bias)

    def apply_slice(
        self, module: _ToyLinear, inputs: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        bias = module.bias[start:end] if module.bias is not None else None
        return F.linear(inputs, module.weight[start:end].to(inputs.dtype), bias)

    def prepare_output_slice(self, module: _ToyLinear, start: int, end: int) -> None:
        pass


class _ToyLinear(nn.Linear):
    use_cute_dsl_bf16_gemm = False
    use_custom_cublas_mm = False
    all_reduce = None

    def __init__(self, quantized: bool, retain_weight: bool) -> None:
        super().__init__(128, 128, bias=False, dtype=torch.bfloat16)
        self.weight.requires_grad_(False)
        self.quant_config = QuantConfig(quant_algo="FP8" if quantized else None)
        if retain_weight and quantized:
            self.register_buffer(
                "_unquantized_weight", self.weight.detach().clone(), persistent=False
            )
        if quantized:
            self.weight = nn.Parameter(self.weight.to(torch.float8_e4m3fn), requires_grad=False)
        self.quant_method = _ToyQuantizedLinearMethod() if quantized else UnquantizedLinearMethod()
        self.use_fused_gemm_allreduce = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, inputs, self.bias)


class _ToyAttention(nn.Module):
    def __init__(self, backend: AttentionBackend) -> None:
        super().__init__()
        self.attn = backend

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        qkv = inputs.unsqueeze(2)
        return self.attn(
            qkv, qkv, qkv, batch_size=inputs.shape[0], seq_len=inputs.shape[1]
        ).squeeze(2)


class ToyDiTBlock(nn.Module):
    """One Linear and one attention module, with explicit random BF16 source weights."""

    def __init__(self, config: DiffusionModelConfig, index: int, *, real_kernels: bool) -> None:
        super().__init__()
        name = f"blocks.{index}.linear"
        quant = config.quant_config
        excluded = quant.is_module_excluded_from_quantization(name)
        if real_kernels:
            self.linear = Linear(
                128,
                128,
                bias=False,
                dtype=torch.bfloat16,
                quant_config=QuantConfig() if excluded else quant,
                force_dynamic_quantization=True,
            )
            weight = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda") * 0.02
            DynamicLinearWeightLoader(config).load_linear_weights(
                self.linear, name, [{"weight": weight}]
            )
            self.linear.post_load_weights()
        else:
            self.linear = _ToyLinear(
                quant.quant_algo is not None and not excluded, bool(quant.ignored_steps)
            )
        arguments = dict(
            backend=config.attention.backend,
            layer_idx=index,
            num_heads=1,
            head_dim=128,
            dtype=torch.bfloat16,
            attention_config=config.attention,
            attention_metadata_state=config.attention_metadata_state,
            module_name=f"blocks.{index}.attention",
        )
        if real_kernels:
            backend = create_attention(**arguments)
        else:
            with patch(
                "tensorrt_llm._torch.visual_gen.attention.backends.utils.get_visual_gen_attention_backend",
                return_value=ToyAttentionBackend,
            ):
                backend = create_attention(**arguments)
        self.attention = _ToyAttention(backend)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.attention(self.linear(inputs))


class ToyDiT(nn.Module):
    """Generic routing fixture; GPU mode uses actual NVFP4 Linear/CuTe DSL kernels."""

    def __init__(
        self,
        *,
        block_count: int = 2,
        linear_ignore: tuple[str, ...] = (),
        attention_ignore: tuple[str, ...] = (),
        linear_quant_ignored_steps: tuple[int, ...] = (),
        attention_quant_ignored_steps: tuple[int, ...] = (),
        quantized_linear: bool = True,
        quantized_attention: bool = True,
        real_kernels: bool = False,
        install_dispatch: bool = True,
    ) -> None:
        super().__init__()
        self.model_config = DiffusionModelConfig(
            quant_config=DiffusionQuantConfig(
                quant_algo=("NVFP4" if real_kernels else "FP8") if quantized_linear else None,
                exclude_modules=list(linear_ignore),
                ignored_steps=list(linear_quant_ignored_steps),
            ),
            dynamic_weight_quant=True,
            force_dynamic_quantization=True,
            attention=AttentionConfig(
                backend="CUTEDSL",
                quant_attention_config=QuantAttentionConfig(
                    qk_dtype="nvfp4",
                    v_dtype="fp8",
                    ignore=list(attention_ignore),
                    ignored_steps=list(attention_quant_ignored_steps),
                )
                if quantized_attention
                else None,
            ),
            attention_metadata_state={},
        )
        self.blocks = nn.ModuleList(
            ToyDiTBlock(self.model_config, index, real_kernels=real_kernels)
            for index in range(block_count)
        )
        if linear_quant_ignored_steps and install_dispatch:
            install_linear_quant_dispatcher(self, self.model_config.attention_metadata_state)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            inputs = block(inputs)
        return inputs
