# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the family's routed-expert module layer (``ops/moe/k3r_experts.py``).

One MoE layer of random checkpoint-format NVFP4 experts (E2M1 codes and power-of-two E4M3 block
scales drawn directly, weight global scales 1, so every weight is exact in FP32) is loaded through
``K3Nvfp4Experts.load_expert`` / ``finalize`` and run through ``K3RoutedExperts.forward``.

``CUTEDSL`` (SM103 / SM107):

* against an FP32 reference on the same NVFP4 input (the module's own quantization) with the
  FC1 output requantized to NVFP4 as the kernel does (``_fp4_moe_test_utils._q_nvfp4``);
* with ``x`` written on another stream and joined through ``input_ready_event``, and chunked
  (``max_chunk_tokens`` below the token count): the plain call's output up to the order of the
  finalize's BF16 scatter-add (:data:`ORDER_ULP`).

``CUTEDSL_FC12`` (SM107 only; skipped elsewhere): the output with a pre-zeroed output and counters
(``new_zeroed_output`` + ``zero_on_side_stream``) is the plain call's up to the scatter-add order,
and the same with die-major weights and die-local claiming from 24 tokens where the GPU has two
dies.

    pytest -s test_modeling_v2_k3r_experts.py
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../../../.."]
from _fp4_moe_test_utils import _q_nvfp4  # noqa: E402
from _validating import assert_within_ulp  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.moe import (  # noqa: E402
    k3r_experts as kx,
)

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE
    or not torch.cuda.is_available()
    or get_sm_version() not in (103, 107),
    reason="the K3 CuTe DSL routed experts run on SM103 / SM107",
)

DEV = "cuda"
NUM_EXPERTS, TOP_K, HIDDEN, INTER = 64, 8, 1024, 512
SITU_BETA, SITU_LINEAR_BETA = 4.0, 4.0
SV = 16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
FP4_RANGE = 448.0 * 6.0
needs_fc12 = pytest.mark.skipif(
    get_sm_version() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="CUTEDSL_FC12 runs on SM107 with the Rubin CuTe DSL",
)


def _rand_nvfp4(n: int, k: int, gen: torch.Generator):
    """``(packed uint8 [n, k / 2], scales e4m3 [n, k / 16], fp32 value [n, k])``."""
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device=DEV, generator=gen)
    exps = torch.randint(-2, 1, (n, k // SV), device=DEV, generator=gen).float()
    sf = torch.exp2(exps).to(torch.float8_e4m3fn)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    value = torch.tensor(E2M1, device=DEV)[codes.long()] * sf.float().repeat_interleave(SV, 1)
    return packed, sf, value


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    situ = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    return situ * (SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA))


class _Layer:
    """Random experts, a routing and an input, with the global scales calibrated on them."""

    def __init__(self, num_tokens: int, seed: int) -> None:
        gen = torch.Generator(device=DEV).manual_seed(seed)
        self.experts = []
        for _ in range(NUM_EXPERTS):
            w1 = _rand_nvfp4(INTER, HIDDEN, gen)  # gate
            w3 = _rand_nvfp4(INTER, HIDDEN, gen)  # up
            w2 = _rand_nvfp4(HIDDEN, INTER, gen)
            self.experts.append((w1, w3, w2))
        self.x = (torch.randn(num_tokens, HIDDEN, device=DEV, generator=gen) * 0.5).to(
            torch.bfloat16
        )
        scores = torch.rand(num_tokens, NUM_EXPERTS, device=DEV, generator=gen)
        self.ids = scores.topk(TOP_K, dim=1).indices.to(torch.int32)
        self.weights = torch.rand(num_tokens, TOP_K, device=DEV, generator=gen)
        self.g1 = FP4_RANGE / self.x.float().abs().max().item()
        amax = 0.0
        with torch.no_grad():
            for e, (w1, w3, _) in enumerate(self.experts):
                tok = (self.ids == e).any(dim=1).nonzero().flatten()
                if tok.numel():
                    xe = self.x.float()[tok]
                    act = _situ(xe @ w1[2].t(), xe @ w3[2].t())
                    amax = max(amax, act.abs().max().item())
        self.g2 = FP4_RANGE / amax

    def load(self, module: torch.nn.Module) -> None:
        b = module.backend
        b.prepare_streaming_load()
        one = torch.ones((), dtype=torch.float32, device=DEV)
        for slot, e in enumerate(range(b.slot_start, b.slot_end)):
            w1, w3, w2 = self.experts[e]
            b.load_expert(
                global_expert_id=e,
                local_slot_id=slot,
                w1_weight=w1[0],
                w1_weight_scale=w1[1],
                w1_weight_scale_2=one,
                w1_input_scale=one / self.g1,
                w2_weight=w2[0],
                w2_weight_scale=w2[1],
                w2_weight_scale_2=one,
                w2_input_scale=one / self.g2,
                w3_weight=w3[0],
                w3_weight_scale=w3[1],
                w3_weight_scale_2=one,
                w3_input_scale=one / self.g1,
            )
        b.finalize()
        torch.cuda.synchronize()

    def route(self, logits, offset, num_local):
        """The fixed top-k of the tokens whose index :func:`_token_logits` wrote in column 0 (the
        module routes each chunk on its slice of the logits)."""
        del offset, num_local
        rows = logits[:, 0].long()
        return kx.SortedRouting(self.ids[rows], self.weights[rows], None)

    def reference(self, module) -> torch.Tensor:
        """FP32 MoE on the module's own NVFP4 input, the FC1 output requantized to NVFP4."""
        x_q, x_sf = module.quantize_input(self.x)
        x_q, x_sf = x_q.view(torch.uint8), x_sf.view(torch.uint8)
        codes = torch.stack([x_q & 0x0F, x_q >> 4], dim=-1).reshape(x_q.shape[0], HIDDEN).long()
        scale = x_sf.view(torch.float8_e4m3fn).float().repeat_interleave(SV, 1)
        x = torch.tensor(E2M1, device=DEV)[codes] * scale / self.g1
        out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=DEV)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for e, (w1, w3, w2) in enumerate(self.experts):
                tok, slot = (self.ids == e).nonzero(as_tuple=True)
                if tok.numel() == 0:
                    continue
                xe = x[tok]
                act = _situ(xe @ w1[2].t(), xe @ w3[2].t())
                act = _q_nvfp4(act, self.g2) / self.g2
                out.index_add_(0, tok, (act @ w2[2].t()) * self.weights[tok, slot].unsqueeze(1))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        return out.to(torch.bfloat16)


def _family(kernels: str = "CUTEDSL", max_chunk_tokens: int = 8192, **fc12) -> torch.nn.Module:
    return kx.K3RoutedExperts(
        kernels=kernels,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        layer_idx=0,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
        aux_stream=torch.cuda.Stream(),
        max_chunk_tokens=max_chunk_tokens,
        **fc12,
    ).to(DEV)


def _token_logits(num_tokens: int) -> torch.Tensor:
    """Router logits that carry each token's index in column 0, for :meth:`_Layer.route`."""
    logits = torch.zeros(num_tokens, NUM_EXPERTS, device=DEV)
    logits[:, 0] = torch.arange(num_tokens, device=DEV)
    return logits


def _forward(module, layer, **kwargs) -> torch.Tensor:
    logits = _token_logits(layer.x.shape[0])
    with torch.inference_mode():
        out = module(layer.x, logits, route=layer.route, **kwargs)
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("num_tokens", [1, 24, 200])
def test_cutedsl_against_fp32_reference(num_tokens):
    """The kernel and the reference read the same NVFP4 operands; they differ in accumulation
    order, the kernel's SiTU approximations and the FC1 outputs whose E2M1 code that flips. The
    shared FP4 suite measures that at up to ~16 bf16 ulp of the row's largest per element and
    ~1 ulp relative RMS (``_fp4_moe_test_utils._assert_moe_close``); the gates are 32 / 4."""
    layer = _Layer(num_tokens, seed=num_tokens)
    module = _family()
    layer.load(module)
    got = _forward(module, layer)
    want = layer.reference(module)
    assert_within_ulp(got, want, 32, 4)


#: The finalize GEMM scatter-adds each token's ``TOP_K`` weighted expert rows into the BF16
#: output with atomics, in an order that varies run to run; each add rounds at the scale of its
#: partial sum. Two runs of one input differ by that alone: on GB300, five runs of 24 / 200 tokens
#: (top-8) differ in a third to a half of the elements, by up to 3.5 bf16 ulp of the row's largest
#: and 0.97 ulp in relative RMS. Gates: 8 element / 2 RMS.
ORDER_ULP, ORDER_RMS_ULP = 8, 2


def test_cutedsl_joins_input_stream_and_chunks():
    layer = _Layer(96, seed=7)
    module = _family()
    layer.load(module)
    plain = _forward(module, layer)

    # x written on a side stream after a delay; the forward routes, then joins it.
    side = torch.cuda.Stream()
    ready = torch.cuda.Event()
    x_src = layer.x.clone()
    layer.x = torch.empty_like(x_src)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        torch.cuda._sleep(50_000_000)
        layer.x.copy_(x_src)
        ready.record()
    late = _forward(module, layer, input_ready_event=ready)
    assert_within_ulp(late, plain, ORDER_ULP, ORDER_RMS_ULP)

    chunked = _family(max_chunk_tokens=40)
    layer.load(chunked)
    out = _forward(chunked, layer)
    assert out.shape == plain.shape
    assert_within_ulp(out, plain, ORDER_ULP, ORDER_RMS_ULP)


def test_construction_refuses_fc12_options_on_cutedsl():
    for kwargs in ({"sm_reserve": 16}, {"die_major": True}, {"die_sched_min_tokens": 24}):
        with pytest.raises(ValueError):
            _family(**kwargs)
    with pytest.raises(ValueError):
        _family(kernels="TRTLLM")


@needs_fc12
@pytest.mark.parametrize("num_tokens", [1, 24, 64])
def test_fc12_prezeroed_output(num_tokens):
    layer = _Layer(num_tokens, seed=300 + num_tokens)
    module = _family("CUTEDSL_FC12", sm_reserve=16)
    layer.load(module)
    plain = _forward(module, layer)
    assert_within_ulp(plain, layer.reference(module), 32, 4)

    with torch.inference_mode():
        zeroed = module.new_zeroed_output(num_tokens, torch.device(DEV))
        zeroed.buffer.fill_(-1)  # garbage the side stream must clear
        fork = torch.cuda.Event()
        fork.record()
        kx.zero_on_side_stream(zeroed, fork)
        logits = _token_logits(num_tokens)
        out = module(layer.x, logits, route=layer.route, zeroed_output=zeroed)
    torch.cuda.synchronize()
    assert out.data_ptr() == zeroed.output.data_ptr()
    assert_within_ulp(out, plain, ORDER_ULP, ORDER_RMS_ULP)


@needs_fc12
@pytest.mark.parametrize("num_tokens", [8, 24, 128])
def test_fc12_die_major(num_tokens):
    layer = _Layer(num_tokens, seed=400 + num_tokens)
    flat = _family("CUTEDSL_FC12", sm_reserve=16)
    layer.load(flat)
    want = _forward(flat, layer)
    module = _family("CUTEDSL_FC12", sm_reserve=16, die_major=True, die_sched_min_tokens=24)
    layer.load(module)
    module.post_load_weights()
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_moe import (
        die_locality,
    )

    if not die_locality.is_die_alternating(module.backend.w3_w1_weight.data):
        pytest.skip("the GPU has no die-alternating memory")
    for name in ("w3_w1_weight", "w2_weight"):
        assert torch.equal(getattr(module.backend, name), getattr(flat.backend, name)), name
    got = _forward(module, layer)
    assert_within_ulp(got, want, ORDER_ULP, ORDER_RMS_ULP)
