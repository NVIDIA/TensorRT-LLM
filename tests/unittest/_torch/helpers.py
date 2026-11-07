# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDAGraphRunner, CUDAGraphRunnerConfig)
from tensorrt_llm.mapping import Mapping


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(
        torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax /
                                                             448.0).view(
                                                                 x_view.size(0),
                                                                 x_view.size(2))


def per_token_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n), sf


def per_block_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 or x.dim() == 3
    squeezed = x.dim() == 2
    if squeezed:
        x = x.unsqueeze(0)

    g, m, n = x.shape
    x_padded = torch.zeros((g, align(m, 128), align(n, 128)),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:, :m, :n] = x
    x_view = x_padded.view(g, -1, 128, x_padded.size(-1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(2, 4), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_scaled = x_scaled.view_as(x_padded)[:, :m, :n].contiguous()
    sf = sf.view(x_view.size(0), x_view.size(1), x_view.size(3))

    if squeezed:
        x_scaled = x_scaled.squeeze(0)
        sf = sf.squeeze(0)

    return x_scaled, sf


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def calc_woq_tolerence(x: torch.Tensor, weight_dtype: torch.dtype):
    # align with woq_assert_near_eq function in tests/unittest/trt/quantization/_utils.py
    if weight_dtype == torch.int8:
        bits_in_type = 8
    elif weight_dtype == torch.quint4x2:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))
    max_val = torch.max(abs(x)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding

    return atol


def reference_moe_torch(x: torch.Tensor,
                        selected_experts: torch.Tensor,
                        final_scales: torch.Tensor,
                        num_experts: int,
                        weights: Dict[str, torch.Tensor],
                        apply_routing_on_input: bool = False) -> torch.Tensor:
    # cast back to the input dtype
    results = torch.zeros_like(x)

    # naive looping over experts
    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        w1_weight = weights[f"{expert_id}.w1.weight"]
        w2_weight = weights[f"{expert_id}.w2.weight"]
        w3_weight = weights[f"{expert_id}.w3.weight"]
        expert_inputs = x[batch_idx]
        if apply_routing_on_input:
            expert_inputs = expert_inputs * final_scales[batch_idx, nth_expert,
                                                         None]
        output = (F.silu(expert_inputs @ w1_weight.t()) *
                  (expert_inputs @ w3_weight.t())) @ w2_weight.t()
        if not apply_routing_on_input:
            output = output * final_scales[batch_idx, nth_expert, None]
        results[batch_idx] += output

    return results.view_as(x)


def reference_bmm_moe_torch(
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        final_scales: torch.Tensor,
        w3_w1_stacked_weight: torch.Tensor,
        w2_stacked_weight: torch.Tensor,
        apply_routing_on_input: bool = True) -> torch.Tensor:
    """Reference for stacked MoE in TRT-LLM format.

    Args:
        x: (seq_len, hidden_size)
        selected_experts: (seq_len, topk)
        final_scales: (seq_len, topk)
        w3_w1_stacked_weight: (num_experts, 2*intermediate_size, hidden_size) - TRT-LLM format
        w2_stacked_weight: (num_experts, hidden_size, intermediate_size) - TRT-LLM format
    """
    num_experts = w3_w1_stacked_weight.shape[0]
    intermediate_size = w3_w1_stacked_weight.shape[1] // 2
    results = torch.zeros_like(x)

    # Loop over experts (matches reference_moe_torch pattern)
    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        if len(batch_idx) == 0:
            continue

        # Get weights for this expert (TRT-LLM format)
        gate_up = w3_w1_stacked_weight[expert_id]  # (2*I, H)
        w3_weight = gate_up[:intermediate_size, :]  # (I, H)
        w1_weight = gate_up[intermediate_size:, :]  # (I, H)
        w2_weight = w2_stacked_weight[expert_id]  # (H, I)

        expert_inputs = x[batch_idx]
        if apply_routing_on_input:
            expert_inputs = expert_inputs * final_scales[batch_idx, nth_expert,
                                                         None]

        # Gated MLP computation - TRT-LLM format uses .t()
        output = (F.silu(expert_inputs @ w1_weight.t()) *
                  (expert_inputs @ w3_weight.t())) @ w2_weight.t()

        if not apply_routing_on_input:
            output = output * final_scales[batch_idx, nth_expert, None]

        results[batch_idx] += output

    return results.view_as(x)


def reference_block_scale_moe_torch(
        x: torch.Tensor, selected_experts: torch.Tensor,
        final_scales: torch.Tensor, num_experts: int,
        weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    results = torch.zeros_like(x)

    # naive looping over experts
    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        w1 = weights[f"{expert_id}.w1.weight"]
        w2 = weights[f"{expert_id}.w2.weight"]
        w3 = weights[f"{expert_id}.w3.weight"]

        w1_fp8, w1_scale = per_block_cast_to_fp8(w1)
        w2_fp8, w2_scale = per_block_cast_to_fp8(w2)
        w3_fp8, w3_scale = per_block_cast_to_fp8(w3)

        x_fp8, x_scale = per_token_cast_to_fp8(x[batch_idx])

        def block_scale_gemm(mat_a: torch.Tensor, mat_scale_a: torch.Tensor,
                             mat_b: torch.Tensor, mat_scale_b: torch.Tensor):
            shape_m, shape_k = mat_a.shape
            shape_n = mat_b.shape[0]
            result = torch.zeros((shape_m, shape_n), dtype=torch.float32).cuda()

            for m in range(shape_m):
                for n in range(shape_n):
                    for k in range(0, shape_k, 128):
                        scale_factor = mat_scale_a[m, k //
                                                   128] * mat_scale_b[n // 128,
                                                                      k // 128]
                        tile_a = mat_a[m, k:k + 128]
                        tile_b = mat_b[n, k:k + 128]
                        tile_d = torch.dot(tile_a.float(), tile_b.float())
                        result[
                            m,
                            n] += scale_factor.cuda() * tile_d.cuda().float()

            result_bf16 = result.bfloat16()

            return result_bf16

        # gemm1
        fc3_output = block_scale_gemm(x_fp8, x_scale, w1_fp8, w1_scale)
        gate_output = F.silu(fc3_output)
        fc1_output = block_scale_gemm(x_fp8, x_scale, w3_fp8, w3_scale)
        act_output = gate_output * fc1_output
        # gemm2
        act_fp8, act_scale = per_token_cast_to_fp8(act_output)
        output = block_scale_gemm(act_fp8, act_scale, w2_fp8, w2_scale)

        results[batch_idx] += final_scales[batch_idx, nth_expert, None] * output

    return results.view_as(x)


def create_mock_cuda_graph_runner(
        batch_size: int,
        use_mrope: bool = False,
        max_num_tokens: int = 1,
        static_input_ids: Optional[torch.Tensor] = None,
        static_position_ids: Optional[torch.Tensor] = None):
    config = CUDAGraphRunnerConfig(use_cuda_graph=True,
                                   cuda_graph_padding_enabled=False,
                                   cuda_graph_batch_sizes=[batch_size],
                                   max_cuda_graph_batch_size=batch_size,
                                   batch_size=batch_size,
                                   max_beam_width=1,
                                   max_num_tokens=max_num_tokens,
                                   use_mrope=use_mrope,
                                   spec_config=None,
                                   enable_attention_dp=False,
                                   original_max_total_draft_tokens=0,
                                   is_encoder_decoder=False,
                                   mapping=Mapping(),
                                   dist=None,
                                   static_input_ids=static_input_ids,
                                   static_position_ids=static_position_ids)
    return CUDAGraphRunner(config)


def make_hf_hybrid_cache_for_tests(
    config,
    *,
    max_cache_len: int,
    max_batch_size: Optional[int] = None,
    device=None,
    dtype=None,
):
    """Build Hugging Face ``past_key_values`` for hybrid / sliding-window models in tests.

    Transformers v5 removed ``HybridCache`` in favor of ``StaticCache`` for fixed-length
    pre-allocated KV.
    """
    del max_batch_size, device, dtype  # StaticCache doesn't accept these kwargs.
    from transformers.cache_utils import StaticCache

    return StaticCache(config=config, max_cache_len=max_cache_len)


def dspark_markov_chain_logits_reference(base_logits: torch.Tensor,
                                         first_prev_tokens: torch.Tensor,
                                         markov_w1: torch.Tensor,
                                         markov_w2: torch.Tensor,
                                         argmax_fn=None) -> torch.Tensor:
    """Eager DSpark vanilla-Markov chain on raw tensors; the biased logits only.

    The reference for the fused chain (``vocab_parallel.markov_chain_logits``).
    ``markov_w2`` may be sliced to one TP rank's vocab shard; ``argmax_fn``
    then maps a shard-local row to its full-vocab token id.
    """
    from tensorrt_llm._torch.models.modeling_speculative import (
        dspark_markov_chain, dspark_markov_step_bias)

    def _step_bias(prev_tokens, step_hidden):
        del step_hidden
        return dspark_markov_step_bias(prev_tokens, markov_w1, markov_w2)

    _, corrected = dspark_markov_chain(base_logits,
                                       first_prev_tokens,
                                       _step_bias,
                                       next_token_fn=argmax_fn,
                                       cast_bias_to_logits=True)
    return corrected


@triton.jit
def _gumbel_scores_kernel(logits_ptr, temp_ptr, seed_ptr, out_ptr, vocab,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = cols < vocab
    x = tl.load(logits_ptr + row * vocab + cols, mask=ok,
                other=0.0).to(tl.float32)
    u = tl.rand(tl.load(seed_ptr), row * vocab + cols)
    u = tl.minimum(tl.maximum(u, 1e-12), 0.99999994)
    x = x / tl.load(temp_ptr + row) - tl.log(-tl.log(u))
    tl.store(out_ptr + row * vocab + cols, x, mask=ok)


def gumbel_max_reference(logits: torch.Tensor, temperatures: torch.Tensor,
                         seed: torch.Tensor,
                         greedy_threshold: float) -> torch.Tensor:
    """Temperature samples (int32 [rows]) of ``logits`` [rows, vocab] by
    Gumbel-max with the noise ``TPKeyExchange.argmax_logits`` draws:
    ``tl.rand(seed, row * vocab + token)`` for ``seed`` (int64 [1]). Rows below
    ``greedy_threshold`` take the argmax."""
    rows, vocab = logits.shape
    scores = torch.empty((rows, vocab),
                         dtype=torch.float32,
                         device=logits.device)
    block = 1024
    grid = (rows, triton.cdiv(vocab, block))
    _gumbel_scores_kernel[grid](logits.contiguous(),
                                temperatures,
                                seed,
                                scores,
                                vocab,
                                BLOCK=block)
    sampled = temperatures >= greedy_threshold
    return torch.where(sampled, scores.argmax(-1),
                       logits.float().argmax(-1)).to(torch.int32)


def mpi_worker_env(**env: str):
    """Decorate a function run in a pooled MPI worker to see ``env`` set, restoring the
    previous values after it: the pool reuses its processes for later tests."""

    def decorate(fn):

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            saved = {name: os.environ.get(name) for name in env}
            os.environ.update(env)
            try:
                return fn(*args, **kwargs)
            finally:
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

        return wrapped

    return decorate


def bf16_bits_differ(ref: torch.Tensor, out: torch.Tensor) -> int:
    """Number of 16-bit elements whose bits differ."""
    return int((ref.view(torch.int16) != out.view(torch.int16)).sum())


def r128c4_scale_offsets(m: int, h: int, device) -> torch.Tensor:
    """``[m, h // 32]`` byte offsets of the first ``m`` rows' UE8M0 1x32 scales of ``[*, h]``
    rows in the R128c4 layout (128-row blocks of 4-column groups)."""
    r = torch.arange(m, device=device).unsqueeze(1)
    c = torch.arange(h // 32, device=device).unsqueeze(0)
    return (((r // 128) * (h // 128) + c // 4) * 512 + (r % 32) * 16 +
            (r % 128) // 32 * 4 + c % 4)


def mnnvl_torch_push(workspace, target, x: torch.Tensor, rank: int) -> None:
    """Producer stand-in for a pushed MNNVL all-reduce: store ``x`` into the Lamport slots of
    ``target`` with plain copies (``-0`` sent as ``+0``), through every rank's unicast
    mapping."""
    torch.cuda.synchronize()  # the previous call's rotation state is final
    flags = target.view.flags.cpu().tolist()
    cur, nbytes = int(flags[0]), int(flags[2])
    world = target.view.world
    m, h = x.shape
    bits = x.view(torch.int16)
    clean = torch.where(bits == -32768, torch.zeros_like(bits),
                        bits).view(torch.bfloat16)
    handle = workspace["handle"]
    elems = 3 * nbytes // 2
    for peer in range(world):
        buf = handle.get_uc_buffer(peer, (elems, ), torch.bfloat16, 0)
        base = cur * nbytes // 2
        if target.one_shot:
            slots = buf[base:base + m * world * h].view(m, world, h)
            slots[:, rank].copy_(clean)
        else:
            for t in range(peer, m, world):
                slot = (t // world) * world + rank
                buf[base + slot * h:base + (slot + 1) * h].copy_(clean[t])
    torch.cuda.synchronize()


SMALL_BATCH_MOE_SITU_BETA = 2.5
SMALL_BATCH_MOE_SITU_LINEAR_BETA = 0.7


def random_nvfp4_experts(
        num: int, rows: int, cols: int,
        gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random e2m1 weights ``[num, rows, cols / 2]`` and E4M3 block scales in
    [0.5, 2) (128x4 layout)."""
    w = torch.randint(0,
                      256, (num, rows, cols // 2),
                      dtype=torch.uint8,
                      device="cuda",
                      generator=gen)
    sf = torch.rand(
        (num, rows, cols // 16), device="cuda", generator=gen) * 1.5 + 0.5
    return w, sf.to(torch.float8_e4m3fn).view(torch.uint8)


def small_batch_moe_operands(
        ids: torch.Tensor,
        scales: torch.Tensor,
        hidden: int,
        inter_size: int,
        num_experts: int,
        num_local: int,
        offset: int,
        gen: torch.Generator,
        weight_gen: Optional[torch.Generator] = None) -> Dict[str, object]:
    """Operands of ``trtllm::cute_dsl_nvfp4_small_batch_moe_rubin`` on a rank
    whose ``num_local`` experts start at ``offset``, for the top-k ``ids`` /
    ``scales`` ``[num_tokens, top_k]``: the ``moe_sort`` tables, NVFP4 tokens
    drawn from ``gen`` and NVFP4 experts drawn from ``weight_gen`` (default
    ``gen``)."""
    num_tokens, top_k = ids.shape
    t2g, t2lim, e2p, p2e, _, nnet = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=offset,
        local_num_experts=num_local,
        tile_tokens_dim=128,
    )
    xb = torch.randn(num_tokens,
                     hidden,
                     device="cuda",
                     dtype=torch.bfloat16,
                     generator=gen)
    x_gsf = xb.abs().max().float() / (448 * 6)
    x, x_sf = torch.ops.trtllm.fp4_quantize(xb, 1 / x_gsf, 16, False, False)
    weight_gen = gen if weight_gen is None else weight_gen
    w1, w1_sf = random_nvfp4_experts(num_local, 2 * inter_size, hidden,
                                     weight_gen)
    w2, w2_sf = random_nvfp4_experts(num_local, hidden, inter_size, weight_gen)
    ramp = torch.linspace(0.5, 1.5, num_local, device="cuda")
    global_sf = torch.tensor([448 * 6 / 64.0],
                             dtype=torch.float32,
                             device="cuda")
    return dict(
        hidden=hidden,
        inter_size=inter_size,
        num_local=num_local,
        top_k=top_k,
        offset=offset,
        ids=ids,
        scales=scales,
        t2g=t2g,
        t2lim=t2lim,
        p2e=p2e,
        e2p=e2p,
        nnet=nnet,
        x=x,
        x_sf=x_sf.view(num_tokens, hidden // 16),
        w1=w1,
        w1_sf=w1_sf,
        w2=w2,
        w2_sf=w2_sf,
        alpha1=(ramp * x_gsf * 0.004).float(),
        alpha2=(ramp * 0.003 / global_sf[0]).float(),
        global_sf=global_sf,
    )


def run_small_batch_moe(d: Dict[str, object],
                        out: torch.Tensor,
                        counters: torch.Tensor,
                        *,
                        token_indexed: bool,
                        deterministic: bool,
                        num_ctas: int = 0,
                        die_major: bool = False,
                        published: bool = False,
                        push=None) -> None:
    """``trtllm::cute_dsl_nvfp4_small_batch_moe_rubin`` on the operands ``d``
    of :func:`small_batch_moe_operands`: on their ``moe_sort`` tables or, with
    ``published``, on their top-k (the routing word holds the token count
    before the launch); ``push``: the ``LamportWorkspaceView`` of a one-shot
    MNNVL all-reduce call the kernel pushes its output into."""
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_small_batch_moe_custom_op import \
        small_batch_moe_route_word

    tables = (d["t2g"], d["t2lim"], d["p2e"], d["nnet"])
    e2p = d["e2p"]
    if published:
        small_batch_moe_route_word(counters).fill_(d["ids"].size(0))
        tables, e2p = (None, ) * 4, d["ids"].reshape(-1)
    torch.ops.trtllm.cute_dsl_nvfp4_small_batch_moe_rubin(
        input=d["x"].view(torch.float4_e2m1fn_x2),
        input_scale=d["x_sf"].view(torch.uint8),
        fc1_weight=d["w1"].view(torch.float4_e2m1fn_x2),
        fc1_weight_scale=d["w1_sf"],
        fc1_alpha=d["alpha1"],
        fc2_weight=d["w2"].view(torch.float4_e2m1fn_x2),
        fc2_weight_scale=d["w2_sf"],
        fc2_alpha=d["alpha2"],
        output=out,
        tile_idx_to_group_idx=tables[0],
        tile_idx_to_mn_limit=tables[1],
        permuted_idx_to_expanded_idx=tables[2],
        expanded_idx_to_permuted_idx=e2p,
        num_non_exiting_tiles=tables[3],
        global_sf=d["global_sf"],
        token_final_scales=d["scales"],
        counters=counters,
        top_k=d["top_k"],
        situ_beta=SMALL_BATCH_MOE_SITU_BETA,
        situ_linear_beta=SMALL_BATCH_MOE_SITU_LINEAR_BETA,
        token_indexed=token_indexed,
        num_ctas=num_ctas,
        deterministic_combine=deterministic,
        die_major=die_major,
        published_routing=published,
        expert_offset=d["offset"],
        push_mc=push.mc_base if push is not None else 0,
        push_flags=push.flags if push is not None else None,
        push_rank=push.rank if push is not None else 0,
        push_world=push.world if push is not None else 0,
    )


def dsv3_router_logits(x: torch.Tensor,
                       router: torch.Tensor,
                       tile_rows: int = 0) -> torch.Tensor:
    """FP32 router logits ``x @ router.T`` from ``trtllm::dsv3_router_gemm_op``,
    one call per ``tile_rows`` rows of ``x`` where given (0: one call)."""
    tile_rows = tile_rows or x.shape[0]
    return torch.cat([
        torch.ops.trtllm.dsv3_router_gemm_op(x[r:r + tile_rows].contiguous(),
                                             router.t(),
                                             bias=None,
                                             out_dtype=torch.float32)
        for r in range(0, x.shape[0], tile_rows)
    ])
