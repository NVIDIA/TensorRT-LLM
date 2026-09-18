"""GPU checks for the folded save-last prefill building blocks.

The folded schedule must reproduce today's two-chunk schedule exactly:
segment A [pos, reachable) runs in place on the snapshot slot S1, the tail
B [reachable, prompt_len) runs from S1 into the terminal slot S2, and the conv
state left in S1 is the ``d_conv - 1`` raw inputs before the fold point.
"""

import pytest
import torch


def _supported_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major in (9, 10)


skip_unsupported = pytest.mark.skipif(
    not _supported_arch(),
    reason="FlashInfer GDN prefill requires SM90 (Hopper) or SM100 (Blackwell)",
)


@torch.no_grad()
def _inputs(total_t, num_q_heads=4, num_v_heads=16, head_dim=128, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(1, total_t, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    k = torch.randn(1, total_t, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    v = torch.randn(1, total_t, num_v_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    g = -torch.rand(1, total_t, num_v_heads, dtype=torch.float32, device=dev) * 0.05
    beta = torch.rand(1, total_t, num_v_heads, dtype=torch.float32, device=dev)
    return q, k, v, g, beta


@skip_unsupported
@pytest.mark.parametrize("split", [64, 70, 32])
@torch.no_grad()
def test_fold_tail_launch_matches_two_chunk_schedule(split):
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule

    total = 87
    q, k, v, g, beta = _inputs(total)
    num_v_heads, head_dim = v.shape[2], v.shape[3]
    dev = q.device
    S1, S2 = 1, 3
    init_pool = torch.randn(4, num_v_heads, head_dim, head_dim, dtype=torch.bfloat16, device=dev) * 0.01

    def run_chunk(rows, indices, out_indices=None, cu=None, pool=None):
        sl = slice(rows[0], rows[1])
        cu_t = torch.tensor([0, rows[1] - rows[0]] if cu is None else cu, dtype=torch.long, device=dev)
        out, _ = chunk_gated_delta_rule(
            q=q[:, sl], k=k[:, sl], v=v[:, sl], g=g[:, sl], beta=beta[:, sl],
            initial_state=pool,
            initial_state_indices=torch.tensor(indices, dtype=torch.int32, device=dev),
            inplace_indexed_state_update=True,
            output_final_state=False,
            cu_seqlens=cu_t,
            output_state_indices=None if out_indices is None else torch.tensor(out_indices, dtype=torch.int32, device=dev),
        )
        return out

    # --- today's schedule: chunk 1 on S1, copy S1 -> S2, chunk 2 on S2 ------
    pool_ref = init_pool.clone()
    out_a = run_chunk((0, split), [S1], pool=pool_ref)
    pool_ref[S2] = pool_ref[S1]
    out_b = run_chunk((split, total), [S2], pool=pool_ref)
    out_ref = torch.cat([out_a, out_b], dim=1)

    # --- folded schedule: one launch with A -> S1 and a discarded B -> S2, then the tail S1 -> S2
    pool_fold = init_pool.clone()
    out_fold = run_chunk((0, total), [S1, S2], cu=[0, split, total], pool=pool_fold)
    tail = run_chunk((split, total), [S1], out_indices=[S2], pool=pool_fold)
    out_fold[:, split:total] = tail

    assert torch.equal(pool_fold[S1], pool_ref[S1]), "snapshot slot S1 differs"
    assert torch.equal(pool_fold[S2], pool_ref[S2]), "terminal slot S2 differs"
    assert torch.equal(out_fold, out_ref), "tail outputs differ"
    # untouched slots stay untouched
    assert torch.equal(pool_fold[0], init_pool[0]) and torch.equal(pool_fold[2], init_pool[2])

    # sanity: the two-launch result is close to the single-segment run
    pool_one = init_pool.clone()
    out_one = run_chunk((0, total), [S1], pool=pool_one)
    torch.testing.assert_close(out_fold.float(), out_one.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(pool_fold[S2].float(), pool_one[S1].float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.no_grad()
def test_conv_state_at_fold_point_is_the_raw_tail():
    """The causal conv kernel leaves the last ``width - 1`` raw inputs of a
    sequence in the conv-state slot; the fold reproduces that for the fold
    point by slicing the pre-conv input."""
    from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn

    dev = "cuda"
    dim, width, total, split = 64, 4, 87, 70
    torch.manual_seed(1)
    x = torch.randn(dim, total, dtype=torch.bfloat16, device=dev)
    weight = torch.randn(dim, width, dtype=torch.bfloat16, device=dev) * 0.1
    bias = torch.randn(dim, dtype=torch.bfloat16, device=dev) * 0.1
    conv_states = torch.zeros(4, dim, width - 1, dtype=torch.bfloat16, device=dev)

    raw_tail = x[:, split - (width - 1):split].clone()   # what _fold_conv_tail captures
    x_a = x[:, :split].clone()
    causal_conv1d_fn(
        x_a, weight, bias, activation="silu", conv_states=conv_states,
        has_initial_state=torch.tensor([False], device=dev),
        cache_indices=torch.tensor([1], dtype=torch.int32, device=dev),
        query_start_loc=torch.tensor([0, split], dtype=torch.int32, device=dev),
    )
    assert torch.equal(conv_states[1], raw_tail)

    # and the state after the whole sequence is the raw tail of the whole sequence,
    # independent of the fold (this is what lands in S2)
    x_full = x.clone()
    causal_conv1d_fn(
        x_full, weight, bias, activation="silu", conv_states=conv_states,
        has_initial_state=torch.tensor([False], device=dev),
        cache_indices=torch.tensor([2], dtype=torch.int32, device=dev),
        query_start_loc=torch.tensor([0, total], dtype=torch.int32, device=dev),
    )
    assert torch.equal(conv_states[2], x[:, total - (width - 1):total])
