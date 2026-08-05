# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ragged CUDA-graph key: fit, publication, and the padding ladder.

Three regression guards for one invariant -- the graph key's token axis must
describe the batch the graph will actually replay over. Each test pins a bug
that shipped: a fit against unreachable pad rows, a key derived by re-summing
the batch instead of reading the fitted value (the ragged-IMA family), and a
capture pass that shaped a batch without publishing its bucket.
"""


class _Runner:
    """Just the fields will_pad_to reads."""

    def __init__(self, supported, cfg_batch_size):
        self.supported_batch_sizes = supported
        self.config = type("C", (), {"batch_size": cfg_batch_size})()


def test_will_pad_to_rejects_row_counts_the_ladder_cannot_reach():
    """The ragged fit must not derive a grid from rows that never appear.

    A run fitted its bucket grid to padded_bs=256 while the batch stayed at 193
    -- the captured ladder held 192 and 256 but not 193 -- so the filled total
    449 matched no captured bucket and every rank lost a graph replay on 16 of
    318 steps. Only reachable with real trimming: without it the total is
    padded_bs * (max_verify_len + 1), a captured bucket by construction.
    """
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import \
        CUDAGraphRunner

    runner = _Runner([1, 2, 4, 8, 128, 192, 256], cfg_batch_size=256)
    will = CUDAGraphRunner.will_pad_to

    assert will(runner, 256, 193), "193 -> 256 is reachable and should be taken"
    assert not will(runner, 193, 193), "193 is not a captured size"
    assert will(runner, 192, 192), "already at a captured size"
    # Padding past the configured batch size is refused by _get_padded_batch,
    # so the fit must not assume it.
    assert not will(_Runner([256, 512], cfg_batch_size=256), 512, 300)
    assert not will(runner, 0, 10)
    assert not will(runner, 256, 0)


def test_graph_key_uses_the_agreed_bucket_not_a_fresh_sum():
    """Every attention-DP rank must key on the same bucket, by construction.

    The key used to re-sum 1 + py_verify_len over generation_requests, which
    walks the batch *after* padding is appended -- a second, independent
    derivation of a value all ranks must match exactly. When the two
    derivations diverged the shape gate dropped every rank out of replay
    (peer_shape_mismatch: 2/265 steps without trimming, 16/318 with it, since
    trimming widens the per-rank window spread). Reading the fitted value
    instead makes agreement structural: it comes from allgathered peer stats
    through rules every rank runs identically.
    """
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import \
        CUDAGraphRunner

    import types

    class _R:
        spec_config = type("S", (), {"enable_ragged_verify": True})()
        agreed_ragged_bucket = None

    def _batch(*verify_lens):
        return types.SimpleNamespace(generation_requests=[
            types.SimpleNamespace(py_verify_len=v) for v in verify_lens
        ])

    r = _R()
    get = CUDAGraphRunner._ragged_verify_bucket
    windowed = _batch(3, 2, 1)

    # Nothing fitted yet -> not ragged, key unchanged from the uniform one.
    assert get(r, windowed) is None

    # A fitted bucket is returned verbatim: the batch's own token total is
    # never re-summed here, which is what makes every rank agree.
    r.agreed_ragged_bucket = 449
    assert get(r, windowed) == 449
    # ... including when that total plainly differs from the fitted value.
    assert get(r, _batch(1, 1)) == 449

    # But a request without a window means the token-major staging chain never
    # ran this step, so a ragged key would replay the PREVIOUS ragged step's
    # row maps -- the ragged IMA. Decline, whatever the fit published.
    assert get(r, _batch(3, None, 1)) is None

    # Cleared between steps so a stale bucket cannot key into the wrong graph.
    r.agreed_ragged_bucket = None
    assert get(r, windowed) is None

    # Config gate still wins: a non-ragged engine keeps the original key shape.
    r.agreed_ragged_bucket = 512
    r.spec_config = type("S", (), {"enable_ragged_verify": False})()
    assert get(r, windowed) is None


def test_capture_publishes_the_bucket_it_shaped_the_batch_to():
    """The other half of the test above, and the regression it let through.

    Once the key reads a fitted value instead of re-summing the batch, whoever
    shapes a batch has to publish the shape. Real steps do both in
    fit_ragged_verify_lens. CUDA-graph capture shapes its warmup batch in
    _set_warmup_ragged_windows and never called the fit, so the bucket stayed
    None while the batch really held `verify_bucket` tokens.

    Both consequences are silent in different ways. The capture-time forward
    ran with metadata describing a uniform batch over ragged tensors and died
    hundreds of frames later in the MoE on `unmatched tensor shape`. Had it
    survived, all three of a batch size's buckets would have been stored under
    one key with no token axis -- colliding with each other, and matching no
    runtime step, so every ragged step would have fallen back to eager without
    an error anywhere.
    """
    import types

    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    runner = types.SimpleNamespace(agreed_ragged_bucket=None)
    requests = [types.SimpleNamespace(py_verify_len=None) for _ in range(4)]
    batch = types.SimpleNamespace(generation_requests=requests)
    engine = types.SimpleNamespace(cuda_graph_runner=runner)

    # bs=4, draft_len=5 -> the t=5 bucket is 4 * 6 = 24 tokens.
    PyTorchModelEngine._set_warmup_ragged_windows(engine, batch, 24, 5)

    assert sum(1 + r.py_verify_len for r in requests) == 24, (
        "the warmup batch must hit the bucket exactly")
    assert runner.agreed_ragged_bucket == 24, (
        "capture shaped the batch to 24 tokens but did not publish it, so the "
        "graph would be keyed as if the batch were uniform")

    # A narrower bucket at the same batch size, to prove it is per-entry and
    # not set once: capture walks every tier for every batch size.
    PyTorchModelEngine._set_warmup_ragged_windows(engine, batch, 8, 5)
    assert sum(1 + r.py_verify_len for r in requests) == 8
    assert runner.agreed_ragged_bucket == 8

    # An empty batch publishes nothing rather than a stale or zero bucket.
    runner.agreed_ragged_bucket = None
    PyTorchModelEngine._set_warmup_ragged_windows(
        engine, types.SimpleNamespace(generation_requests=[]), 24, 5)
    assert runner.agreed_ragged_bucket is None
