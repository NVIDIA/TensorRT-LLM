# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the noaux_tc_op catalog entry."""

import torch

from tensorrt_llm._torch.staircase.catalog.moe.noaux_tc_op import noaux_tc_op

assert torch.cuda.is_available(), "noaux_tc_op requires a CUDA device"


def _kernel_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """The sigmoid the kernel actually evaluates: 0.5 * tanh(x/2) + 0.5, fp32.

    Algebraically identical to 1/(1+exp(-x)) but numerically different in the
    tails: it saturates to exactly 1.0 at x >= ~17 and to exactly 0.0 at
    x <= ~-18.5. Pinned by test_sigmoid_is_the_tanh_form.
    """
    return 0.5 * torch.tanh(0.5 * x.float()) + 0.5


def _ref(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
    sigmoid=_kernel_sigmoid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 reference built from native torch ops only.

    A *stable* descending sort keeps equal selection scores in ascending expert
    order, which is the kernel's observed tie-break.
    """
    num_experts = router_logits.shape[-1]
    scores = sigmoid(router_logits)
    choice = scores + bias.float()
    if n_group > 1:
        grouped = choice.view(-1, n_group, num_experts // n_group)
        group_score = torch.topk(grouped, k=2, dim=-1).values.sum(-1)
        keep = torch.sort(group_score, dim=-1, descending=True, stable=True).indices[:, :topk_group]
        mask = torch.zeros_like(group_score).scatter_(-1, keep, 1.0)
        mask = mask.unsqueeze(-1).expand_as(grouped).reshape(choice.shape)
        choice = torch.where(mask.bool(), choice, torch.tensor(float("-inf"), device=choice.device))
    ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :topk]
    weights = torch.gather(scores, 1, ids)
    # The scores are summed in fp32, but the division and the scaling are
    # evaluated in fp64 (pinned by test_routed_scaling_factor: an all-fp32
    # normalization disagrees on a third of the elements at
    # routed_scaling_factor = 2.5, an fp64 one is bit-exact).
    total = weights.sum(-1, keepdim=True).double()
    weights = weights.double() / (total + 1e-20) * routed_scaling_factor
    return weights.to(router_logits.dtype), ids.to(torch.int32)


def _check(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights, ids = noaux_tc_op(
        router_logits, bias, n_group, topk_group, topk, routed_scaling_factor
    )
    ref_weights, ref_ids = _ref(
        router_logits, bias, n_group, topk_group, topk, routed_scaling_factor
    )
    num_tokens = router_logits.shape[0]
    assert weights.shape == (num_tokens, topk), weights.shape
    assert ids.shape == (num_tokens, topk), ids.shape
    assert weights.dtype == router_logits.dtype, weights.dtype
    assert ids.dtype == torch.int32, ids.dtype
    assert weights.is_contiguous() and ids.is_contiguous()
    assert weights.device == router_logits.device
    assert ids.device == router_logits.device
    torch.testing.assert_close(ids, ref_ids)
    # Dual gate. The fp32 reference is not an approximation of this kernel, it
    # is the kernel — the only observed disagreement is the final rounding of
    # the renormalized weight, which can step by one last-bit unit because the
    # kernel adds up the topk scores in a different order than torch.sum
    # (measured: 1 element in 49152 at 8192x72 bf16). So the tolerance is
    # exactly one ulp (rtol = the dtype's eps is the largest relative size one
    # ulp can have; atol = 0), and on top of that almost every element must be
    # bit-identical.
    torch.testing.assert_close(
        weights,
        ref_weights,
        rtol=torch.finfo(router_logits.dtype).eps,
        atol=0.0,
    )
    mismatched = int((weights != ref_weights).sum())
    budget = 1 + weights.numel() // 10000
    assert mismatched <= budget, (
        f"{mismatched} of {weights.numel()} weights are not bit-exact (budget "
        f"{budget}): dtype={router_logits.dtype} shape={tuple(router_logits.shape)} "
        f"n_group={n_group} topk={topk}"
    )
    return weights, ids


def test_deepseek_v3_lite_config() -> None:
    # 72 routed experts, top-6, n_group=1, topk_group=1, scaling 2.0, bf16
    # router logits and a bf16 [72] correction bias. Decode-like through
    # prefill-like token counts.
    torch.manual_seed(0)
    bias = torch.randn(72, dtype=torch.bfloat16, device="cuda") * 0.1
    for num_tokens in [1, 2, 7, 64, 512, 2048, 4096, 8192]:
        logits = torch.randn(num_tokens, 72, dtype=torch.bfloat16, device="cuda")
        weights, _ = _check(logits, bias, 1, 1, 6, 2.0)
        # renormalized then scaled: every row sums to routed_scaling_factor
        torch.testing.assert_close(
            weights.float().sum(-1),
            torch.full((num_tokens,), 2.0, device="cuda"),
            rtol=8e-3,  # 6 bf16 addends, each rounded to 8 mantissa bits
            atol=0.0,
        )


def test_matches_hf_deepseek_sigmoid_reference() -> None:
    # The HF DeepSeek-V3 gate written with the textbook sigmoid. Over the
    # logit range a real router produces (|logit| <= 8) the kernel's tanh-form
    # sigmoid and torch.sigmoid agree to well inside dtype tolerance.
    torch.manual_seed(1)
    for dtype in [torch.bfloat16, torch.float32]:
        for num_tokens in [1, 256, 2048]:
            logits = (torch.randn(num_tokens, 72, device="cuda") * 2.5).to(dtype)
            bias = (torch.randn(72, device="cuda") * 0.1).to(dtype)
            weights, ids = noaux_tc_op(logits, bias, 1, 1, 6, 2.0)
            ref_weights, ref_ids = _ref(
                logits, bias, 1, 1, 6, 2.0, sigmoid=lambda x: torch.sigmoid(x.float())
            )
            torch.testing.assert_close(ids, ref_ids)
            torch.testing.assert_close(weights, ref_weights)


def test_selection_uses_bias_but_weights_do_not() -> None:
    # The trap this entry exists to pin: the bias enters selection only.
    # A bias large enough to reorder the experts must move the ids, and the
    # returned weights must come from the *unbiased* sigmoid.
    torch.manual_seed(2)
    logits = torch.randn(512, 72, dtype=torch.float32, device="cuda")
    bias = torch.randn(72, dtype=torch.float32, device="cuda")
    weights, ids = noaux_tc_op(logits, bias, 1, 1, 6, 2.0)

    unbiased_ids = torch.sort(
        _kernel_sigmoid(logits), dim=-1, descending=True, stable=True
    ).indices[:, :6]
    assert not torch.equal(ids, unbiased_ids.to(torch.int32)), (
        "the bias did not affect selection — test input too weak"
    )

    scores = _kernel_sigmoid(logits)
    gathered = torch.gather(scores, 1, ids.long())
    expect = gathered / (gathered.sum(-1, keepdim=True) + 1e-20) * 2.0
    torch.testing.assert_close(weights, expect)

    # negative control: gathering the *biased* score instead is a different,
    # plausible-looking answer that the kernel does not produce
    biased = torch.gather(scores + bias, 1, ids.long())
    wrong = biased / (biased.sum(-1, keepdim=True) + 1e-20) * 2.0
    assert (weights - wrong).abs().max().item() > 1e-2, (
        "biased-weight reference is indistinguishable — test input too weak"
    )


def test_slot_order_is_descending_by_biased_score() -> None:
    torch.manual_seed(3)
    logits = torch.randn(256, 72, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(72, dtype=torch.bfloat16, device="cuda")
    _, ids = noaux_tc_op(logits, bias, 1, 1, 6, 2.0)
    choice = _kernel_sigmoid(logits) + bias.float()
    selected = torch.gather(choice, 1, ids.long())
    assert bool((selected[:, :-1] >= selected[:, 1:]).all()), (
        "slots are not ordered by decreasing selection score"
    )
    # the same row is *not* sorted by the unbiased score, i.e. the ordering
    # key really is the biased one
    unbiased = torch.gather(_kernel_sigmoid(logits), 1, ids.long())
    assert not bool((unbiased[:, :-1] >= unbiased[:, 1:]).all())


def test_ties_break_toward_lower_expert_index() -> None:
    # Opposite to torch.topk, which emits the larger index first among equals.
    logits = torch.zeros(4, 16, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros(16, dtype=torch.bfloat16, device="cuda")
    logits[0, :] = 1.0
    logits[1, 3] = 5.0
    logits[1, 7] = 5.0
    logits[2, ::2] = 2.0
    logits[3, 0] = 1.0
    logits[3, 15] = 1.0
    _, ids = noaux_tc_op(logits, bias, 1, 1, 4, 2.0)
    expected = torch.tensor(
        [[0, 1, 2, 3], [3, 7, 0, 1], [0, 2, 4, 6], [0, 15, 1, 2]],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(ids, expected)
    _check(logits, bias, 1, 1, 4, 2.0)

    # tie-saturated inputs: ties from the logits, and ties created by a coarse
    # bias on top of continuous logits
    torch.manual_seed(4)
    for num_tokens, num_experts, topk in [(64, 128, 8), (512, 72, 6), (256, 16, 4)]:
        coarse = torch.randint(0, 3, (num_tokens, num_experts), device="cuda").to(torch.bfloat16)
        zero = torch.zeros(num_experts, dtype=torch.bfloat16, device="cuda")
        _check(coarse, zero, 1, 1, topk, 2.0)
        smooth = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device="cuda")
        coarse_bias = torch.randint(0, 2, (num_experts,), device="cuda").to(torch.bfloat16)
        _check(smooth, coarse_bias, 1, 1, topk, 2.0)


def test_dtype_matrix() -> None:
    # weights come back in the *router_logits* dtype; the bias dtype is
    # independent, except that fp16 logits reject a bf16 bias.
    torch.manual_seed(5)
    base = torch.randn(128, 72, device="cuda")
    bias_base = torch.randn(72, device="cuda") * 0.1
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    for logits_dtype in dtypes:
        for bias_dtype in dtypes:
            if logits_dtype is torch.float16 and bias_dtype is torch.bfloat16:
                continue  # rejected; covered by test_op_rejects_unsupported_domains
            logits = base.to(logits_dtype)
            bias = bias_base.to(bias_dtype)
            weights, _ = _check(logits, bias, 1, 1, 6, 2.0)
            assert weights.dtype == logits_dtype, (
                logits_dtype,
                bias_dtype,
                weights.dtype,
            )


def test_routed_scaling_factor() -> None:
    torch.manual_seed(6)
    logits = torch.randn(256, 72, dtype=torch.float32, device="cuda")
    bias = torch.randn(72, dtype=torch.float32, device="cuda") * 0.1
    for scaling in [1.0, 2.0, 2.5, 3.0, 0.5, 0.0, -1.0, 1000.0]:
        weights, _ = _check(logits, bias, 1, 1, 6, scaling)
        torch.testing.assert_close(weights.sum(-1), torch.full((256,), scaling, device="cuda"))
        # negative control for the fp64 normalization: an all-fp32 reference
        # is a different answer for scaling factors that are not a power of two
        scores = _kernel_sigmoid(logits)
        _, ids = noaux_tc_op(logits, bias, 1, 1, 6, scaling)
        gathered = torch.gather(scores, 1, ids.long())
        fp32_form = gathered / (gathered.sum(-1, keepdim=True) + 1e-20) * scaling
        differs = int((weights != fp32_form).sum())
        if scaling in (2.5, 3.0, 1000.0):
            assert differs > 0.1 * weights.numel(), (
                f"fp32 and fp64 normalization are indistinguishable at {scaling}"
            )


def test_sigmoid_is_the_tanh_form() -> None:
    # Pins the kernel's sigmoid out to |logits| ~ 40, where the tanh form has
    # long saturated. _check's reference is the tanh form; the exp form is
    # ruled out separately in test_saturation_floor_and_ceiling.
    torch.manual_seed(7)
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for scale in [1.0, 4.0, 10.0, 20.0, 40.0]:
            logits = (torch.randn(256, 72, device="cuda") * scale).to(dtype)
            bias = (torch.randn(72, device="cuda") * 0.1).to(dtype)
            weights, _ = _check(logits, bias, 1, 1, 6, 2.0)
            assert bool(torch.isfinite(weights.float()).all())


def test_saturation_floor_and_ceiling() -> None:
    # The tail behaviour that follows from the tanh form.
    bias = torch.zeros(16, dtype=torch.float32, device="cuda")
    high = torch.full((2, 16), 20.0, dtype=torch.float32, device="cuda")
    weights, _ = noaux_tc_op(high, bias, 1, 1, 4, 2.0)
    # every selected score saturates to exactly 1.0 -> equal weights
    assert torch.equal(weights, torch.full_like(weights, 0.5)), weights

    low = torch.full((2, 16), -20.0, dtype=torch.float32, device="cuda")
    weights, _ = noaux_tc_op(low, bias, 1, 1, 4, 2.0)
    # every selected score saturates to exactly 0.0; the guard term in the
    # denominator turns 0/0 into exactly 0 instead of NaN
    assert torch.equal(weights, torch.zeros_like(weights)), weights

    # Negative control that separates the two sigmoid forms: with every
    # selected logit inside the tanh form's lossy band, a 1/(1+exp(-x))
    # reference lands three orders of magnitude outside the one-ulp gate
    # _check applies (fp32 eps * 0.33 ~ 4e-8).
    torch.manual_seed(8)
    band = (torch.rand(128, 72, device="cuda") * 12.0 - 20.0).to(torch.float32)
    zero = torch.zeros(72, dtype=torch.float32, device="cuda")
    weights, _ = _check(band, zero, 1, 1, 6, 2.0)
    exp_weights, _ = _ref(band, zero, 1, 1, 6, 2.0, sigmoid=lambda x: torch.sigmoid(x.float()))
    assert (weights - exp_weights).abs().max().item() > 1e-5, (
        "exp-form and tanh-form references are indistinguishable here"
    )


def test_grouped_routing() -> None:
    # n_group > 1: group score = sum of the two largest biased scores in the
    # group; the topk_group best groups survive, the rest are masked out.
    torch.manual_seed(9)
    for num_experts, n_group, topk_group, topk in [
        (256, 8, 4, 8),  # DeepSeek-V3 full
        (128, 4, 2, 6),
        (72, 8, 2, 6),
        (72, 4, 2, 6),
        (64, 8, 4, 8),
    ]:
        for num_tokens in [1, 64, 1024]:
            logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device="cuda")
            bias = torch.randn(num_experts, dtype=torch.bfloat16, device="cuda") * 0.1
            _check(logits, bias, n_group, topk_group, topk, 2.0)
        # tie-saturated grouping
        coarse = torch.randint(0, 3, (64, num_experts), device="cuda").to(torch.bfloat16)
        zero = torch.zeros(num_experts, dtype=torch.bfloat16, device="cuda")
        _check(coarse, zero, n_group, topk_group, topk, 2.0)

    # keeping every group is the same as not grouping at all
    logits = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(256, dtype=torch.bfloat16, device="cuda") * 0.1
    grouped = noaux_tc_op(logits, bias, 8, 8, 8, 2.0)
    flat = noaux_tc_op(logits, bias, 1, 1, 8, 2.0)
    assert torch.equal(grouped[0], flat[0]) and torch.equal(grouped[1], flat[1])


def test_topk_group_ignored_when_n_group_is_one() -> None:
    torch.manual_seed(10)
    logits = torch.randn(64, 72, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(72, dtype=torch.bfloat16, device="cuda") * 0.1
    reference = noaux_tc_op(logits, bias, 1, 1, 6, 2.0)
    for topk_group in [0, 1, 2, 5, 32, 100]:
        weights, ids = noaux_tc_op(logits, bias, 1, topk_group, 6, 2.0)
        assert torch.equal(weights, reference[0]) and torch.equal(ids, reference[1]), (
            f"topk_group={topk_group} changed the result at n_group=1"
        )


def test_num_experts_and_topk_sweep() -> None:
    torch.manual_seed(11)
    for num_experts in [1, 2, 7, 32, 64, 72, 100, 128, 256, 257, 512, 1024]:
        logits = torch.randn(8, num_experts, dtype=torch.bfloat16, device="cuda")
        bias = torch.randn(num_experts, dtype=torch.bfloat16, device="cuda")
        _check(logits, bias, 1, 1, min(6, num_experts), 2.0)
    logits = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(64, dtype=torch.bfloat16, device="cuda")
    for topk in [1, 2, 3, 6, 8, 16, 31, 32]:
        _check(logits, bias, 1, 1, topk, 2.0)
    # topk == num_experts returns the full ranking
    small = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
    small_bias = torch.randn(8, dtype=torch.bfloat16, device="cuda")
    _check(small, small_bias, 1, 1, 8, 2.0)


def test_degenerate_shapes() -> None:
    logits = torch.randn(8, 72, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(72, dtype=torch.bfloat16, device="cuda")
    weights, ids = noaux_tc_op(logits, bias, 1, 1, 0, 2.0)
    assert weights.shape == (8, 0) and ids.shape == (8, 0)
    empty = torch.randn(0, 72, dtype=torch.bfloat16, device="cuda")
    weights, ids = noaux_tc_op(empty, bias, 1, 1, 6, 2.0)
    assert weights.shape == (0, 6) and ids.shape == (0, 6)
    assert weights.dtype == torch.bfloat16 and ids.dtype == torch.int32


def test_input_not_mutated_and_deterministic() -> None:
    torch.manual_seed(12)
    logits = torch.randn(4096, 72, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(72, dtype=torch.bfloat16, device="cuda")
    logits_snapshot, bias_snapshot = logits.clone(), bias.clone()
    weights_a, ids_a = noaux_tc_op(logits, bias, 1, 1, 6, 2.0)
    weights_b, ids_b = noaux_tc_op(logits, bias, 1, 1, 6, 2.0)
    assert torch.equal(logits, logits_snapshot), "router_logits was mutated"
    assert torch.equal(bias, bias_snapshot), "bias was mutated"
    assert torch.equal(weights_a, weights_b) and torch.equal(ids_a, ids_b), (
        "two identical calls disagreed"
    )
    assert weights_a.data_ptr() != weights_b.data_ptr()
    assert weights_a.data_ptr() != logits.data_ptr()


def test_op_rejects_unsupported_domains() -> None:
    logits = torch.randn(8, 72, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(72, dtype=torch.bfloat16, device="cuda")
    big = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")
    big_bias = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    cases = [
        ("1D router_logits", lambda: noaux_tc_op(logits[0], bias, 1, 1, 6, 2.0)),
        (
            "3D router_logits",
            lambda: noaux_tc_op(logits.unsqueeze(0), bias, 1, 1, 6, 2.0),
        ),
        (
            "fp64 router_logits",
            lambda: noaux_tc_op(logits.double(), bias, 1, 1, 6, 2.0),
        ),
        ("int32 router_logits", lambda: noaux_tc_op(logits.int(), bias, 1, 1, 6, 2.0)),
        ("fp64 bias", lambda: noaux_tc_op(logits, bias.double(), 1, 1, 6, 2.0)),
        ("int32 bias", lambda: noaux_tc_op(logits, bias.int(), 1, 1, 6, 2.0)),
        (
            "fp16 logits + bf16 bias",
            lambda: noaux_tc_op(logits.half(), bias, 1, 1, 6, 2.0),
        ),
        ("2D bias", lambda: noaux_tc_op(logits, bias.unsqueeze(0), 1, 1, 6, 2.0)),
        ("short bias", lambda: noaux_tc_op(logits, bias[:71], 1, 1, 6, 2.0)),
        ("cpu tensors", lambda: noaux_tc_op(logits.cpu(), bias.cpu(), 1, 1, 6, 2.0)),
        ("bias on cpu", lambda: noaux_tc_op(logits, bias.cpu(), 1, 1, 6, 2.0)),
        ("negative topk", lambda: noaux_tc_op(logits, bias, 1, 1, -1, 2.0)),
        ("topk 33", lambda: noaux_tc_op(big, big_bias, 1, 1, 33, 2.0)),
        (
            "num_experts 1025",
            lambda: noaux_tc_op(
                torch.randn(8, 1025, dtype=torch.bfloat16, device="cuda"),
                torch.randn(1025, dtype=torch.bfloat16, device="cuda"),
                1,
                1,
                6,
                2.0,
            ),
        ),
        (
            "n_group does not divide num_experts",
            lambda: noaux_tc_op(logits, bias, 5, 1, 6, 2.0),
        ),
        (
            "n_group 33",
            lambda: noaux_tc_op(
                torch.randn(8, 132, dtype=torch.bfloat16, device="cuda"),
                torch.randn(132, dtype=torch.bfloat16, device="cuda"),
                33,
                1,
                6,
                2.0,
            ),
        ),
        (
            "grouped topk_group > n_group",
            lambda: noaux_tc_op(big, big_bias, 8, 9, 8, 2.0),
        ),
        ("grouped topk > 8", lambda: noaux_tc_op(big, big_bias, 8, 4, 9, 2.0)),
        (
            "grouped experts_per_group > 32",
            lambda: noaux_tc_op(big, big_bias, 4, 2, 8, 2.0),
        ),
        (
            "grouped num_experts > 256",
            lambda: noaux_tc_op(
                torch.randn(8, 512, dtype=torch.bfloat16, device="cuda"),
                torch.randn(512, dtype=torch.bfloat16, device="cuda"),
                8,
                4,
                8,
                2.0,
            ),
        ),
    ]
    for tag, call in cases:
        try:
            call()
        except (RuntimeError, ValueError, NotImplementedError):
            continue
        raise AssertionError(f"op accepted an unsupported domain: {tag}")


def test_wrapper_rejects_silently_wrong_inputs() -> None:
    torch.manual_seed(13)
    bias = torch.randn(72, dtype=torch.bfloat16, device="cuda")
    wide = torch.randn(8, 144, dtype=torch.bfloat16, device="cuda")
    logits = torch.randn(8, 72, dtype=torch.bfloat16, device="cuda")
    wide_bias = torch.randn(144, dtype=torch.bfloat16, device="cuda")

    # A column slice is not routed on its own elements: the kernel reads the
    # first 8*72 elements of the underlying storage as a dense [8, 72] buffer.
    sliced = wide[:, :72]
    assert not sliced.is_contiguous()
    reinterpreted = wide.reshape(-1)[: 8 * 72].reshape(8, 72).contiguous()
    raw = torch.ops.trtllm.noaux_tc_op(sliced, bias, 1, 1, 6, 2.0)
    dense = torch.ops.trtllm.noaux_tc_op(reinterpreted, bias, 1, 1, 6, 2.0)
    correct = torch.ops.trtllm.noaux_tc_op(sliced.contiguous(), bias, 1, 1, 6, 2.0)
    assert torch.equal(raw[1], dense[1]), "strided read is not a dense reinterpretation"
    assert not torch.equal(raw[1], correct[1]), "strided view happened to be correct"

    bad_inputs = [
        ("strided router_logits", lambda: noaux_tc_op(sliced, bias, 1, 1, 6, 2.0)),
        (
            "transposed router_logits",
            lambda: noaux_tc_op(wide.t()[:72, :], bias, 1, 1, 6, 2.0),
        ),
        ("strided bias", lambda: noaux_tc_op(logits, wide_bias[::2], 1, 1, 6, 2.0)),
        ("topk > num_experts", lambda: noaux_tc_op(logits, bias, 1, 1, 73, 2.0)),
    ]
    for tag, call in bad_inputs:
        try:
            call()
        except AssertionError:
            continue
        raise AssertionError(f"wrapper accepted a silently-wrong input: {tag}")

    # the same data made contiguous routes correctly
    _check(sliced.contiguous(), bias, 1, 1, 6, 2.0)
