# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unified target's core at tp4, tep4, dep4 and dep2: each rank's forward widths and its
experts.

The core is built on the meta device from the checkpoint's text and vision config values at
every rank of a tp4, tep4, dep4 or dep2 mapping, so nothing here touches a device. What is pinned is
what the forward addresses: the fused attention row and the split the norm + rope reads it
through, the Gated-DeltaNet projection's column split, the routed experts a rank holds (an
expert window, or tp4's padded intermediate slice of every expert) and the per-expert scalars
cut to them, and the attention-DP round trip's padding and slicing. MTP under attention DP is
pinned in `test_mtp_attention_dp.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.configs import Qwen4ExpTextConfig, Qwen4ExpVisionConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi import QSASparseAttentionConfig
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

NUM_LAYERS = 48
WORLD = 4

# Qwen3.8-Flash-Next's text config.
_TEXT = dict(
    architectures=["Qwen4ExpForCausalLM"],
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=248044,
    dtype="bfloat16",
    eos_token_id=248044,
    full_attention_interval=4,
    hc_count=4,
    hc_lowrank=320,
    head_dim=256,
    heads_per_ngram=8,
    hidden_act="silu",
    hidden_size=2560,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    indexer_head_dim=128,
    indexer_kv_heads=1,
    indexer_n_heads=4,
    layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(NUM_LAYERS)],
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_value_head_dim=128,
    make_ngram_vocab_size_divisible_by=128,
    mamba_ssm_dtype="float32",
    max_position_embeddings=262144,
    moe_intermediate_size=640,
    mtp=dict(
        hybrid=True,
        layer_types=["full_attention"],
        mtp_use_hidden_state_from_layer=None,
        num_hidden_layers=1,
        rope_theta=10000000,
    ),
    mtp_num_hidden_layers=1,
    mtp_use_dedicated_embeddings=False,
    ngram_size=3,
    ngram_vocab_size_base=20000000,
    norm_topk_prob=True,
    num_attention_heads=24,
    num_experts=512,
    num_experts_per_tok=10,
    num_hidden_layers=NUM_LAYERS,
    num_key_value_heads=2,
    number_of_conv_states=3,
    output_gate_type="sigmoid",
    partial_rotary_factor=0.25,
    ple_conv_kernel_size=4,
    ple_embed_dim=2560,
    ple_layer_ids=[2],
    rms_norm_eps=1e-06,
    rope_parameters=dict(
        mrope_interleaved=True,
        mrope_section=[11, 11, 10],
        partial_rotary_factor=0.25,
        rope_theta=10000000,
        rope_type="default",
    ),
    shared_expert_intermediate_size=640,
    split_ngram_parts=128,
    tie_word_embeddings=False,
    vocab_size=248320,
)
_VISION = dict(
    deepstack_visual_indexes=[],
    depth=27,
    dtype="bfloat16",
    hidden_act="gelu_pytorch_tanh",
    hidden_size=1152,
    in_channels=3,
    intermediate_size=4304,
    num_heads=16,
    num_position_embeddings=2304,
    out_hidden_size=2560,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
)

# (attention DP, MoE TP, MoE EP) of the three members.
_TEP4 = dict(enable_attention_dp=False, moe_tp_size=1, moe_ep_size=4)
_DEP4 = dict(enable_attention_dp=True, moe_tp_size=1, moe_ep_size=4)
_DEP2 = dict(enable_attention_dp=True, moe_tp_size=1, moe_ep_size=2)
_TP4 = dict(enable_attention_dp=False, moe_tp_size=4, moe_ep_size=1)


def _core(rank: int, layout: dict, spec_config=None, world: int = WORLD):
    """The core of rank `rank` of a `world`-rank engine at `layout`, every parameter on meta."""
    model_config = ModelConfig(
        pretrained_config=Qwen4ExpTextConfig(**_TEXT),
        mapping=Mapping(world_size=world, tp_size=world, rank=rank, **layout),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        moe_backend="TRTLLM",
        moe_backend_requested="TRTLLM",
        sparse_attention_config=QSASparseAttentionConfig(),
        spec_config=spec_config,
    )
    composite = SimpleNamespace(
        vision_config=Qwen4ExpVisionConfig(**_VISION), image_token_id=248056, video_token_id=248057
    )
    # The core asserts the device it runs on; the weights it declares stay on meta.
    with (
        mock.patch.object(torch.cuda, "get_device_capability", return_value=M._SM),
        torch.device("meta"),
    ):
        return M.Qwen4ExpUnifiedModelingV2(model_config, composite)


@pytest.mark.parametrize("rank", range(WORLD))
def test_a_tep4_rank_runs_its_share_of_the_heads(rank) -> None:
    """Six q heads and one of the two KV heads (repeated on rank pairs) over a 4352-wide
    fused row: `[q 1536 | k 256 | v 256 | gate 1536 | index q 512, k 128 | v slot 128]`.
    The main norm + rope reads it as 6 q, 1 k and 10 "v" heads of 256 (the row stride
    alone), which is the cell `fused_qk_norm_rope` certifies."""
    core = _core(rank, _TEP4)
    assert (core.heads_q, core.heads_kv, core.kv_head_start) == (6, 1, rank * 2 // WORLD)
    assert (core.q_width, core.kv_width, core.qkv_width) == (1536, 256, 2048)
    assert (core.index_width, core.index_scratch_width) == (640, 768)
    assert core.attn_in_width == 4352
    assert core.attn_in_tail_heads == 10
    assert core.index_col == 3584
    assert tuple(core.w["l3_attn_in_w"].shape) == (core.attn_in_width, 2560)
    assert tuple(core.w["l3_o_w"].shape) == (2560, core.q_width)


@pytest.mark.parametrize("rank", range(WORLD))
def test_a_tep4_rank_runs_its_share_of_the_recurrent_heads(rank) -> None:
    """Four key and twelve value heads: the conv runs over `[q | k | v]` = 512 + 512 + 1536
    channels, and the one input projection's output splits `[q|k|v 2560 | z 1536 | a 12 |
    b 12]`."""
    core = _core(rank, _TEP4)
    assert (core.num_k_heads, core.num_v_heads) == (4, 12)
    assert (core.key_dim, core.value_dim, core.conv_dim) == (512, 1536, 2560)
    assert tuple(core.in_proj_split) == (2560, 1536, 12, 12)
    assert tuple(core.w["l0_in_proj_w"].shape) == (sum(core.in_proj_split), 2560)
    assert tuple(core.w["l0_out_w"].shape) == (2560, core.value_dim)
    assert tuple(core.w["l0_conv_w"].shape) == (core.conv_dim, 1, 4)


@pytest.mark.parametrize("rank", range(WORLD))
def test_a_tep4_rank_holds_one_128_expert_window(rank) -> None:
    """The routed experts' window is `[128r, 128r + 128)`; everything the router, the shared
    expert and the hyper-connections read is whole on every rank."""
    core = _core(rank, _TEP4)
    assert (core.local_experts, core.expert_offset) == (128, 128 * rank)
    assert core.w["l0_fc1_w"].shape[0] == 128
    assert core.w["l0_fc2_w"].shape[0] == 128
    assert tuple(core.w["l0_router"].shape) == (512, 2560)
    assert tuple(core.w["embed"].shape) == (248320, 2560)


@pytest.mark.parametrize("rank", range(WORLD))
def test_a_tp4_rank_holds_every_expert_at_its_padded_intermediate_slice(rank) -> None:
    """tp4 cuts attention and Gated-DeltaNet as tep4 does, and holds all 512 experts at a
    160-row intermediate slice stored 256 wide: the runner is handed the whole routing space
    (`local_expert_offset` 0, `local_num_experts` 512) at `intermediate_size` 256, the
    stack `fp4_block_scale_moe_runner` certifies, and the scalars span every expert."""
    core = _core(rank, _TP4)
    assert (core.heads_q, core.heads_kv, core.kv_head_start) == (6, 1, rank * 2 // WORLD)
    assert (core.num_k_heads, core.num_v_heads, core.attn_in_width) == (4, 12, 4352)
    assert (core.dp_size, core.attn_tp_size) == (1, WORLD)
    assert (core.local_experts, core.expert_offset, core.expert_inter) == (512, 0, 256)
    assert core.moe_inter == 640
    assert tuple(core.w["l0_fc1_w"].shape) == (512, 2 * 256, 2560 // 2)
    assert tuple(core.w["l0_fc1_s"].shape) == (512, 2 * 256, 2560 // 16)
    assert tuple(core.w["l0_fc2_w"].shape) == (512, 2560, 256 // 2)
    assert tuple(core.w["l0_fc2_s"].shape) == (512, 2560, 256 // 16)
    assert tuple(core.w["l0_e_isc2"].shape) == (512,)


@pytest.mark.parametrize("layout", [_TEP4, _DEP4], ids=["tep4", "dep4"])
def test_an_expert_window_runs_at_the_whole_intermediate(layout) -> None:
    core = _core(1, layout)
    assert (core.local_experts, core.expert_offset, core.expert_inter) == (128, 128, 640)


def test_the_expert_scalars_are_cut_to_the_window() -> None:
    """The runner takes one scalar per expert of its window: each is the whole routing
    space's product, sliced, and the FC1 activation scale is the layer's one number."""
    num_experts, start, count = 512, 256, 128
    gen = torch.Generator().manual_seed(0)
    isc1 = torch.full((num_experts,), 0.5)
    ws2_1 = torch.rand(num_experts, generator=gen) + 0.5
    isc2 = torch.rand(num_experts, generator=gen) + 0.5
    ws2_2 = torch.rand(num_experts, generator=gen) + 0.5
    w = {
        "l0_e_isc1": isc1,
        "l0_e_isc1_up": isc1.clone(),
        "l0_e_ws2_1": ws2_1,
        "l0_e_ws2_1_up": ws2_1.clone(),
        "l0_e_isc2": isc2,
        "l0_e_ws2_2": ws2_2,
    }
    s = M._moe_layer_scalars(w, 0, start, count)
    window = slice(start, start + count)
    gate1 = isc1 * ws2_1
    torch.testing.assert_close(s["global_scale"], torch.tensor([2.0]), rtol=0, atol=0)
    torch.testing.assert_close(s["output1_scale_gate_scalar"], gate1[window], rtol=0, atol=0)
    torch.testing.assert_close(s["output1_scale_scalar"], (gate1 / isc2)[window], rtol=0, atol=0)
    torch.testing.assert_close(s["output2_scale_scalar"], (isc2 * ws2_2)[window], rtol=0, atol=0)
    assert all(t.shape == (count,) for k, t in s.items() if k != "global_scale")


def test_a_spread_fc1_input_scale_is_refused() -> None:
    """One activation quantization feeds every expert, so the layer must have one
    `gate_proj.input_scale`."""
    isc1 = torch.full((512,), 0.5)
    isc1[7] = 0.25
    ones = torch.ones(512)
    w = {
        "l0_e_isc1": isc1,
        "l0_e_isc1_up": isc1.clone(),
        "l0_e_ws2_1": ones,
        "l0_e_ws2_1_up": ones.clone(),
        "l0_e_isc2": ones,
        "l0_e_ws2_2": ones,
    }
    with pytest.raises(AssertionError, match="one gate_proj.input_scale"):
        M._moe_layer_scalars(w, 0, 0, 128)


@pytest.mark.parametrize("rank", range(WORLD))
def test_a_dep4_rank_runs_every_head(rank) -> None:
    """Attention DP keeps the single-rank geometry: 24 q and both KV heads over the 14080-wide
    fused row (55 heads of 256: 24 q, 2 k, 29 "v"), and all 16 / 48 GDN heads over 10240 conv
    channels. Nothing is summed after the mixers."""
    core = _core(rank, _DEP4)
    assert (core.heads_q, core.heads_kv, core.kv_head_start) == (24, 2, 0)
    assert (core.q_width, core.kv_width, core.qkv_width) == (6144, 512, 7168)
    assert core.attn_in_width == 14080
    assert core.attn_in_tail_heads == 29
    assert core.index_col == 13312
    assert tuple(core.w["l3_attn_in_w"].shape) == (14080, 2560)
    assert tuple(core.w["l3_o_w"].shape) == (2560, 6144)
    assert (core.num_k_heads, core.num_v_heads, core.conv_dim) == (16, 48, 10240)
    assert tuple(core.in_proj_split) == (10240, 6144, 48, 48)
    assert core.attn_tp_size == 1


@pytest.mark.parametrize("rank", range(WORLD))
def test_a_dep4_rank_runs_one_128_expert_window_for_the_group(rank) -> None:
    """The MoE group is the TP group, which spans the DP ranks: rank `r` is DP rank `r` and
    holds the window `[128r, 128r + 128)`, over the rows the whole group gathers."""
    core = _core(rank, _DEP4)
    assert (core.dp_size, core.dp_rank, core.dp_group) == (WORLD, rank, list(range(WORLD)))
    assert (core.local_experts, core.expert_offset) == (128, 128 * rank)
    assert core.w["l0_fc1_w"].shape[0] == 128
    assert _core(rank, _TEP4).dp_size == 1


@pytest.mark.parametrize("rank", range(2))
def test_a_dep2_rank_runs_every_head_and_one_256_expert_window(rank) -> None:
    """dep2, the disaggregated-serving engines' layout, is dep4 at half the group: the same
    single-rank attention and Gated-DeltaNet geometry, and rank `r` holds the 256-expert
    window `[256r, 256r + 256)` over the rows both ranks gather."""
    core, dep4 = _core(rank, _DEP2, world=2), _core(rank, _DEP4)
    for name in (
        "heads_q",
        "heads_kv",
        "kv_head_start",
        "qkv_width",
        "attn_in_width",
        "attn_in_tail_heads",
        "index_col",
        "num_k_heads",
        "num_v_heads",
        "conv_dim",
        "attn_tp_size",
    ):
        assert getattr(core, name) == getattr(dep4, name), name
    assert tuple(core.in_proj_split) == tuple(dep4.in_proj_split)
    for key in ("l3_attn_in_w", "l3_o_w", "l0_in_proj_w", "l0_router", "l0_sh_gu"):
        assert core.w[key].shape == dep4.w[key].shape, key
    assert (core.dp_size, core.dp_rank, core.dp_group) == (2, rank, [0, 1])
    assert (core.local_experts, core.expert_offset) == (256, 256 * rank)
    assert core.w["l0_fc1_w"].shape[0] == 256
    assert core.w["l0_fc1_w"].shape[1:] == dep4.w["l0_fc1_w"].shape[1:]


def test_attention_dp_pads_to_the_groups_largest_row_count() -> None:
    """Each rank pads to the largest of `all_rank_num_tokens`; a count that disagrees with
    the rows this rank holds would gather the wrong split, and is refused."""
    core = SimpleNamespace(dp_size=WORLD, dp_rank=2)
    md = SimpleNamespace(all_rank_num_tokens=[5, 17, 9, 1])
    assert M.Qwen4ExpUnifiedModelingV2._dp_pad_rows(core, md, 9) == 8
    with pytest.raises(AssertionError, match="wrong split"):
        M.Qwen4ExpUnifiedModelingV2._dp_pad_rows(core, md, 10)
    # Without attention DP the metadata is not read.
    assert M.Qwen4ExpUnifiedModelingV2._dp_pad_rows(SimpleNamespace(dp_size=1), None, 3) == 0


@pytest.mark.parametrize("rank", range(WORLD))
def test_the_attention_dp_round_trip_returns_this_ranks_rows_through_every_window(
    rank, monkeypatch
) -> None:
    """The round trip pads this rank's rows, gathers the group's in rank order, runs this
    rank's window over all of them, reduce-scatters and drops the padding: with each window
    a function of the gathered rows, this rank gets back the sum over windows of its own rows.
    The collectives are stood in for by the group's arithmetic."""
    counts = [3, 7, 1, 4]
    hidden = 8
    gen = torch.Generator().manual_seed(rank)
    xs = [torch.randn(n, hidden, generator=gen) for n in counts]
    rows = max(counts)
    gathered = torch.cat([nn_pad(x, rows) for x in xs], 0)

    def window(w):
        return lambda t: t * (w + 1) + w

    full = sum(window(w)(gathered) for w in range(WORLD))
    seen = {}

    def fake_allgather(t, sizes, group, process_group):
        assert sizes is None and group == list(range(WORLD)) and process_group is None
        assert tuple(t.shape) == (rows, hidden)
        torch.testing.assert_close(t, nn_pad(xs[rank], rows), rtol=0, atol=0)
        return gathered

    def fake_reducescatter(t, sizes, group, process_group):
        assert sizes is None and group == list(range(WORLD)) and process_group is None
        seen["partial"] = t
        return full[rank * rows : (rank + 1) * rows]

    monkeypatch.setattr(M, "allgather", fake_allgather)
    monkeypatch.setattr(M, "reducescatter", fake_reducescatter)
    monkeypatch.setattr(M, "mpi_disabled", lambda: False)
    core = SimpleNamespace(dp_size=WORLD, dp_group=list(range(WORLD)), mapping=None)
    expert_sum = M.Qwen4ExpUnifiedModelingV2._expert_sum_op(core)
    out = expert_sum(window(rank), xs[rank], rows - counts[rank])
    torch.testing.assert_close(seen["partial"], window(rank)(gathered), rtol=0, atol=0)
    torch.testing.assert_close(out, sum(window(w)(xs[rank]) for w in range(WORLD)))
    assert tuple(out.shape) == (counts[rank], hidden)


def nn_pad(x: torch.Tensor, rows: int) -> torch.Tensor:
    """`x` with zero rows appended up to `rows`."""
    return torch.cat([x, x.new_zeros(rows - x.shape[0], x.shape[1])], 0)


@pytest.mark.parametrize("max_draft_len", [1, 3, 7])
def test_mtp_widens_the_verify_step_and_keys_the_draft_layer_after_the_trunk(
    max_draft_len,
) -> None:
    """A verify step carries `max_draft_len + 1` rows a generation request; the draft layer is
    global layer 48, the index every KV manager maps it at and the weight table keys it at.
    Without speculation the MTP module is not declared."""
    core = _core(0, _TEP4, MTPDecodingConfig(max_draft_len=max_draft_len))
    assert core.mtp_enabled and core.max_verify_rows == max_draft_len + 1
    assert core.mtp_layer_idx == NUM_LAYERS
    assert core.mtp_theta == core.theta == 1e7
    assert core.spec_rollback is not None
    assert any(k.startswith(f"l{NUM_LAYERS}_mtp_") for k in core.w)
    plain = _core(0, _TEP4)
    assert not plain.mtp_enabled and plain.max_verify_rows == 1 and plain.spec_rollback is None
    assert not any("_mtp_" in k for k in plain.w)


@pytest.mark.parametrize("rank", range(WORLD))
def test_a_tep4_rank_holds_its_share_of_the_draft_layer(rank) -> None:
    """The draft layer's attention is cut like a trunk attention layer's: six q heads (their
    q and interleaved gate rows), one KV head and the o_proj columns of the six heads; its
    QSA indexer, front end, router, shared expert and mixers are whole. Its FP8 experts are
    the rank's window `[128r, 128r + 128)` over the TP group, FP8 128x128 block scales."""
    core = _core(rank, _TEP4, MTPDecodingConfig(max_draft_len=3))
    p = f"l{NUM_LAYERS}_mtp_"
    w = core.w
    assert tuple(w[f"{p}q_w"].shape) == (2 * 1536, 2560)
    assert tuple(w[f"{p}k_w"].shape) == tuple(w[f"{p}v_w"].shape) == (256, 2560)
    assert tuple(w[f"{p}o_w"].shape) == (2560, 1536)
    assert tuple(w[f"{p}idx_w"].shape) == (640, 2560)
    assert tuple(w[f"{p}fc_e"].shape) == tuple(w[f"{p}fc_h"].shape) == (2560, 2560)
    assert tuple(w[f"{p}pre_h_norm"].shape) == (4 * 2560,)
    assert tuple(w[f"{p}router"].shape) == (512, 2560)
    assert tuple(w[f"{p}sh_gu"].shape) == (2 * 640 + 8, 2560)
    assert (core.mtp_local_experts, core.mtp_expert_offset) == (128, 128 * rank)
    assert tuple(w[f"{p}fc1_w"].shape) == (128, 2 * 640, 2560)
    assert w[f"{p}fc1_w"].dtype == torch.float8_e4m3fn
    assert tuple(w[f"{p}fc1_s"].shape) == (128, 2 * 640 // 128, 2560 // 128)
    assert tuple(w[f"{p}fc2_w"].shape) == (128, 2560, 640)
    assert tuple(w[f"{p}fc2_s"].shape) == (128, 2560 // 128, 640 // 128)


@pytest.mark.parametrize("rank", range(WORLD))
def test_a_tp4_rank_holds_whole_expert_draft_windows_beside_its_intermediate_slice(rank) -> None:
    """At tp4 the trunk's routed experts are every expert at a 160-row intermediate slice
    stored 256 wide, but the draft layer's FP8 experts cannot be cut that way (their 128x128
    weight blocks would split mid-block). They are the rank's whole-expert window
    `[128r, 128r + 128)` over the TP group, exactly tep4's, and the four windows partition
    the routing space. The draft attention is cut as at tep4."""
    max_draft_len = 3
    core = _core(rank, _TP4, MTPDecodingConfig(max_draft_len=max_draft_len))
    assert core.mtp_enabled and core.max_verify_rows == max_draft_len + 1
    assert (core.local_experts, core.expert_offset, core.expert_inter) == (512, 0, 256)
    assert (core.mtp_local_experts, core.mtp_expert_offset, core.moe_inter) == (
        128,
        128 * rank,
        640,
    )
    p = f"l{NUM_LAYERS}_mtp_"
    w = core.w
    assert tuple(w[f"{p}fc1_w"].shape) == (128, 2 * 640, 2560)
    assert w[f"{p}fc1_w"].dtype == torch.float8_e4m3fn
    assert tuple(w[f"{p}fc1_s"].shape) == (128, 2 * 640 // 128, 2560 // 128)
    assert tuple(w[f"{p}fc2_w"].shape) == (128, 2560, 640)
    assert tuple(w[f"{p}fc2_s"].shape) == (128, 2560 // 128, 640 // 128)
    assert tuple(w[f"{p}q_w"].shape) == (2 * 1536, 2560)
    assert tuple(w[f"{p}k_w"].shape) == tuple(w[f"{p}v_w"].shape) == (256, 2560)
    assert tuple(w[f"{p}o_w"].shape) == (2560, 1536)
    assert tuple(w[f"{p}router"].shape) == (512, 2560)
    windows = [_core(r, _TP4, MTPDecodingConfig(max_draft_len=1)) for r in range(WORLD)]
    covered = sorted(
        e
        for c in windows
        for e in range(c.mtp_expert_offset, c.mtp_expert_offset + c.mtp_local_experts)
    )
    assert covered == list(range(512))


def test_the_tp4_draft_windows_sum_to_the_whole_draft_moe(monkeypatch) -> None:
    """`_mtp_moe_block` at tp4 hands the FP8 runner this rank's whole-expert window -- the
    draft layer's `local_expert_offset` / `local_num_experts` and the whole 640-row
    intermediate, never the trunk's 256-wide padded slice -- and completes the window's
    partial with the TP sum once. The entries are stood in for by arithmetic that names the
    expert a slot routes to, so the four ranks' partials sum to the all-expert output."""
    tokens, hidden, num_experts, topk = 5, 2560, 512, 10
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(tokens, hidden, generator=gen)
    router = torch.randn(hidden, num_experts, generator=gen)
    logits = x @ router
    top_logits, top_ids = torch.topk(logits, topk, dim=-1)
    top_w = torch.softmax(top_logits, dim=-1)

    def expert_out(ids, wts, start, count):
        keep = ((ids >= start) & (ids < start + count)).to(x.dtype)
        return (wts * keep * (ids + 1).to(x.dtype)).sum(-1, keepdim=True) * x

    full = expert_out(top_ids, top_w, 0, num_experts)

    def fake_cublas_mm(a, b, *args, **kwargs):
        return a @ b

    def fake_routing(lg, topk, output_dtype):
        assert topk == 10 and lg.shape == (tokens, num_experts)
        return top_ids, top_w

    def fake_runner(**kw):
        core = seen["core"]
        assert kw["intermediate_size"] == core.moe_inter == 640
        assert kw["local_expert_offset"] == core.mtp_expert_offset
        assert kw["local_num_experts"] == core.mtp_local_experts == 128
        assert tuple(kw["gemm1_weights"].shape) == (128, 2 * 640, hidden)
        assert tuple(kw["gemm2_weights"].shape) == (128, hidden, 640)
        assert kw["num_experts"] == num_experts and kw["top_k"] == topk
        return expert_out(
            kw["topk_ids"], kw["topk_weights"], kw["local_expert_offset"], kw["local_num_experts"]
        )

    monkeypatch.setattr(M, "cublas_mm", fake_cublas_mm)
    monkeypatch.setattr(M, "renorm_moe_routing_op", fake_routing)
    monkeypatch.setattr(M, "fp8_quantize_1x128", lambda t, use_ue8m0: (t, None))
    monkeypatch.setattr(M, "fp8_block_scale_moe_runner", fake_runner)
    monkeypatch.setattr(M, "silu_and_mul_rows", lambda gug, inter: gug[:, :inter])
    # The shared expert and its gate are replicated and checked elsewhere: keep the routed sum.
    monkeypatch.setattr(M, "fused_sigmoid_gate_mul_add", lambda routed, gate, shared: routed)

    seen = {}
    partials = []
    for rank in range(WORLD):
        core = _core(rank, _TP4, MTPDecodingConfig(max_draft_len=3))
        seen["core"] = core
        summed = []
        core.tp_sum = lambda t: summed.append(t) or t
        p = f"l{NUM_LAYERS}_mtp_"
        views = SimpleNamespace(
            router=router,
            fc1_w=core.w[f"{p}fc1_w"],
            fc1_s=core.w[f"{p}fc1_s"],
            fc2_w=core.w[f"{p}fc2_w"],
            fc2_s=core.w[f"{p}fc2_s"],
            sh_gug=torch.zeros(hidden, 2 * 640 + 8),
            sh_dn=torch.zeros(640, hidden),
        )
        out = M._mtp_moe_block(core, views, x)
        assert len(summed) == 1, "the window's partial is summed over the TP group once"
        partials.append(out)
    torch.testing.assert_close(sum(partials), full)


def test_the_draft_layer_views_are_the_trunks_gemm_operands() -> None:
    """`_mtp_views` hands the GEMMs column-major views: the shared expert's one
    `[gate | up | gate logit | pad]` operand (the trunk layers' 2560 -> 1288 shape), the q
    projection with its interleaved gate, and the grouped norms as one slice a stream."""
    core = _core(0, _TEP4, MTPDecodingConfig(max_draft_len=3))
    with torch.device("cpu"):
        for key, t in core.w.items():
            core.w[key] = torch.nn.Parameter(
                torch.empty(t.shape, dtype=t.dtype, device="cpu"), requires_grad=False
            )
    M_ = M._mtp_views(core)
    assert tuple(M_.sh_gug.shape) == (2560, 1288) and M_.sh_gug.stride() == (1, 2560)
    assert tuple(M_.sh_dn.shape) == (640, 2560) and M_.sh_dn.stride() == (1, 640)
    assert tuple(M_.o.shape) == (1536, 2560) and M_.o.stride() == (1, 1536)
    assert tuple(M_.q.shape) == (2560, 3072) and M_.q.stride() == (1, 2560)
    assert len(M_.pre_h_norm) == len(M_.ahc_norm) == len(M_.fhc_norm) == 4
    assert all(tuple(s.shape) == (2560,) for s in M_.mhc_norm)
