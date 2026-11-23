# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter module's own consistency, on CPU: host bundle, pools and weights.

`test_draft.py` holds the drafter's geometry and weight table against the
checkpoint. This module holds the *module* against both: that the geometry one
of its five layers reads off its host is the target sparse layer's, that a draft
layer writes and reads the pool of the target layer the config names for it, and
that the names its block reads are exactly the names the table declares. A key
the block reads but no entry covers is a `KeyError` on the first GPU step; a
declared entry nothing reads is a tensor loaded and never used.

What this module cannot see: any kernel's numerics, and whether the block
metadata the engine hands the drafter is the block it draft -- those need a
device and the acceptance gate.
"""

from __future__ import annotations

# The checkpoint stand-in this directory's tests share; pytest puts the test
# file's own directory on the path.
from _target import core, drafter, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.dims import (  # noqa: E501
    THETA_SPARSE,
    dims,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.draft import (  # noqa: E501
    DRAFT_KEY_PREFIX,
    DflyBlockSelection,
    draft_bundle,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.spec.dfly_block_prepare import (  # noqa: E501
    BLOCK_WIDTH,
    MASK_TOKEN_ID,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.weights import (  # noqa: E501
    DRAFT_WEIGHTS,
)

pytestmark = requires_checkpoint

#: Every parameter one draft layer's block reads. The core's layer has a
#: hyper-connection and (on its sparse layers) an indexer, and the drafter has
#: neither: a draft block is a plain pre-norm residual pair of sublayers, so its
#: weights are the sandwich norms, the MLA with its sink and gate, the FFN.
PER_LAYER_ROLES = (
    "input_layernorm",
    "post_attention_layernorm",
    "q_a_proj",
    "q_a_layernorm",
    "kv_a_proj_with_mqa",
    "kv_a_layernorm",
    "q_b_proj",
    "q_b_proj_scale",
    "k_b_proj",
    "v_b_proj",
    "linear_gate",
    "linear_gate_scale",
    "o_proj",
    "o_proj_scale",
    "sink",
    "post_proj_norm",
    "mlp_gate",
    "mlp_gate_scale",
    "mlp_up",
    "mlp_up_scale",
    "mlp_down",
    "mlp_down_scale",
)

#: The four heads and the norm, which the whole drafter carries once.
SINGLETON_ROLES = (
    "fusion_enorm",
    "fusion_hnorm",
    "fusion_eh_proj",
    "fusion_gate",
    "corr_hidden_norm",
    "corr_embed_norm",
    "corr_gate",
    "corr_up",
    "corr_down",
    "conf_proj",
    "conf_bias",
    "final_norm",
)


def test_the_bundle_repeats_the_target_sparse_geometry():
    """One geom for five layers, and it is the target's sparse layer's."""
    stand_in = core()
    bundle = draft_bundle(stand_in)
    c = dims(stand_in)
    assert len(bundle.layers) == bundle.num_layers == 5
    assert bundle.target_kv_layers == (25, 29, 33, 37, 41)
    for geom in bundle.layers:
        assert geom == bundle.layers[0]
        assert geom.sparse is False, "a draft layer selects with the target's picks"
        assert geom.window is None, "a draft row attends the whole selection"
        assert geom.theta == THETA_SPARSE
        assert (geom.heads, geom.heads_local) == (c.heads_sparse, c.heads_sparse // c.attn_tp)


def test_the_host_hands_the_reused_mla_layer_every_width_it_reads():
    """`MlaLayer` reads these off its host; a missing one is an AttributeError on GPU."""
    bundle = draft_bundle(core())
    for field in (
        "layers",
        "tokens_per_block",
        "max_seq_len",
        "max_batch",
        "max_context_len",
        "hidden",
        "heads",
        "heads_local",
        "nope",
        "rope_dim",
        "v_dim",
        "kv_lora",
        "q_lora",
        "qk_dim",
        "lat_dim",
        "q_scaling",
        "rms_eps",
        "attn_tp",
        "mlp_tp",
        "block_width",
        "target_kv_layers",
        "index_width",
    ):
        assert hasattr(bundle, field), field
    # The block's per-row selection is wider than the picks it holds: the row
    # attends its picks and its own block, rounded up to the entry's alignment.
    assert bundle.index_width == 2176
    assert bundle.index_width > bundle.index_topk + bundle.block_width


def test_a_draft_layer_reads_the_target_layer_the_config_names():
    """Layer i's pool row is a sparse target layer's, and it is the paired one."""
    d = drafter()
    c = dims(core())
    for i in range(d.dims.num_layers):
        row = d._pool_row(None, i)
        assert row == d.dims.target_kv_layers[i]
        assert c.layers[row].sparse, f"target layer {row} has no pooled picks"
        assert d.layers[i].indexer is None, "the drafter has no indexer of its own"
        assert d.layers[i].geom == d.dims.layers[i]


def test_the_engine_surface_the_drafter_declares():
    """What `DFlashWorker` reads before it runs a drafter, and the draft's keys."""
    d = drafter()
    assert d.block_size == BLOCK_WIDTH == 6
    assert d.mask_token_id == MASK_TOKEN_ID == 120009
    assert DRAFT_KEY_PREFIX == "model.dfly_layers."
    assert d.checkpoint_key_prefixes == (DRAFT_KEY_PREFIX,)
    # DFly2 draws one coherent path with a hidden correction: it is neither a
    # DSpark candidate selector nor an lili-correction, and the engine must take
    # the plain block path (`_sample_block_draft_tokens`).
    assert d.has_candidate_selector is False
    assert d.is_dflash2 is False
    assert d.lilicorr is None
    assert DflyBlockSelection._fields == (
        "packed",
        "lengths",
        "slots",
        "positions",
        "active",
        "attn_metadata",
    )


def test_the_block_reads_exactly_the_weights_the_table_declares():
    """No `KeyError` on the first GPU step, and no tensor loaded and never read."""
    d = drafter()
    for i in range(d.dims.num_layers):
        for role in PER_LAYER_ROLES:
            assert f"l{i}_{role}" in d.w, f"l{i}_{role}"
    for role in SINGLETON_ROLES:
        assert role in d.w, role
    declared = {key.split("_", 1)[1] for key in d.w if key.startswith("l0_")}
    assert declared == set(PER_LAYER_ROLES)
    assert set(d.w) == set(DRAFT_WEIGHTS.manifest(d)), "manifest/parameter drift"
    assert len(d.w) == 5 * len(PER_LAYER_ROLES) + len(SINGLETON_ROLES)
