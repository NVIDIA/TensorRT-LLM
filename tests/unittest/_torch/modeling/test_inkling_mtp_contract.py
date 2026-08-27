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
"""The interface MTPWorker calls the draft chain with.

``MTPWorker.__call__`` does, per depth:

    hidden = mtp_layer(embed_tokens=draft_model.embed_tokens, **draft_inputs)
    logits = mtp_layer.shared_head(hidden, draft_model.lm_head, attn_metadata)

where ``draft_inputs`` carries ``input_ids``, ``position_ids``,
``hidden_states`` and ``attn_metadata``. A signature that does not match is a
TypeError deep inside the speculative loop, on a multi-GPU run, after several
minutes of model load -- the most expensive place to discover a keyword name.

These assertions are signature-level on purpose: they run without a GPU, a
checkpoint or a built extension, so the contract is checked in seconds rather
than at the end of an end-to-end job.
"""

import inspect

import pytest

from tensorrt_llm._torch.configs.inkling import InklingConfig
from tensorrt_llm._torch.models.modeling_inkling import InklingMTPBlock, InklingMTPHead

# The keys MTPWorker builds in prepare_drafter_inputs and forwards as **kwargs.
_DRAFT_INPUT_KEYS = {"input_ids", "position_ids", "hidden_states", "attn_metadata"}


def test_block_accepts_every_draft_input_key():
    """Each key MTPWorker passes must be a named parameter or reach **kwargs."""
    params = inspect.signature(InklingMTPBlock.forward).parameters
    named = set(params) - {"self"}
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = _DRAFT_INPUT_KEYS - named
    assert not missing or has_var_kw, (
        f"MTPWorker passes {sorted(missing)} which the block neither names nor "
        "absorbs into **kwargs"
    )


def test_block_takes_embed_tokens_as_a_keyword():
    """The worker passes ``embed_tokens=`` explicitly, not positionally.

    It hands over the TARGET model's embedding table -- the draft chain shares
    it rather than owning a second copy -- so the name has to match exactly.
    """
    params = inspect.signature(InklingMTPBlock.forward).parameters
    assert "embed_tokens" in params
    assert params["embed_tokens"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def test_shared_head_signature_matches_the_worker_call():
    """``shared_head(hidden_states, lm_head, attn_metadata)``, positionally."""
    params = list(inspect.signature(InklingMTPHead.forward).parameters.values())[1:]
    positional = [p.name for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[:3] == ["hidden_states", "lm_head", "attn_metadata"], (
        f"MTPWorker calls shared_head positionally; got {positional[:3]}"
    )


def test_block_exposes_shared_head_attribute():
    """The worker reaches ``mtp_layer.shared_head`` by attribute name."""
    assert "shared_head" in inspect.getsource(InklingMTPBlock.__init__)


def test_head_norm_is_built_only_when_the_checkpoint_declares_it():
    """Both shipped checkpoints set chain_hidden_post_norm False.

    Building the norm unconditionally would create a parameter with no
    checkpoint tensor behind it, which the loader then has to explain away.

    Built and inspected rather than grepped: the head is cheap to construct on
    CPU (an optional RMSNorm over a hidden-width vector), so there is no reason
    to assert on its source. The block half of this contract -- that it passes
    the checkpoint's flag down -- stays a source assertion below, because
    constructing a block means constructing a whole decoder layer.
    """
    from types import SimpleNamespace

    import torch

    from tensorrt_llm._torch.models.modeling_inkling import InklingMTPHead

    cfg = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            hidden_size=16, rms_norm_eps=1e-5, torch_dtype=torch.float32
        )
    )
    assert InklingMTPHead(cfg, use_norm=False).norm is None, (
        "a checkpoint that does not declare chain_hidden_post_norm must not get a norm"
    )
    assert InklingMTPHead(cfg, use_norm=True).norm is not None

    # Not convertible without building a decoder layer: that the block reads the
    # checkpoint flag and hands it to the head as use_norm.
    assert "chain_hidden_post_norm" in inspect.getsource(InklingMTPBlock.__init__)


# --- how MTPForCausalLM constructs the chain -------------------------------
# It does, for each depth:
#   mtp_layer(model_config, layer_idx + start_layer_idx, model.aux_stream_dict)
# with start_layer_idx = the TARGET's num_hidden_layers, and reads the chain
# depth from pretrained_config.num_nextn_predict_layers. Inkling declares the
# depth on mtp_config, so the mirroring is what makes the framework able to
# build the chain at all.

_CKPT_MTP = {"num_nextn_predict_layers": 8, "local_layer_ids": [0, 2, 4, 5, 6, 7]}


def test_chain_depth_is_visible_under_the_framework_name():
    """MTPForCausalLM reads ``pretrained_config.num_nextn_predict_layers``.

    Inkling declares it on ``mtp_config``. Without the mirror the framework
    reads None and builds a zero-depth chain -- speculative decoding silently
    does nothing rather than failing.
    """
    text = InklingConfig(text_config={}, mtp_config=dict(_CKPT_MTP)).text_config
    assert text.num_nextn_predict_layers == 8


def test_block_constructor_takes_the_frameworks_three_positionals():
    """``mtp_layer(model_config, layer_idx, aux_stream_dict)``."""
    params = list(inspect.signature(InklingMTPBlock.__init__).parameters.values())[1:]
    positional = [p.name for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[:3] == ["model_config", "depth", "aux_stream_dict"], (
        f"MTPForCausalLM constructs layers positionally; got {positional[:3]}"
    )
    assert params[2].default is None, "aux_stream_dict must stay optional for direct construction"


def test_offset_layer_index_maps_back_onto_the_chain():
    """The framework passes ``depth + target_num_hidden_layers``.

    A 42-layer trunk means depth 0 arrives as 42. Indexing the chain's geometry
    with 42 would read past the chain and treat every depth as global -- wrong
    windows on every banded depth, and no crash to show it.

    This asserted a MODULO until the shipped numbers were checked. 42 % 8 = 2,
    so every block was built with depth ``b + 2``'s geometry while carrying
    depth ``b``'s weights; it survived at max_draft_len 3 only because the
    banded set [0, 2, 4, 5, 6, 7] gives 0,1,2 and 2,3,4 the same banded/global
    pattern. Block 3 is where it diverges.
    """
    from tensorrt_llm._torch.models.modeling_inkling import _mtp_depth_from_global_index

    text = InklingConfig(
        text_config={"num_hidden_layers": 42},
        mtp_config={"num_nextn_predict_layers": 8, "local_layer_ids": [0, 2, 4, 5, 6, 7]},
    ).text_config
    assert 42 % 8 != 0, "the shape that makes a modulo and a subtraction differ"
    for depth in range(8):
        assert _mtp_depth_from_global_index(text, 42 + depth) == depth


def test_an_index_outside_the_chain_is_refused():
    """Off the end means the caller and the config disagree about the trunk.

    Every symptom of that is silent -- wrong window, wrong KV-head count, a conv
    pool sized from a different depth -- so it raises instead of clamping.
    """
    from tensorrt_llm._torch.models.modeling_inkling import _mtp_depth_from_global_index

    text = InklingConfig(
        text_config={"num_hidden_layers": 42},
        mtp_config={"num_nextn_predict_layers": 8, "local_layer_ids": [0, 2]},
    ).text_config
    with pytest.raises(ValueError, match="disagree about the trunk"):
        _mtp_depth_from_global_index(text, 41)
    with pytest.raises(ValueError, match="disagree about the trunk"):
        _mtp_depth_from_global_index(text, 50)


def test_inkling_is_registered_in_the_mtp_dispatch_table():
    """`get_draft_model` picks the MTP class by model_type."""
    import tensorrt_llm._torch.models.modeling_speculative as spec

    src = inspect.getsource(spec.MTPForCausalLM.__init__)
    assert "InklingMTPBlock" in src
    # Both model_types reach it: the text tower reports "inkling_text" and the
    # multimodal wrapper "inkling_mm_model". Registering only one leaves the
    # other raising "Model type ... not supported for MTP" after model load.
    assert "inkling_text" in src and "inkling_mm_model" in src


# --- the draft chain's KV cache manager ------------------------------------


def test_draft_kv_head_list_is_indexed_globally_not_by_depth():
    """The chain's entries must be APPENDED to the trunk's, not replace them.

    ``KVCacheManagerV2`` sets ``num_layers = len(layer_mask)`` and the draft
    mask is ``[False]*trunk + [True]*depths``, so it reads ``num_kv_heads[i]``
    at global indices trunk..trunk+depths. Handing it the chain's list alone
    fails the length assertion; handing it the trunk's alone fails the same
    assertion from the other side -- which is exactly the pair of end-to-end
    failures this branch was written for.
    """
    import inspect

    from tensorrt_llm._torch.pyexecutor import _util

    src = inspect.getsource(_util._create_kv_cache_manager)
    assert "mtp_num_kv_heads_per_layer" in src, (
        "the draft manager must be sized from the chain's geometry"
    )
    assert "num_key_value_heads +\n" in src or "num_key_value_heads + chain" in src, (
        "the chain's entries must be appended so their global indices line up with layer_mask"
    )


# --- loading the draft weights ---------------------------------------------


def test_load_branch_is_a_no_op_without_a_chain():
    """No draft chain means nothing to load, and no error.

    The overwhelmingly common case is a server with speculative decoding off:
    the checkpoint still carries 160 MTP tensors and they must simply be
    ignored, not raise.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    src = inspect.getsource(InklingForCausalLM._load_mtp_weights)
    assert 'getattr(draft_model, "mtp_layers", None)' in src
    assert "if not mtp_layers:" in src and "return" in src


def test_only_the_built_depths_are_loaded():
    """The runtime caps the chain at min(max_draft_len, checkpoint depths).

    A server asking for 3 draft tokens builds 3 blocks out of the checkpoint's
    8, so the extra depths must be filtered out and the shortfall reported --
    otherwise "capped by max_draft_len" looks like a loading bug later.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    src = inspect.getsource(InklingForCausalLM._load_mtp_weights)
    assert "< built" in src
    assert "len(available) > built" in src


def test_draft_weights_go_through_the_generic_loader():
    """``load_state_dict`` cannot load this checkpoint, in either direction.

    The checkpoint carries raw per-projection names (wq_du/wk_dv/wv_dv,
    w13_dn) and full-width tensors; the block has fused qkv_proj/gate_up_proj,
    NVFP4 scale tensors and TP-sharded widths. Fusion, scales and sharding are
    the loader's work. With strict=True that mismatch is at least loud; with
    strict off it would load nothing and leave a drafter proposing noise, which
    shows up only as disappointing throughput.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    src = inspect.getsource(InklingForCausalLM._load_mtp_weights)
    assert "_load_weights_impl" in src
    # The docstring explains why load_state_dict is wrong here, so check the
    # body past it rather than the whole source.
    quote = '"' * 3
    body = src[src.index(quote, src.index(quote) + 3) + 3 :]
    assert "load_state_dict" not in body


def test_the_mapper_renames_the_draft_chain_like_the_trunk():
    """The chain's decoder tail needs the trunk's renames, not a copy of them.

    ``wq_du`` etc. must arrive as separate q/k/v for the loader to fuse, and
    the gate/up-INTERLEAVED w13_dn has to be split before anything sees it --
    concatenating instead of de-interleaving is the classic silent version of
    this bug.
    """
    from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
        InklingHfWeightMapper,
    )

    src = inspect.getsource(InklingHfWeightMapper._map_mtp)
    assert "_LAYER_RENAMES" in src
    assert "_split_interleaved_gate_up" in src
    assert "mtp_layers." in src


def test_both_load_paths_reach_the_draft_chain():
    """Text-only and multimodal both have to load the draft weights.

    The draft chain hangs off the causal-LM, but only the multimodal subclass
    overrides load_weights. Without an override on the base class the text-only
    path runs the inherited loader and the draft blocks keep their initial
    values -- speculative decoding then drafts garbage that the target rejects
    every time, which is a silent throughput regression, not an error.
    """
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForCausalLM,
        InklingForConditionalGeneration,
    )

    for cls in (InklingForCausalLM, InklingForConditionalGeneration):
        assert "load_weights" in vars(cls), f"{cls.__name__} does not override load_weights"
        assert "_load_mtp_weights" in inspect.getsource(vars(cls)["load_weights"]), (
            f"{cls.__name__}.load_weights never loads the draft chain"
        )


# --- one-engine speculative plumbing ---------------------------------------
# The draft chain, its KV cache and the verify-step kernels are all necessary
# and none of them is what makes the framework RUN speculative decoding. That
# is the base class: it builds the draft model, creates the spec worker, and
# routes the forward through it.


def test_causal_lm_is_a_one_engine_spec_model():
    """Otherwise nothing builds the draft model or the spec worker.

    Bolting a draft chain onto a plain DecoderModelForCausalLM produces a model
    that loads, allocates a draft KV cache, and then returns flat logits the
    speculative sampler cannot index -- an IndexError in HandleLogits naming
    neither speculation nor Inkling.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM
    from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM

    assert issubclass(InklingForCausalLM, SpecDecOneEngineForCausalLM)


def test_logits_are_taken_at_the_gather_ids_when_speculating():
    """A verify step needs one logit row per verified position, not per token.

    ``spec_metadata.gather_ids`` is what selects them; passing the full hidden
    states hands the sampler a batch of the wrong length.
    """
    src = inspect.getsource(
        __import__(
            "tensorrt_llm._torch.models.modeling_inkling", fromlist=["x"]
        ).InklingForCausalLM.forward
    )
    assert "spec_metadata.gather_ids" in src
    assert "self.spec_worker" in src


def test_the_draft_chain_gets_the_undivided_hidden_states():
    """muP divides the lm_head input, not the residual stream.

    The chain continues the trunk's stream, so dividing before handing it over
    would scale every draft block's input by 1/mup -- wrong numbers, no error,
    and a drafter that simply proposes badly. SGLang passes the undivided
    hidden states for the same reason.
    """
    src = inspect.getsource(
        __import__(
            "tensorrt_llm._torch.models.modeling_inkling", fromlist=["x"]
        ).InklingForCausalLM.forward
    )
    assert "head_input = hidden_states / self.mup_multiplier" in src
    assert "hidden_states=hidden_states," in src


def test_the_draft_input_is_cast_to_the_projection_dtype():
    """RMSNorm can emit fp32; the NVFP4 quantize op refuses it.

    ``fp4_quantize only supports input tensor with dtypes fp16/bf16/e4m3`` is
    what that looks like from inside the first draft forward. The trunk casts at
    the same boundary; the draft block has to as well.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingMTPBlock

    src = inspect.getsource(InklingMTPBlock.forward)
    # The cast target is a real parameter built at the compute dtype. The
    # quantized Linear's weight (packed storage), the config's torch_dtype
    # (declared, not actual) and the incoming hidden states (whatever the spec
    # worker hands over) were each tried on the cluster and each left fp32 in
    # place somewhere downstream.
    assert "combined.to(self.embed_norm.weight.dtype)" in src


def test_the_mtp_head_returns_one_row_per_sequence():
    """MTPWorker samples one draft token per sequence from this.

    It then writes the result back at ``last_tokens_idx``, one index per
    sequence, so a row per TOKEN turns that assignment into "value tensor of
    shape [draft_len] cannot be broadcast to indexing result of shape [1]" --
    several minutes into a multi-GPU run, in the speculative loop, naming
    neither Inkling nor the head. Gathering here also keeps the vocab-sized
    projection off every token of the batch, which is why DeepSeek's MTP head
    does the same.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingMTPHead

    src = inspect.getsource(InklingMTPHead.forward)
    assert "seq_lens_cuda" in src and "hidden_states[last_tokens]" in src


def test_the_head_can_still_return_every_token():
    """``return_context_logits`` keeps the full-batch path available."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingMTPHead

    params = inspect.signature(InklingMTPHead.forward).parameters
    assert "return_context_logits" in params
    assert params["return_context_logits"].default is False


# ---------------------------------------------------------------------------
# What the causal LM hands the spec worker
# ---------------------------------------------------------------------------
# Behavioural rather than signature-level: the two defects below were both
# "the right method was called with the wrong value", which a signature check
# cannot see. The forward is exercised unbound against a stub `self`, so this
# still needs no GPU, checkpoint or built extension.


class _RecordingSpecWorker:
    """Stands in for MTPWorker and keeps what it was called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return "spec-output"


class _StubLogitsProcessor:
    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits):
        return hidden_states


class _StubAttnMetadata:
    def __init__(self, num_tokens, padded_num_tokens=None):
        self.num_tokens = num_tokens
        self.padded_num_tokens = padded_num_tokens


class _StubSpecMetadata:
    def __init__(self, gather_ids):
        self.gather_ids = gather_ids


def _run_causal_lm_forward(*, input_ids, kwargs, padded_num_tokens=None, num_tokens=4):
    """Drive InklingForCausalLM.forward against a stub self, return the worker."""
    import torch

    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    total_rows = padded_num_tokens if padded_num_tokens is not None else num_tokens
    worker = _RecordingSpecWorker()

    class _Stub:
        mup_multiplier = 2.0
        spec_worker = worker
        logits_processor = _StubLogitsProcessor()
        lm_head = None
        draft_model = None

        def model(self, **_kw):
            return torch.arange(total_rows, dtype=torch.float32).unsqueeze(1)

    stub = _Stub()
    InklingForCausalLM.forward(
        stub,
        _StubAttnMetadata(num_tokens, padded_num_tokens),
        input_ids=input_ids,
        position_ids=torch.arange(total_rows, dtype=torch.int32),
        spec_metadata=_StubSpecMetadata(torch.tensor([0])),
        **kwargs,
    )
    assert len(worker.calls) == 1
    return worker.calls[0]


def test_the_worker_gets_the_pre_fusion_ids_on_a_multimodal_request():
    """``fuse_input_embeds`` returns input_ids as None; the worker subscripts it.

    On a request carrying an image the token stream becomes an embedding
    stream, so ``input_ids`` arrives here as None. MTPWorker does
    ``input_ids[:num_ctx_tokens]`` in prepare_drafter_inputs, which is a
    TypeError -- Inkling is the only MTP model that is also multimodal, so
    nothing else exercises this. The wrapper forwards the pre-fusion ids under
    ``orig_input_ids`` and this is where they are picked back up.
    """
    import torch

    orig = torch.arange(4, dtype=torch.int32)
    call = _run_causal_lm_forward(input_ids=None, kwargs={"orig_input_ids": orig})
    assert call["input_ids"] is not None, (
        "the spec worker was handed input_ids=None; it subscripts them"
    )
    assert torch.equal(call["input_ids"], orig)


def test_real_input_ids_win_over_the_multimodal_fallback():
    """A text-only request must not be rerouted through the fallback."""
    import torch

    real = torch.arange(10, 14, dtype=torch.int32)
    stale = torch.zeros(4, dtype=torch.int32)
    call = _run_causal_lm_forward(input_ids=real, kwargs={"orig_input_ids": stale})
    assert torch.equal(call["input_ids"], real)


def test_padding_rows_do_not_reach_the_spec_worker():
    """Padded rows are scratch the batch was rounded up to, not tokens.

    ``padded_num_tokens`` set means hidden_states / input_ids / position_ids all
    carry rows past ``num_tokens``. The worker indexes by request and would read
    them as real; the one-engine base trims all three, and this override has to
    repeat that rather than inherit it.
    """
    import torch

    ids = torch.arange(8, dtype=torch.int32)
    call = _run_causal_lm_forward(input_ids=ids, kwargs={}, padded_num_tokens=8, num_tokens=5)
    assert call["input_ids"].shape[0] == 5
    assert call["position_ids"].shape[-1] == 5
    assert call["hidden_states"].shape[0] == 5


# --- the speculative-decoding guard, called rather than read ----------------
# It has five raises and had only source-text coverage. Each one stands for a
# failure that is otherwise silent or lands far from its cause, and the first
# thing it must do is stay out of the way of a server that is not speculating.


def _spec_guard_config(
    *, depths=8, draft_len=3, cuda_graph=False, vanilla=True, relaxed_thinking=False
):
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    text = InklingConfig(
        text_config={"num_hidden_layers": 42},
        mtp_config=(
            {"num_nextn_predict_layers": depths, "local_layer_ids": [0, 2]} if depths else None
        ),
    ).text_config
    spec_config = MTPDecodingConfig(
        max_draft_len=draft_len,
        use_relaxed_acceptance_for_thinking=relaxed_thinking,
    )
    # What the resolver would have set; done by hand so each case is explicit.
    spec_config.num_nextn_predict_layers = depths if vanilla else 1
    spec_config.use_mtp_vanilla = bool(vanilla)

    class _Cfg:
        pretrained_config = text
        use_cuda_graph = cuda_graph

    cfg = _Cfg()
    cfg.spec_config = spec_config
    return cfg


def test_the_spec_guard_is_a_no_op_without_speculation():
    """An ordinary server must not be refused by any of this."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    class _Cfg:
        pretrained_config = InklingConfig(text_config={}).text_config
        spec_config = None
        use_cuda_graph = True  # irrelevant when not speculating

    assert InklingForCausalLM._assert_inkling_spec_conv_state(_Cfg()) is None


def test_a_valid_speculative_setup_passes_the_guard():
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    assert InklingForCausalLM._assert_inkling_spec_conv_state(_spec_guard_config()) is None


def test_a_checkpoint_without_a_chain_is_named_as_such():
    """The framework reads the depth as a bare attribute; absent is an
    AttributeError from inside it, which says nothing about the checkpoint."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    with pytest.raises(ValueError, match="declares no MTP chain"):
        InklingForCausalLM._assert_inkling_spec_conv_state(_spec_guard_config(depths=None))


def test_a_non_vanilla_mode_is_refused():
    """EAGLE builds one block and replays it; Inkling's depths are distinct."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    cfg = _spec_guard_config(vanilla=False)
    with pytest.raises(ValueError, match="needs vanilla MTP"):
        InklingForCausalLM._assert_inkling_spec_conv_state(cfg)


def test_attention_dp_with_speculation_is_refused():
    """Measured: either feature alone works, together the engine asserts.

    ``assert len(num_kv_heads) == self.num_layers`` three frames inside
    KVCacheManagerV2, naming neither number, while building the TARGET manager
    (jobs 6339819/6340465/6340809, with single-feature arms passing in the same
    allocation). The refusal replaces that with a statement; it does not fix the
    underlying mismatch, which is not yet diagnosed.
    """
    from types import SimpleNamespace

    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    cfg = _spec_guard_config()
    cfg.mapping = SimpleNamespace(enable_attention_dp=True)
    with pytest.raises(ValueError, match="attention DP"):
        InklingForCausalLM._assert_inkling_spec_conv_state(cfg)


def test_attention_dp_alone_is_untouched():
    """The guard must fire only when speculating; attention DP on its own is
    supported and was measured working in the same job."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    class _Cfg:
        pretrained_config = InklingConfig(text_config={}).text_config
        spec_config = None
        use_cuda_graph = False
        mapping = SimpleNamespace(enable_attention_dp=True)

    assert InklingForCausalLM._assert_inkling_spec_conv_state(_Cfg()) is None


def test_relaxed_acceptance_for_thinking_is_refused():
    """It is lossy, and it is gated on tokens Inkling does not have.

    ``begin/end_thinking_phase_token`` default to DeepSeek-R1's 128798/128799.
    Those are the ids relaxed acceptance uses to decide where it may relax, and
    in Inkling's vocabulary they are ordinary tokens -- so left alone the mode
    relaxes acceptance in the wrong places and never says so.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    with pytest.raises(ValueError, match="relaxed acceptance"):
        InklingForCausalLM._assert_inkling_spec_conv_state(
            _spec_guard_config(relaxed_thinking=True)
        )


def test_the_thinking_phase_defaults_are_not_inkling_tokens():
    """Why the refusal above exists, pinned against the shipped defaults.

    If a future release makes these configurable per model -- or Inkling's
    tokenizer grows a paired think/end-think -- this is the test that should
    fail and prompt revisiting the refusal rather than leaving it in place.
    """
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    cfg = MTPDecodingConfig(max_draft_len=3)
    # DeepSeek-R1's <think>/</think>, carried as the framework-wide default.
    assert (cfg.begin_thinking_phase_token, cfg.end_thinking_phase_token) == (128798, 128799)
    # Inkling opens thinking with <|content_thinking|> and closes it by
    # switching channel, so neither id names anything in its alphabet.
    from tensorrt_llm.llmapi.inkling_tokens import INKLING_CONTENT_THINKING

    assert INKLING_CONTENT_THINKING == "<|content_thinking|>"


def test_cuda_graphs_are_refused_at_construction():
    """The verify step walks drafted positions one at a time and cannot be
    captured; the backend's own raise lands inside warmup, minutes later."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    with pytest.raises(ValueError, match="cannot run with CUDA graphs"):
        InklingForCausalLM._assert_inkling_spec_conv_state(_spec_guard_config(cuda_graph=True))


def test_a_zero_draft_length_is_refused_by_the_config_not_here():
    """The invariant lives on MTPDecodingConfig, which is closer to the user.

    Inkling carried its own ``max_draft_len < 1`` check -- the conv capture
    buffers are sized from it -- but it is unreachable: the Pydantic validator
    rejects the value at construction, before any model exists, and says so in
    better words. Pinning where the check actually lives so the duplicate is not
    reintroduced.
    """
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    with pytest.raises(ValueError, match="max_draft_len must be > 0"):
        MTPDecodingConfig(max_draft_len=0)
