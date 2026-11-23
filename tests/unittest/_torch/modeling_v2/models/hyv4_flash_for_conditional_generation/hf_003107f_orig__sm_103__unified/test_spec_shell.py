# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one-engine DFly2 shell and its worker's step bookkeeping, on CPU.

The engine reaches a one-engine speculative model through a fixed set of
attributes and hands the worker the step's own metadata. `test_draft.py` and
`test_draft_module.py` hold the drafter against the checkpoint; this module
holds the two things around it:

* the shell's contract with the loader, the engine and the post-load walk --
  which attributes `_build_drafter` must set, what `load_draft_weights` must
  forward, and which configurations are refused;
* the worker's block bookkeeping -- the column convention its drafts are drawn
  from, the gathered logits it verifies through, and the sequence lengths its
  block's forward and the next target step both read.

What this module cannot see: any device or kernel, so the block's native
preparation and attention, and whether the engine hands the worker the block it
planned, are the acceptance gate's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

# The checkpoint stand-in this directory's tests share; pytest puts the test
# file's own directory on the path.
from _target import core, drafter, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.cache_manager import (
    Hyv4FlashCacheManager,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified import (  # noqa: E501
    spec_worker as spec_worker_module,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.draft import (  # noqa: E501
    DRAFT_KEY_PREFIX,
    Hyv4FlashDraft,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.modeling import (  # noqa: E501
    ModelingV2Hyv4FlashForConditionalGeneration,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.spec.dfly_block_prepare import (  # noqa: E501
    BLOCK_WIDTH,
    MASK_TOKEN_ID,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.spec_worker import (  # noqa: E501
    Hyv4FlashDFlashWorker,
)
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.speculative import should_use_separate_draft_kv_cache
from tensorrt_llm._torch.speculative.dflash import DFlashWorker, dflash_draft_slot_ids
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

pytestmark = requires_checkpoint


def spec_config(**over) -> DFlashDecodingConfig:
    """The gates' own DFly2 config: five drafts, the checkpoint's mask token."""
    fields = dict(decoding_type="DFlash", max_draft_len=5, mask_token_id=MASK_TOKEN_ID)
    fields.update(over)
    return DFlashDecodingConfig(**fields)


class _Shell:
    """The shell's own state at the point `_build_drafter` runs.

    `Hyv4FlashCore` declares the checkpoint's whole weight table, so the shell
    cannot be built without a device's worth of memory; what the wiring reads is
    these attributes. The engine's boot gate proves the built shell sets them.
    """

    def __init__(self, spec_config, stand_in):
        self.spec_config = spec_config
        self.model = stand_in
        # Only their identity is read here: the drafter draws its drafts through
        # the target's own head and logits processor.
        self.logits_processor = LogitsProcessor()
        self.lm_head = object()
        self.epilogue = []
        self.draft_model = None
        self.draft_config = None
        self.spec_worker = None
        # Deliberately the opposite of what the wiring computes: the shell must
        # decide this, not inherit whatever a caller left behind.
        self.use_separate_draft_kv_cache = True


class _ModuleShell(_Shell, torch.nn.Module):
    """The same state as a real `nn.Module`, which is what registers children.

    `_Shell` is a plain object, so a drafter assigned to it is a plain
    attribute; the engine's shell is an `nn.Module`, where `self.draft_model =
    drafter` makes the drafter a submodule -- and a reference back up to the
    shell from the drafter a module cycle.
    """

    def __init__(self, spec_config, stand_in):
        torch.nn.Module.__init__(self)
        _Shell.__init__(self, spec_config, stand_in)


def built(spec_config) -> _Shell:
    """One shell with the drafter and the worker `_build_drafter` builds under it."""
    stand_in = core()
    stand_in.model_config.torch_dtype = torch.bfloat16
    shell = _Shell(spec_config, stand_in)
    ModelingV2Hyv4FlashForConditionalGeneration._build_drafter(shell, stand_in.model_config)
    return shell


def built_module_shell(spec_config) -> _ModuleShell:
    """`built`'s wiring on the `nn.Module` shell, so submodules are registered."""
    stand_in = core()
    stand_in.model_config.torch_dtype = torch.bfloat16
    shell = _ModuleShell(spec_config, stand_in)
    ModelingV2Hyv4FlashForConditionalGeneration._build_drafter(shell, stand_in.model_config)
    return shell


class _StepLengths:
    """A step's length tensors, as the engine's attention metadata holds them.

    `aliased` is the shape the metadata has while it runs and the split one a
    warmup clone leaves behind; the worker must write both.
    """

    def __init__(self, lengths, num_contexts: int, aliased: bool = True) -> None:
        self.kv_lens_cuda = torch.tensor(lengths, dtype=torch.int32)
        self.kv_lens_cuda_runtime = self.kv_lens_cuda if aliased else self.kv_lens_cuda.clone()
        self._seq_lens = torch.zeros(len(lengths), dtype=torch.int32)
        self._seq_lens_cuda = torch.zeros(len(lengths), dtype=torch.int32)


class _Recorder:
    """A drafter that records what the loader read into it."""

    def __init__(self) -> None:
        self.calls = []

    def load_weights(self, weights, weight_mapper=None, allow_partial_loading=False) -> None:
        self.calls.append((weights, weight_mapper, allow_partial_loading))


# -- the shell's contract with the engine --------------------------------------


def test_the_engine_surface_the_shell_declares():
    """What the loader, the engine and the post-load walk read off the shell."""
    cls = ModelingV2Hyv4FlashForConditionalGeneration
    # `ModelEngine` reads this to decide the vocabulary-shard fast paths; False
    # keeps the acceptance on the gathered target logits, which is what the
    # worker's verification needs.
    assert cls.spec_decode_fast_paths is False
    assert cls.runtime_hooks.kv_cache_manager_cls is Hyv4FlashCacheManager
    for name in ("_build_drafter", "load_draft_weights", "set_guided_decoder"):
        assert callable(getattr(cls, name)), name


def test_the_shell_forwards_the_draft_read_and_the_guided_decoder():
    """The call shapes `model_loader` and `ModelEngine` use."""
    cls = ModelingV2Hyv4FlashForConditionalGeneration
    drafter_recorder = _Recorder()
    shell = SimpleNamespace(draft_model=drafter_recorder, spec_worker=None)
    weights = {"model.dfly_layers.0.q_a_proj.weight": 1}
    cls.load_draft_weights(shell, weights, "the target's mapper")
    assert drafter_recorder.calls == [(weights, "the target's mapper", False)]
    # Without a worker there is nothing to mask, and the engine takes False.
    assert cls.set_guided_decoder(shell, object()) is False
    shell.spec_worker = SimpleNamespace(set_guided_decoder=lambda guided: True)
    assert cls.set_guided_decoder(shell, object()) is True


def test_build_drafter_refuses_a_config_that_is_not_a_one_engine_dflash():
    """An MTP/Eagle3 one-engine config is not this drafter, and says so."""
    stand_in = core()
    stand_in.model_config.torch_dtype = torch.bfloat16
    shell = _Shell(
        SimpleNamespace(
            spec_dec_mode=SpeculativeDecodingMode.EAGLE3_ONE_MODEL,
            uses_external_draft_model=True,
        ),
        stand_in,
    )
    with pytest.raises(ValueError, match="serves it in one engine"):
        ModelingV2Hyv4FlashForConditionalGeneration._build_drafter(shell, stand_in.model_config)


def test_build_drafter_refuses_a_config_that_asks_for_a_second_draft_cache():
    """This drafter's block lives in the target's pools, so TRTLLM attention is refused."""
    config = spec_config(attention_backend="TRTLLM")
    assert should_use_separate_draft_kv_cache(config) is True, "the premise"
    with pytest.raises(ValueError, match="keeps no KV cache of its own"):
        built(config)


def test_build_drafter_wires_the_drafter_and_the_worker_it_runs():
    """The attributes the one-engine speculative path reads off the model."""
    shell = built(spec_config())
    drafter_model, worker = shell.draft_model, shell.spec_worker
    assert isinstance(drafter_model, Hyv4FlashDraft)
    assert isinstance(worker, Hyv4FlashDFlashWorker)
    # The draft read has no architecture of its own to resolve a mapper from:
    # the target's mapper with the drafter's key prefixes narrows the second read.
    assert shell.draft_config is None
    assert drafter_model.checkpoint_key_prefixes == (DRAFT_KEY_PREFIX,)
    assert drafter_model.logits_processor is shell.logits_processor
    assert drafter_model.lm_head is shell.lm_head
    # Both are registered on the shell as submodules, so the loader's post-load
    # walk reaches their `post_load_weights`; `epilogue` is the same pair as a
    # plain list (registered attributes are not reachable through it).
    for name in ("draft_model", "spec_worker"):
        assert getattr(shell, name) is not None, name
    assert shell.epilogue == [drafter_model, worker]
    assert shell.use_separate_draft_kv_cache is False
    assert worker._draft_model_ref() is drafter_model
    assert worker._target_model_ref() is shell
    # One block per step: five drafts plus the bonus row the target verified.
    assert (worker.max_draft_len, worker._draft_tokens_per_req) == (5, BLOCK_WIDTH)
    assert worker._resolve_mask_token_id(drafter_model) == MASK_TOKEN_ID


def test_the_drafters_module_tree_does_not_cycle_back_to_the_shell():
    """The engine's weight materialization walks children with no memo.

    `memory_init.init_meta_tensors_on_cuda` runs `model._apply(...)` and the
    line after it `model.to("cuda")`; both recurse over `children()`, so a
    module cycle raises `RecursionError: maximum recursion depth exceeded` at
    engine creation, before a single weight is read.
    """
    shell = built_module_shell(spec_config())
    drafter = shell.draft_model
    assert drafter is not None, "the premise: the drafter is a submodule here"
    # The property that walk needs: nothing under the drafter leads back up.
    assert shell not in set(drafter.modules())
    # And the walk itself terminates, over both the drafter and the worker.
    shell._apply(lambda tensor: tensor)


def test_the_drafter_registers_its_weights_in_one_table():
    """The shell's unfed-parameter check excuses exactly this one prefix.

    `Hyv4FlashUnifiedWeights.load` reads the shell's own parameters (lm_head) and
    the core's table, and takes `draft_model.w.*` as fed by the *second* read --
    `load_draft_weights`, whose `DRAFT_WEIGHTS.load` checks that table's own
    coverage. That is only sound while the drafter declares every weight it owns
    in that one dict: a parameter anywhere else under it is excused by the shell
    and read by nobody.
    """
    shell = built_module_shell(spec_config())
    names = {name for name, _ in shell.named_parameters()}
    assert names, "the premise: the drafter declares parameters here"
    assert all(name.startswith("draft_model.w.") for name in names), sorted(names)


def test_the_second_read_is_narrowed_to_the_drafters_keys():
    """The loader narrows the target's own directory with these prefixes.

    `select_draft_weights` keeps the keys under them and drops the rest; the
    drafter's table is what decides, so this is the pair the loader reads.
    """
    from tensorrt_llm._torch.models.checkpoints.model_hooks import select_draft_weights

    weights = {
        "model.embed_tokens.weight": 0,
        "model.dfly_layers.0.q_a_proj.weight": 1,
        "model.layers.41.mlp.gate.weight": 2,
    }
    kept = select_draft_weights(weights, (DRAFT_KEY_PREFIX,))
    assert set(kept) == {"model.dfly_layers.0.q_a_proj.weight"}


# -- the worker's block bookkeeping --------------------------------------------


def test_the_worker_draws_the_blocks_own_columns():
    """Column `s`'s corrected hidden draws column `s`'s own token: columns 1..K.

    Column 0 is the anchor, whose token the target has already produced, so no
    draft is drawn from it. The worker states this rather than leaving it to
    inheritance, since the engine's other family with a `shift_label`
    convention reads columns 0..K-1.
    """
    assert Hyv4FlashDFlashWorker._draft_slot_ids is not DFlashWorker._draft_slot_ids
    num_gens, num_drafts = 2, 5
    ids = dflash_draft_slot_ids(num_gens, BLOCK_WIDTH, num_drafts, False, device="cpu")
    assert ids.tolist() == [
        request * BLOCK_WIDTH + column
        for request in range(num_gens)
        for column in range(1, num_drafts + 1)
    ]


def test_the_worker_asks_for_those_columns(monkeypatch):
    """The worker's own call: the block's columns 1..K, its width and its draft count."""
    worker = built(spec_config()).spec_worker
    seen = {}

    def record(num_gens, block_size, num_draft_tokens, shift_label, device="cuda"):
        seen.update(
            num_gens=num_gens,
            block_size=block_size,
            num_draft_tokens=num_draft_tokens,
            shift_label=shift_label,
        )
        return torch.zeros(0, dtype=torch.long)

    monkeypatch.setattr(spec_worker_module, "dflash_draft_slot_ids", record)
    worker._draft_slot_ids(None, 3, BLOCK_WIDTH, 5)
    assert seen == dict(num_gens=3, block_size=BLOCK_WIDTH, num_draft_tokens=5, shift_label=False)


def test_the_walk_pairs_each_column_with_the_token_before_it():
    """Step `s`: column `s + 1`'s hidden, corrected with column `s`'s token.

    A row's hidden carries its own column number, and a step's logits draw
    exactly that number, so the recorded pairs say which row and which
    conditioning token the walk used: the anchor first, and then the token the
    step before it drew -- one step per column, in order.
    """
    worker = built(spec_config()).spec_worker
    num_gens, num_drafts = 2, 5
    keywords = 8
    # The gather's own order: request-major, columns 1..K.
    rows = torch.zeros(num_gens * num_drafts, keywords)
    rows[:, 0] = torch.arange(1, num_drafts + 1).repeat(num_gens)
    pairs = []

    class _Numbered:
        """Rows carry their column; a step draws the column of the row it read."""

        def hidden_correction(self, hidden, previous):
            pairs.append((hidden[:, 0].tolist(), previous.tolist()))
            return hidden, None

        def block_logits(self, corrected, attn_metadata):
            del attn_metadata
            return torch.nn.functional.one_hot(corrected[:, 0].long(), keywords).float()

    worker.greedy_sample_draft_with_tp_gather = lambda logits, spec_metadata: logits.argmax(dim=-1)
    tokens, vocab = worker._sample_block_draft_tokens(
        _Numbered(),
        torch.zeros(num_gens, num_drafts, keywords),
        rows,
        {
            "first_prev_tokens": torch.tensor([100, 200]),
            "block_placement": SimpleNamespace(attn_metadata=None),
        },
        SimpleNamespace(wants_advanced_draft_sampling=False),
        num_contexts=0,
        batch_size=num_gens,
    )
    assert vocab == keywords
    assert tokens.tolist() == [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], "the drawn columns"
    assert pairs == [
        ([1, 1], [100, 200]),
        ([2, 2], [1, 1]),
        ([3, 3], [2, 2]),
        ([4, 4], [3, 3]),
        ([5, 5], [4, 4]),
    ]


def test_the_worker_verifies_through_the_gathered_target_logits():
    """A vocabulary-shard selection is declined: the walk reads the whole vocab."""
    worker = built(spec_config()).spec_worker
    assert worker._vocab_parallel is None
    # A plain processor and a shardable head is exactly the pair the engine's
    # DFlash binds a `VocabParallelSelection` for.
    worker.bind_target_lm_head(torch.nn.Linear(4, 4), LogitsProcessor())
    assert worker._vocab_parallel is None
    assert worker.target_vocab_parallel_head is None
    assert worker._vocab_parallel_draft_tokens(None) is None
    assert worker.verifies_on_target_shard is False
    assert worker.reads_lm_head_dtype_logits is False


def test_the_worker_writes_the_blocks_last_position_into_every_length():
    """`anchors + width`: the rope's own sequence length, and the next step's."""
    worker = built(spec_config()).spec_worker
    meta = _StepLengths([10, 20, 30, 40], num_contexts=2)
    worker._write_block_kv_lens(meta, 2, 4, torch.tensor([100, 200]), BLOCK_WIDTH)
    assert meta.kv_lens_cuda.tolist() == [10, 20, 106, 206]
    assert meta.kv_lens_cuda_runtime.tolist() == [10, 20, 106, 206]
    # The generation rows' own length: the block's width, not the whole sequence.
    assert meta._seq_lens.tolist() == [0, 0, 6, 6]
    assert meta._seq_lens_cuda.tolist() == [0, 0, 6, 6]


def test_the_worker_writes_a_split_runtime_view_too():
    """A warmup clone leaves `kv_lens_cuda_runtime` distinct; both must carry it."""
    worker = built(spec_config()).spec_worker
    meta = _StepLengths([10, 20, 30, 40], num_contexts=2, aliased=False)
    worker._write_block_kv_lens(meta, 2, 4, torch.tensor([100, 200]), BLOCK_WIDTH)
    assert meta.kv_lens_cuda.tolist() == [10, 20, 106, 206]
    assert meta.kv_lens_cuda_runtime.tolist() == [10, 20, 106, 206]


def test_the_worker_defers_no_rewind():
    """The block's length is the step's own end state, so there is nothing to undo."""
    worker = built(spec_config()).spec_worker
    meta = _StepLengths([10, 20], num_contexts=0)
    worker._prepare_kv_for_draft_forward(meta, torch.zeros(2, dtype=torch.int32), 0, 2)
    assert worker._kv_rewind_amount is None
    assert worker._kv_rewind_pending is True
    # A metadata without the alias has no length to set, so no rewind either.
    worker._prepare_kv_for_draft_forward(SimpleNamespace(), torch.zeros(2, dtype=torch.int32), 0, 2)
    assert worker._kv_rewind_amount is None
    assert worker._kv_rewind_pending is False


def test_the_worker_resolves_the_mask_token_the_way_the_engine_asks():
    """The config's value, then the drafter's own, then a clear refusal."""
    drafter_model = drafter()
    configured = built(spec_config()).spec_worker
    assert configured._resolve_mask_token_id(drafter_model) == MASK_TOKEN_ID
    from_drafter = built(spec_config(mask_token_id=None)).spec_worker
    assert from_drafter._resolve_mask_token_id(drafter_model) == MASK_TOKEN_ID
    # A drafter that publishes no mask token either is a config to fix, and says so.
    with pytest.raises(ValueError, match="DFlash requires mask_token_id"):
        built(spec_config(mask_token_id=None)).spec_worker._resolve_mask_token_id(SimpleNamespace())
