---
name: trtllm-modeling-v2-multimodal
description: Load when a target must accept images or other media alongside text — an extra input path with its own tower, projector and weights, plus an input processor its family's routing module hands the launcher. Gated on MMMU-Pro rather than a text benchmark. Carried end to end by two targets.
---

# trtllm-modeling-v2-multimodal

`--features multimodal`. Two targets have carried this end to end, neither in
this tree: a Qwen3.6-35B-A3B NVFP4 tp1 target with a 27-block ViT on sm_100,
which the gate section below and most of `references/vision-path.md` measure,
and a second vision-language target on sm_103, which measured the processor hand-off
through the routing module and the splice under block reuse. They are
measurements on *those* checkpoints' shapes, not laws.

| reference | |
|---|---|
| `references/vision-path.md` | the extra input path: tower, projector, weights, and the processor hand-off — now written as answers with evidence, including the traps below that cost the most |

## Why it is a feature and not core

A multimodal checkpoint can be assembled **text-only**, and that is the
default: `--features` empty means the vision weights go into the
known-not-loaded key set and the target serves text. That is a legitimate,
gateable target — so implementing the vision path is a choice, which is what
makes it a feature rather than something the checkpoint's `config.json`
forces.

## What it changes

It is the only feature that adds an **input path** rather than changing the
forward's math:

- a second weight family — the tower and the projector — with its own layout
  conventions, usually unquantized while the language model is not
- construction of a second module tree in `__init__`
- an input processor the launcher builds, where media arrive already decoded
  and batched, handed over by the family's routing module
- image tokens spliced into the sequence before the language model runs

Three things here turned out to have a shape worth naming up front, because
getting any of them wrong produces a **passing-looking engine with a dead
vision path**:

- **the processor goes through the routing module, not
  `@register_input_processor`.** `create_input_processor` finds a decorated
  processor through the model class the built-in zoo maps the architecture
  to, and a modeling_v2 target is never in the zoo: the decorator registers a
  processor nothing asks for, and the launcher builds the text-only default.
  The family's routing module defines `input_processor(pretrained_config)`
  beside `route`, returning a `RoutedInputProcessor`; where the tree has no
  such hook, adding it is an engine task of its own. Measured on that
  target (sm_103); `references/vision-path.md`, answer 2, has the shape.
- **the splice belongs in the shell, as `inputs_embeds`.** The inherited
  `DecoderModelForCausalLM.forward` forwards only the keywords it names, so
  `multimodal_params` riding in `**kwargs` is silently dropped and the model's
  own forward never sees it.
- **give the boot gate images that differ from each other.** A dead splice
  answers every image identically; one image case passes anyway.

And one thing that is not an input path at all: **a VLM whose text model uses
mRoPE runs a degenerate mRoPE on text** (all three position rows equal), so a
text-only assembly may legitimately take row 0 and run plain RoPE. Turning the
tower on breaks that, and the rotation has to read all three rows. Branch on
the per-engine feature constant so the text path stays bit-identical.

## Gates

boot → accuracy on **MMMU-Pro**, both with `modeling_v2=require`.
**No third gate.**

This feature is not blind to the first two: get the vision path wrong and the
output is wrong. There is no distribution-preserving construction hiding the
defect, and no resource-distribution value a correct-output run could be
missing.

What it does need is a benchmark that **looks at the image**. MMLU and GSM8K
are text; they pass at full score with the vision tower producing garbage,
because nothing in them reads a pixel — and that is not a hypothetical: the
first engine run of the one target that has done this had a dead splice, and
its text gates would have passed untouched. So the accuracy case for a
multimodal configuration reads its anchor from the image benchmark's file
under `tests/integration/defs/accuracy/references/`, and the journal says
plainly that the text benchmark's score is evidence about the language model
only.

**A target that serves both text and images needs two gates, not one**, and
re-pointing one gate's task at the image benchmark would silently re-gate the
text configurations — including runs already recorded. Write them as **two
cases** in the family's accuracy file
(`tests/integration/defs/accuracy/test_modeling_v2_<family>.py`): the
multimodal case evaluates MMMU-Pro, the text cases keep their text benchmark,
and each reads its own anchor because the reference files are keyed by
`TestClass::test_method`. Name the case after the configuration it gates.

**Read how it is scored before you trust a number.** `trtllm-eval mmmu` and
`mmmu_pro` are first-class subcommands, registered in
`tensorrt_llm/evaluate/lm_eval.py` — go read that command: how the dataset is
loaded, what is fed to the model, which metric is reported and which filter
selects it. Nothing stops you, and a gate you cannot explain is a gate you
cannot debug. The GSM8K precedent is the reason this is not optional: the same
run read 25.47 under the strict filter and 90.60 under `flexible-extract`, and
the low number looked exactly like a broken model.

**`mmmu` did not run at lm_eval 0.4.12 — use `mmmu_pro`.** Established by
running it against **stock trtllm** on a real checkpoint, which is the control
that says whose bug it is:

```
AttributeError: 'dict' object has no attribute 'build_all_requests'
  lm_eval/evaluator.py:537  task.build_all_requests(...)
```

The hub's `mmmu_val` is a **doubly** nested group — `mmmu_val` → one group per
discipline → the tasks. lm-eval 0.4.12's legacy-format shim
(`evaluator_utils.py:_handle_back_comp`) unwraps **one** level: for a group key
it does `tasks.update(value)`, so the discipline groups land in `tasks` where
Task objects are expected, and the first `build_all_requests` raises. Nothing
model-side is reached, so this fails identically on any model.

`mmmu_pro` is the way through and is a first-class subcommand: trtllm ships its
own task yamls under `evaluate/lm_eval_tasks/mmmu_pro/`, and
`get_task_dict(['mmmu_pro_standard_4'])` returns a **flat** `str ->
ConfigurableTask`. `--subset standard_10` (the default, 10 options) or
`standard_4`. Either is a valid gate as long as the reference and the target are
measured on the same one — the gate is a comparison, and an anchor measured here
is measured on whichever you pick.

*Verified at lm_eval 0.4.12. Re-check by running `mmmu` against stock trtllm
first: if it still raises there, it is still upstream's.*

**And KV-cache block reuse needs the images' content keys.** With reuse on —
which is the **default** — a request's blocks are keyed on its images'
content, and the KV-cache manager builds those keys only when the engine
knows it serves a multimodal model: it then hands the manager the vocabulary
size (`tensorrt_llm/_torch/pyexecutor/_util.py`). Where it does not, the
first multimodal request dies in the executor:

```
RequestError: gen_multimodal_cache_key_tokens(): incompatible function arguments
  Invoked with types: NoneType, bytes, int, kwargs = { token_offset: int }
```

`id_offset` arrives `None` where an `int` is required, from the
`kv_cache_manager_v2` path that hashes multimodal content into a reuse key.
Reproduced on **stock trtllm** on its own model, and it is why every VL test
in trtllm's own accuracy suite constructs
`KvCacheConfig(..., enable_block_reuse=False)` rather than taking the default.

A processor handed over through the routing module is a
`BaseMultimodalInputProcessor`, so the engine knows, and the hand-off pins
content hashing on: an image it cannot hash fails rather than reaching reuse
unkeyed. That target served its gates that way with reuse on:
one image keyed alike twice, another differently, and a repeated image
resumed from the reused prefix inside its own span. Consequences a target
carries rather than discovers:

- with reuse on, a context chunk can start or end inside an image, so the
  splice places the image rows the chunk covers and no others
  (`references/vision-path.md`, answer 4);
- a target without the hand-off serves images with reuse off: the gate
  invocation needs `--disable_kv_cache_reuse` (trtllm-eval) /
  `enable_block_reuse: false` (LLM-API), the multimodal configuration sets it
  too, and the journal says why, because the default is then the broken
  value and a future reader will otherwise "clean it up".

**And the gate needs an explicit `scores_filter`, or scoring raises.** Without
one the evaluator averages every metric the task reports — including lm-eval's
`alias`, which is a string — and numpy raises

```
TypeError: the resolved dtypes are not compatible with add.reduce.
Resolved (dtype('<U32'), dtype('<U32'), dtype('<U64'))
```

*after* the whole eval has run, which is the expensive place to find out. The
mmmu_pro task's metric is `acc`, so the filter is **`acc,none`**. `trtllm-eval`
exposes no CLI flag for it; the accuracy case binds it onto the task's
evaluate kwargs before running — the same way the modeling_v2 deepseek case
binds `exact_match,flexible-extract` for GSM8K — and a hand-rolled reference
run must bind it the same way.

This is the general trap the repo already records for GSM8K, reaching
multimodal for the same reason: the result dict carries a string alias.

*Verified on sm_100.*

**The gate has now been run end to end.** The qwen3.6-35b-a3b-nvfp4 / sm_100 /
tp1 target measured **65.3179 ± 1.1446** on `mmmu_pro_standard_10` (n = 1730)
against a stock-trtllm anchor of 62.8324 and a threshold of 57.83 — 1689 s of
engine execution, about 32 minutes wall on one B200. Two practical notes from
that run:

- **`trtllm-eval` fetches responses in submission order**, so its progress bar
  sits near zero for most of the run and then jumps to the full count. At the
  30-minute mark its ETA read *12 hours* on a run that was 2 minutes from
  finishing. Do not size a timeout, or a decision to restart, from that bar.
- a reasoning model under this task's `max_output_length` of 32000 writes long
  chains (~4200 output tokens on a sample of eight documents, all stopping on
  their own). That is the anchor's protocol too — leave it alone.

Two things worth knowing before you start:

- **the threshold is enforced on our side, not by the subcommand.** `mmmu`
  carries no `--check_accuracy`; the workflow's accuracy gate parses the score
  and compares it with the floor it derives from the reference file, for every
  benchmark alike
- **images do traverse the hand-off** — verified by a passing gate.
  `MultimodalLmEvalWrapper` builds the prompt itself and submits
  `multi_modal_data={"image": [...]}` to the serving endpoint; lm-eval's own
  VLM adapters (`hf_vlms.py`) are not on the path
- **prompt construction is part of the protocol and it moves the score.** The
  wrapper reads `model_type` from the checkpoint's `config.json` and looks the
  registered `MultimodalPlaceholderMetadata` up by it; bulk-prepended versus
  interleaved placeholders is a different prompt for every MMMU-Pro document.
  If the anchor was measured on stock trtllm, register the same values the
  stock model_type carries — readable without touching model code through
  `MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_metadata(...)`

Record the anchor's source in the reference entry as usual — an MMMU score is
never invented, and if no published number for this checkpoint exists, measure
stock trtllm on it and mark the source accordingly.

## Measure the blind baseline before you trust the threshold

The gate is `anchor - tol`, and that bar is only meaningful if an engine which
**ignores its images** falls below it. Agreement with stock trtllm does not
establish this: it shows two systems match, not that either reads a pixel.

Measure it. Patch the task's image supplier so every image becomes a flat grey
of the same size — count, placeholder tokens and sequence lengths unchanged, only
the pixel information removed — and re-run the same configuration over the same
documents.

Measured once, on Qwen3.6-35B-A3B / MMMU-Pro standard_10, 1730 docs: **normal
63.06, blind 33.24**, against a threshold of 57.83. The gate discriminates by
~26 combined standard errors.

**The blind score is the number worth having, and it is not the random
baseline.** Random on ten options is 10; blind measured **33**. A text-only
reader gets real signal from the question stem, so "our score is far above
random, therefore we see the image" is not an argument — the honest contrast is
against the blind score. On a checkpoint where blind lands near the threshold,
the gate is not discriminating and needs a harder benchmark or a tighter
tolerance, and you would never learn that from a passing run.

Verify the control itself: count the images you replaced and fail loudly if the
patch bound nothing. A silently-inert control returns the normal score and reads
as proof.

*Measured on sm_100; the blind figure is a property of the checkpoint and the
benchmark, so re-measure it per target rather than carrying this one across.*

## Report additions

- everything you had to establish by observation, named precisely — this skill
  is speculative until a target fills it in, and the report is how that
  happens
- the benchmark chosen and why it exercises the vision path
