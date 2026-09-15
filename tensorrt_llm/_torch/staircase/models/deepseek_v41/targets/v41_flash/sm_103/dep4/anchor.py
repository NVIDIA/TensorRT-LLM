# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rung 1 of this target's reference ladder: the checkpoint's own implementation.

``task.yaml`` ships no ``accuracy_anchor`` for DeepSeek-V4.1-Flash and no
usable one exists: the checkpoint publishes no GSM8K/MMLU score, and stock
TensorRT-LLM cannot serve ``DeepseekV41ForCausalLM`` at all, so the usual
"measure stock trtllm on the same checkpoint" fallback is unavailable too.
The anchor therefore has to be *produced*, by running the checkpoint's own
``inference/`` reference implementation under the protocol the target will
later be measured under.

This module is that run. It is not part of the target's forward and nothing in
the engine imports it: ``tilelang`` and the checkpoint's ``inference/`` package
are not TensorRT-LLM dependencies, so every one of them is imported inside a
function, from a path given on the command line.

Two things it is careful about, both from the plan's risk register:

* **Identical token ids on both sides.** ``trtllm-eval gsm8k`` builds its
  prompts with ``lm_eval``; so does this. The request set is constructed here
  exactly as ``tensorrt_llm.evaluate.lm_eval.LmEvalEvaluator.__init__`` builds
  it at the ``gsm8k`` subcommand's defaults, and every prompt's token ids are
  written to the per-sample export so the two legs can be diffed rather than
  assumed equal.
* **Scoring by the same code.** The generations go back through
  ``lm_eval.evaluate``, so ``exact_match,flexible-extract`` is computed by
  lm-eval's own filters and not by a re-implementation that could disagree.

Run it with one process per model-parallel rank; every rank executes the same
deterministic lm-eval flow so the reference's collectives line up, and rank 0
writes the artifacts.

    torchrun --nproc-per-node 4 anchor.py \\
        --ref-dir   <hf ckpt>/inference \\
        --ckpt-path <converted mp4 dir> \\
        --out       <artifact dir> \\
        [--num-samples 32] [--batch-size 1] [--doc-range 0:220]

Modes, cheapest first. The three that need no GPU and no model --
``--probe-requests``, ``--freeze-fixtures`` and ``--merge`` -- are where the
protocol, the fixtures and the score are actually decided; the GPU modes only
produce generations.

* ``--probe-requests``   build and freeze the request set: prompts, token ids,
                         digest. No model.
* ``--fixtures-only``    the candidate fixed prompts, greedy, one row at a time.
* ``--replay-docs``      named documents at a chosen ``--batch-size``: the
                         control for whether a generation is a property of the
                         model or of how the driver batched it.
* ``--doc-range LO:HI``  generate one range of the frozen request set. Writes
                         generations with no score.
* ``--merge a,b,...``    validate a shard set and score its union in one
                         lm-eval pass. No model.
* ``--freeze-fixtures``  pick the frozen fixtures from measured cross-run
                         agreement. No model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# The protocol. These are `trtllm-eval gsm8k`'s own defaults, restated here so
# that the reference leg and the target leg are the same measurement. A change
# on either side has to change this block too.
# ---------------------------------------------------------------------------
TASK = "gsm8k"
RANDOM_SEED = 0
APPLY_CHAT_TEMPLATE = False
FEWSHOT_AS_MULTITURN = False
SYSTEM_PROMPT = None
MAX_INPUT_LENGTH = 4096  # trtllm-eval --max_input_length
MAX_OUTPUT_LENGTH = 256  # trtllm-eval --max_output_length
SCORES_FILTER = "exact_match,flexible-extract"
TORCH_SEED = 33377335  # inference/generate.py's own seed, kept so this is its run

#: Candidate fixture prompts. Plain completions, no chat template -- the same
#: regime GSM8K's few-shot protocol puts the model in, so a keyword frozen here
#: is evidence about the path the gate exercises.
#:
#: This is the *candidate* list, not the frozen set. Greedy decoding on this
#: stack is not reproducible across physical nodes: nine batch-1 runs of the
#: first five prompts on eight distinct GB300 nodes produced 3, 2, 2, 1 and 7
#: distinct 48-token continuations respectively. A near-tie at one position
#: forks the continuation, and the 4-rank MoE/all-reduce reduction order is not
#: bitwise stable between nodes. So which of these become fixtures is decided by
#: ``--freeze-fixtures`` from measured agreement across independent runs, never
#: by picking them here.
FIXTURE_PROMPTS = [
    "Question: There are 15 trees in the grove. Grove workers will plant trees "
    "in the grove today. After they are done, there will be 21 trees. How many "
    "trees did the grove workers plant today?\nAnswer:",
    "The capital city of France is",
    "Water freezes at a temperature of",
    "Question: What is 17 multiplied by 24?\nAnswer:",
    "The first four prime numbers are 2, 3, 5, and",
    "Question: If there are 3 cars in the parking lot and 2 more cars arrive, "
    "how many cars are in the parking lot?\nAnswer:",
    "Question: What is 100 minus 37?\nAnswer:",
    "The chemical symbol for gold is",
    "Question: Sam had 12 eggs and used 5 of them. How many eggs does he have left?\nAnswer:",
]
FIXTURE_MAX_NEW_TOKENS = 48

#: Fixtures are generated one prompt at a time, never batched. Jobs 723978 /
#: 724075 / 724100 / 724124 measured the reference's batched ``generate()``
#: corrupting rows whose generation starts near the batch's shared prefill
#: boundary -- eight prompts of identical length at batch 8 all collapsed into
#: repetition loops. The one batch-5 fixture artifact taken before that was
#: understood (``fixtures.json``, job 723957) disagrees with all eight batch-1
#: runs on two of the five prompts, which is that same defect. A fixture is
#: evidence about the checkpoint, so it is produced the only way this driver
#: produces trustworthy generations.
FIXTURE_BATCH_SIZE = 1

#: Bumped whenever the fixture artifact's meaning changes. ``--freeze-fixtures``
#: refuses artifacts that do not carry it, which is what keeps the pre-batch-1
#: artifacts out of a frozen set.
FIXTURE_RECORD_VERSION = 2

#: Marks a generations-only artifact. ``--merge`` refuses anything else, so a
#: full scored run can never be laundered into a shard set: job 724183 merged
#: the known-bad batch-32 canary and the merged file still called itself
#: "batch size 1, generated in shards", which is exactly the failure this
#: marker plus the equality checks in ``_merge`` exist to make impossible.
SHARD_KIND = "deepseek-v41-flash-reference-shard/1"

#: The reference is only read as a result at batch size 1 (jobs 723978 /
#: 724075 / 724100 / 724124: eight identical-length prompts at batch 8 all
#: collapsed into repetition loops, and the same 32 documents scored 43.75 at
#: batch 32 against 78.1250 at batch 1). ``--merge`` enforces it rather than
#: documenting it.
ANCHOR_BATCH_SIZE = 1

#: The reference's model-parallel size. It is 4 because that is the rank count
#: this anchor is generated at and the rank count ``convert.py`` was run for --
#: NOT because the target is dep4. The reference is tensor-parallel; the target
#: is attention-DP + expert-parallel. They share a world size and nothing else,
#: and pinning this here stops an mp8 conversion from being merged in as though
#: it were the same measurement.
ANCHOR_MODEL_PARALLEL = 4


# ---------------------------------------------------------------------------
# lm-eval plumbing
# ---------------------------------------------------------------------------
def _build_task_dict(random_seed: int):
    """Mirror ``LmEvalEvaluator.__init__`` for ``gsm8k`` at its CLI defaults."""
    import lm_eval
    import lm_eval.tasks
    from lm_eval.tasks import TaskManager

    task_dict = lm_eval.tasks.get_task_dict(TASK, task_manager=TaskManager())
    adjusted = {}
    for name, task_obj in task_dict.items():
        assert not isinstance(task_obj, dict), f"unexpected task group {name}"
        task_obj.set_fewshot_seed(seed=random_seed)
        adjusted[name] = task_obj
        data = adjusted[name].dataset
        for split in data.keys():
            data[split] = data[split].shuffle(random_seed)
    return adjusted


def _make_lm(model, tokenizer, batch_size, record, log, replay=None):
    """An ``lm_eval`` LM whose ``generate_until`` is the reference forward.

    ``replay`` short-circuits generation entirely, returning a recorded text
    per ``(doc_id, idx)``. That is what makes the stitch honest: the shards
    contribute generations only, and the single final pass scores all 1319 of
    them through lm-eval's own filters, exactly as an unsharded run would --
    and every check in that branch exists so the pass can tell that it is
    scoring the generations it thinks it is.
    """
    from lm_eval.api.model import TemplateLM

    class ReferenceLM(TemplateLM):
        """Greedy batch completion through the checkpoint's own generate()."""

        def __init__(self):
            super().__init__()
            self._tokenizer = tokenizer

        @property
        def eot_token_id(self) -> int:
            return self._tokenizer.eos_token_id

        @property
        def tokenizer_name(self) -> str:
            return "deepseek-v41-flash-reference"

        def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs):
            if add_special_tokens is not None:
                kwargs["add_special_tokens"] = add_special_tokens
            return self._tokenizer.encode(string, **kwargs)

        def _loglikelihood_tokens(self, requests, **kwargs):
            raise NotImplementedError("gsm8k is generate_until only")

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError("gsm8k is generate_until only")

        def generate_until(self, requests, disable_tqdm: bool = False):
            prompts, keys, stop_lists, budgets = [], [], [], []
            for req in requests:
                prompt, gen_kwargs = req.args
                gen_kwargs = dict(gen_kwargs)
                # trtllm-eval passes SamplingParams(max_tokens=MAX_OUTPUT_LENGTH)
                # and lets the task yaml's max_gen_toks override it; gsm8k's
                # yaml sets no max_gen_toks, so 256 is what both sides use.
                budgets.append(int(gen_kwargs.get("max_gen_toks", MAX_OUTPUT_LENGTH)))
                stop_lists.append(list(gen_kwargs.get("until") or []))
                # The task yaml pins greedy decoding; anything else would make
                # this measurement unrepeatable and is rejected rather than
                # silently honoured.
                assert not gen_kwargs.get("do_sample", False), gen_kwargs
                assert float(gen_kwargs.get("temperature", 0.0)) == 0.0, gen_kwargs
                prompts.append(prompt)
                # Keyed by the document, not by position: lm-eval attaches
                # responses to the request objects and then re-emits one
                # ``samples`` entry per (doc, filter), so a positional join
                # against that list would silently mispair the export.
                keys.append((req.doc_id, req.idx))
            assert len(set(keys)) == len(keys), "duplicate (doc_id, idx) in the request set"
            assert len(set(budgets)) == 1, f"mixed output budgets {set(budgets)}"
            assert all(u == stop_lists[0] for u in stop_lists), "mixed stop strings"

            if replay is not None:
                missing = [k for k in keys if k not in replay]
                assert not missing, (
                    f"{len(missing)} documents have no recorded generation, "
                    f"first {missing[:4]}; the shards do not cover the request set"
                )
                extra = [k for k in replay if k not in set(keys)]
                assert not extra, (
                    f"{len(extra)} recorded generations are not in the request set, "
                    f"first {extra[:4]}; the shards cover documents this protocol "
                    f"does not score, so their union is not this measurement"
                )
                # The check that makes the stitch a measurement rather than a
                # claim: the text about to be scored has to have been generated
                # from the prompt lm-eval is scoring it against. A shard built
                # from a differently-seeded shuffle, a different few-shot count
                # or a stale request set fails here rather than contributing a
                # plausible-looking number.
                for k, prompt in zip(keys, prompts):
                    assert replay[k]["prompt"] == prompt, (
                        f"doc_id {k[0]}: the recorded generation was produced from a "
                        f"different prompt than the one being scored\n"
                        f"  recorded: {replay[k]['prompt'][:160]!r}...\n"
                        f"  scoring:  {prompt[:160]!r}..."
                    )
                for k in keys:
                    record[k] = replay[k]
                return [replay[k]["text"] for k in keys]

            return _generate_batched(
                model,
                tokenizer,
                prompts,
                keys=keys,
                max_new_tokens=budgets[0],
                until=stop_lists[0],
                batch_size=batch_size,
                record=record,
                log=log,
            )

    return ReferenceLM()


def _truncate_at_stop(text: str, until) -> str:
    """What the engine's ``stop`` does: cut at the first stop string, exclusive."""
    cut = len(text)
    for s in until:
        if not s:
            continue
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def _generate_batched(
    model,
    tokenizer,
    prompts,
    *,
    keys,
    max_new_tokens,
    until,
    batch_size,
    record,
    log,
    expect_token_ids=None,
):
    """Greedy completions for every prompt, batched by similar length.

    Batching is by sorted prompt length because the reference's ``generate``
    starts decoding at the batch's *shortest* prompt and runs every row to the
    same total length; mixing a 400-token and a 1300-token prompt in one batch
    would spend 900 extra steps on the short one. The permutation is undone
    before returning, and it is identical on every rank, so the collectives
    inside the forward stay in lockstep.

    ``expect_token_ids`` is the frozen request set's ids for these prompts. When
    given, every prompt's tokenization is checked against it *here*, on the leg
    that is about to feed the model -- the one place where "both legs see the
    same token ids" stops being a claim about two probe jobs and becomes a
    property of the generation that produced the score.
    """
    import torch

    generate = _reference_generate()
    order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i))
    out = [None] * len(prompts)
    t0 = time.time()
    done = 0
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        token_lists = []
        for i in chunk:
            ids = tokenizer.encode(prompts[i])
            assert len(ids) <= MAX_INPUT_LENGTH, (
                f"prompt {i} is {len(ids)} tokens, above the protocol's "
                f"max_input_length={MAX_INPUT_LENGTH}; the reference leg does not "
                f"implement prompt truncation, so this would not be the same "
                f"measurement as the target's"
            )
            if expect_token_ids is not None:
                assert ids == expect_token_ids[i], (
                    f"prompt {i} (key {keys[i]}) tokenizes to {len(ids)} ids here but "
                    f"the frozen request set records {len(expect_token_ids[i])}; this "
                    f"environment does not reproduce the protocol's token ids, so "
                    f"nothing generated from it is the same measurement"
                )
            token_lists.append(ids)
        completions = generate(
            model,
            token_lists,
            max_new_tokens,
            tokenizer.eos_token_id,
        )
        for i, ids, gen in zip(chunk, token_lists, completions):
            raw = tokenizer.decode(gen)
            text = _truncate_at_stop(raw, until)
            out[i] = text
            record[keys[i]] = {
                "prompt": prompts[i],
                "prompt_token_ids": ids,
                "n_prompt_tokens": len(ids),
                "generated_token_ids": gen,
                "n_generated_tokens": len(gen),
                "raw_text": raw,
                "text": text,
            }
        done += len(chunk)
        torch.cuda.synchronize()
        log(f"  [gen] {done}/{len(prompts)} prompts, {time.time() - t0:.1f}s elapsed")
    assert all(o is not None for o in out)
    return out


# ---------------------------------------------------------------------------
# the reference implementation
# ---------------------------------------------------------------------------
def _reference_generate():
    from generate import generate  # ty: ignore[unresolved-import]

    return generate


def _build_reference(ref_dir: str, ckpt_path: str, max_batch_size: int, max_seq_len: int, log):
    """Construct and load the reference Transformer, exactly as generate.py does."""
    import torch
    import torch.distributed as dist
    from model import ModelArgs, Transformer  # ty: ignore[unresolved-import]
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    # The allocator policy is set through PYTORCH_CUDA_ALLOC_CONF in the job, not
    # here: torch.cuda.memory._set_allocator_settings is deprecated in this
    # torch and its replacement is private, so the supported route is the
    # environment variable, read before the first allocation.
    assert "expandable_segments:True" in os.getenv("PYTORCH_CUDA_ALLOC_CONF", ""), (
        "PYTORCH_CUDA_ALLOC_CONF must request expandable_segments before the ranks "
        "start; 130 GB of shard load into a fragmenting allocator is the one "
        "configuration this run has no headroom for"
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(TORCH_SEED)

    with open(os.path.join(ref_dir, "config.json")) as f:
        args = ModelArgs(**json.load(f))
    args.temperature = 0.0  # greedy: model.sample() takes argmax at exactly 0
    args.max_batch_size = max_batch_size
    args.max_seq_len = max_seq_len

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    assert tokenizer is not None, f"no tokenizer at {ckpt_path}"
    log(
        f"build model: {args.n_layers} layers, mp={world_size}, "
        f"max_batch_size={max_batch_size}, max_seq_len={max_seq_len}"
    )
    with torch.device("cuda"):
        model = Transformer(args, tokenizer)
    shard = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    assert os.path.exists(shard), f"missing reference shard {shard}"
    log(f"load model: {shard}")
    load_model(model, shard)
    torch.set_default_device("cuda")
    free, total = torch.cuda.mem_get_info()
    log(f"loaded: HBM used {(total - free) / 2**30:.1f} GiB of {total / 2**30:.1f} GiB")
    return model, tokenizer, world_size, rank


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _software() -> dict:
    """The stack the reference actually ran on.

    An anchor is only repeatable if the reference implementation's own
    dependencies are pinned with it: ``inference/requirements.txt`` pins
    ``tilelang==0.1.8``, and tilelang 0.1.8 in turn only registers against an
    apache-tvm-ffi contemporary with it. Both are read from the installed
    packages rather than restated, so a drifting environment shows up in the
    record instead of silently changing the number.
    """
    import platform

    out = {"python": platform.python_version()}
    for mod in (
        "torch",
        "tilelang",
        "tvm_ffi",
        "transformers",
        "tokenizers",
        "safetensors",
        "lm_eval",
    ):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001 - a missing pin is itself the record
            out[mod] = f"unavailable: {exc}"
    try:
        import torch

        out["cuda_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability())
        out["gpu"] = torch.cuda.get_device_name()
    except Exception as exc:  # noqa: BLE001
        out["cuda_capability"] = f"unavailable: {exc}"
    return out


def _identity(hf_ckpt: str, ckpt_path: str) -> dict:
    """Everything a later reader needs to know which weights produced a score."""
    out = {"host": os.uname().nodename, "hf_checkpoint": hf_ckpt, "reference_checkpoint": ckpt_path}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        p = os.path.join(hf_ckpt, name)
        if os.path.exists(p):
            out[f"sha256:{name}"] = _sha256_file(p)
    shards = sorted(Path(ckpt_path).glob("model*-mp*.safetensors"))
    out["reference_shards"] = [{"name": p.name, "bytes": p.stat().st_size} for p in shards]
    return out


#: Sentinel for "this dict has no such key", so that absent and ``None`` are
#: distinguishable when two provenance blocks are compared.
_ABSENT = "<absent>"

#: The protocol facts every shard of one anchor must carry, and agree on.
_REQUIRED_PROTOCOL_KEYS = (
    "task",
    "driver",
    "model_parallel",
    "batch_size",
    "max_seq_len",
    "num_fewshot",
    "random_seed",
    "apply_chat_template",
    "fewshot_as_multiturn",
    "max_input_length",
    "max_output_length",
    "decoding",
    "scores_filter",
    "torch_seed",
    "lm_eval_version",
)

#: Facts that legitimately differ between shards of one anchor: the command
#: names a different range, and a shard's own sample count is its range's size.
_PER_RUN_PROTOCOL_KEYS = frozenset({"command", "n_samples"})

#: The one ``identity`` field that is a property of the run rather than of the
#: weights. Shards of one anchor are generated on whatever nodes the scheduler
#: hands out, so the merged record collects every host instead of demanding one.
_PER_RUN_IDENTITY_KEYS = frozenset({"host"})


def _assert_same(got: dict, want: dict, what: str, path: str, ref: str, ignore=frozenset()) -> None:
    """Two shards of one anchor must describe the same weights and the same stack.

    Strict on both sides: a key one shard carries and the other does not is a
    difference, because for ``identity`` and ``software`` that is exactly what a
    changed checkpoint or a drifted environment looks like. ``ignore`` is for the
    one field that is deliberately per-run -- the host the shard was generated
    on -- which is collected into the merged record instead of compared.
    """
    diff = sorted(
        k
        for k in (set(got) | set(want)) - set(ignore)
        if got.get(k, _ABSENT) != want.get(k, _ABSENT)
    )
    assert not diff, (
        f"{path} and {ref} disagree on {what}: "
        + "; ".join(f"{k}={got.get(k, _ABSENT)!r} vs {want.get(k, _ABSENT)!r}" for k in diff)
        + ". Shards that do not describe the same run are not one measurement."
    )


def _assert_protocol_compatible(got: dict, want: dict, path: str, ref: str) -> None:
    """The required protocol core must be present in both shards and equal.

    Keys outside that core are still checked wherever two shards both carry
    them, but their absence is survivable: the shard writer gained fields over
    this campaign, and every fact that decides whether two shards are the same
    measurement is in the core.
    """
    for d, name in ((got, path), (want, ref)):
        missing = [k for k in _REQUIRED_PROTOCOL_KEYS if k not in d]
        assert not missing, f"{name} records no {missing} in its protocol"
    shared = (set(got) & set(want)) - _PER_RUN_PROTOCOL_KEYS
    diff = sorted(k for k in shared if got[k] != want[k])
    assert not diff, (
        f"{path} and {ref} disagree on protocol: "
        + "; ".join(f"{k}={got[k]!r} vs {want[k]!r}" for k in diff)
        + ". Shards that do not describe the same run are not one measurement."
    )


def _generation_protocol(args, world_size: int, meta: dict) -> dict:
    """The protocol every shard of one anchor must agree on, exactly.

    Only equality-checked facts belong here. Per-run facts -- the host, the
    command, the elapsed time -- live in the artifact's ``run`` block, because
    requiring shards to agree on those would forbid sharding.
    """
    return {
        "task": TASK,
        "driver": "checkpoint reference implementation (inference/generate.py)",
        "model_parallel": world_size,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "num_fewshot": meta["num_fewshot"],
        "random_seed": RANDOM_SEED,
        "apply_chat_template": APPLY_CHAT_TEMPLATE,
        "fewshot_as_multiturn": FEWSHOT_AS_MULTITURN,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_output_length": MAX_OUTPUT_LENGTH,
        "decoding": "greedy (temperature 0.0)",
        "scores_filter": SCORES_FILTER,
        "torch_seed": TORCH_SEED,
        "lm_eval_version": meta["lm_eval_version"],
        "requests_file": os.path.basename(str(args.requests or "requests.jsonl")),
        "requests_sha256": meta["prompt_tokens_sha256"],
        "n_requests": meta["n_requests"],
    }


def _probe_requests(args, hf_ckpt: str) -> int:
    """Build the request set and write its token ids -- no model, no GPU.

    The cheap half of the anchor, and the one that has to hold for the target
    leg too: if the two legs do not see identical prompt token ids, nothing
    downstream of them is comparable. Running it alone also sizes the job --
    ``max_seq_len`` has to cover the longest prompt plus the output budget.
    """
    import lm_eval
    from transformers import AutoTokenizer

    task_dict = _build_task_dict(RANDOM_SEED)
    task = task_dict[TASK]
    task.build_all_requests(
        limit=args.num_samples,
        rank=0,
        world_size=1,
        cache_requests=False,
        rewrite_requests_cache=False,
        system_instruction=SYSTEM_PROMPT,
        apply_chat_template=APPLY_CHAT_TEMPLATE,
        fewshot_as_multiturn=FEWSHOT_AS_MULTITURN,
        chat_template=None,
        tokenizer_name="",
    )
    instances = task.instances
    assert instances, "gsm8k produced no requests"
    tok = AutoTokenizer.from_pretrained(args.ckpt_path)
    assert tok is not None, f"no tokenizer at {args.ckpt_path}"

    records, digest = [], hashlib.sha256()
    for inst in instances:
        prompt, gen_kwargs = inst.args
        ids = tok.encode(prompt)
        digest.update(json.dumps(ids).encode())
        records.append(
            {
                "index": len(records),
                "doc_id": inst.doc_id,
                "prompt": prompt,
                "prompt_token_ids": ids,
                "n_prompt_tokens": len(ids),
                "gen_kwargs": gen_kwargs,
                "target": task.doc_to_target(inst.doc),
            }
        )
    lens = sorted(r["n_prompt_tokens"] for r in records)
    meta = {
        "identity": _identity(hf_ckpt, args.ckpt_path),
        "software": _software(),
        "task": TASK,
        "n_requests": len(records),
        "num_fewshot": task.config.num_fewshot,
        "random_seed": RANDOM_SEED,
        "gen_kwargs": records[0]["gen_kwargs"],
        "lm_eval_version": lm_eval.__version__,
        "prompt_tokens_sha256": digest.hexdigest(),
        "prompt_len_min": lens[0],
        "prompt_len_median": lens[len(lens) // 2],
        "prompt_len_max": lens[-1],
        "max_input_length": MAX_INPUT_LENGTH,
        "max_output_length": MAX_OUTPUT_LENGTH,
        "command": " ".join(sys.argv),
    }
    assert lens[-1] <= MAX_INPUT_LENGTH, (
        f"longest prompt is {lens[-1]} tokens, above max_input_length={MAX_INPUT_LENGTH}"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    with open(out_dir / f"requests{tag}.jsonl", "w") as f:
        f.write(json.dumps({"__meta__": meta}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


def _load_requests(path, log) -> tuple[dict, list]:
    """Read the frozen request set and re-derive its digest.

    ``requests.jsonl`` is the protocol artifact: one job built it, a second job
    rebuilt it under the *repo* environment the target leg runs in, and the two
    agreed on ``prompt_tokens_sha256`` (jobs 723954 / 723979). Everything
    downstream reads it instead of rebuilding the task, which is what removes
    ``lm_eval``'s shared HuggingFace datasets cache -- and its ``filelock`` --
    from the GPU jobs entirely. Six shards racing that lock on this filesystem
    is what killed job 724364 with ``OSError: [Errno 116] Stale file handle``
    before it generated anything.

    The digest is recomputed rather than trusted so a truncated, appended-to or
    hand-edited request set cannot quietly become a different protocol.
    """
    records = []
    with open(path) as f:
        meta = json.loads(f.readline())["__meta__"]
        for line in f:
            records.append(json.loads(line))
    digest = hashlib.sha256()
    for r in records:
        digest.update(json.dumps(r["prompt_token_ids"]).encode())
    assert digest.hexdigest() == meta["prompt_tokens_sha256"], (
        f"{path} does not hash to the prompt_tokens_sha256 it declares: "
        f"{digest.hexdigest()} vs {meta['prompt_tokens_sha256']}"
    )
    assert len(records) == meta["n_requests"], (
        f"{path} holds {len(records)} requests but declares {meta['n_requests']}"
    )
    doc_ids = [r["doc_id"] for r in records]
    assert len(set(doc_ids)) == len(doc_ids), "duplicate doc_id in the frozen request set"
    log(
        f"requests: {len(records)} from {path}, "
        f"prompt_tokens_sha256 {meta['prompt_tokens_sha256'][:12]}..."
    )
    return meta, records


def _shard(args, model, tokenizer, world_size, rank, out_dir, tag, log) -> int:
    """Generate one ``--doc-range`` of the frozen request set. No scoring.

    A shard is generations only. Scoring happens once, in ``--merge``, over the
    union of every shard, through ``lm_eval``'s own filters -- so a sharded
    anchor and an unsharded one compute the same aggregate from the same code.
    Sharding exists because the reference is only trustworthy at batch size 1,
    which costs 44.6 s/document, and 1319 of those do not fit the partition's
    four-hour limit.
    """
    req_path = Path(args.requests) if args.requests else out_dir / "requests.jsonl"
    meta, records = _load_requests(req_path, log)
    lo, hi = (int(x) for x in args.doc_range.split(":"))
    selected = [r for r in records if lo <= r["doc_id"] < hi]
    assert selected, f"no documents in [{lo}, {hi}) -- nothing to generate"
    log(f"shard: doc_id in [{lo}, {hi}), {len(selected)} of {len(records)} documents")

    # Every request in the set must carry the same decoding contract; a mixed
    # one would mean the shards are not one measurement.
    until = list(selected[0]["gen_kwargs"].get("until") or [])
    budget = int(selected[0]["gen_kwargs"].get("max_gen_toks", MAX_OUTPUT_LENGTH))
    for r in selected:
        gk = r["gen_kwargs"]
        assert list(gk.get("until") or []) == until, f"doc {r['doc_id']}: mixed stop strings"
        assert int(gk.get("max_gen_toks", MAX_OUTPUT_LENGTH)) == budget, (
            f"doc {r['doc_id']}: mixed output budget"
        )
        assert not gk.get("do_sample", False), f"doc {r['doc_id']}: not greedy"
        assert float(gk.get("temperature", 0.0)) == 0.0, f"doc {r['doc_id']}: not greedy"

    record: dict = {}
    t0 = time.time()
    _generate_batched(
        model,
        tokenizer,
        [r["prompt"] for r in selected],
        keys=[r["doc_id"] for r in selected],
        max_new_tokens=budget,
        until=until,
        batch_size=args.batch_size,
        record=record,
        log=log,
        expect_token_ids=[r["prompt_token_ids"] for r in selected],
    )
    elapsed = time.time() - t0

    if rank == 0:
        samples = [
            {
                "doc_id": r["doc_id"],
                "prompt": record[r["doc_id"]]["prompt"],
                "prompt_token_ids": record[r["doc_id"]]["prompt_token_ids"],
                "n_prompt_tokens": record[r["doc_id"]]["n_prompt_tokens"],
                "generated_token_ids": record[r["doc_id"]]["generated_token_ids"],
                "n_generated_tokens": record[r["doc_id"]]["n_generated_tokens"],
                "raw_text": record[r["doc_id"]]["raw_text"],
                "text": record[r["doc_id"]]["text"],
                "target": r["target"],
            }
            for r in selected
        ]
        payload = {
            "kind": SHARD_KIND,
            "identity": _identity(args.hf_ckpt or "", args.ckpt_path),
            "software": _software(),
            "protocol": _generation_protocol(args, world_size, meta),
            "doc_range": args.doc_range,
            "doc_ids": [r["doc_id"] for r in selected],
            "run": {
                "host": os.uname().nodename,
                "command": " ".join(sys.argv),
                "elapsed_seconds": elapsed,
            },
            "scores": None,
            "score": None,
            "samples": samples,
        }
        (out_dir / f"reference{tag}.json").write_text(json.dumps(payload, indent=2))
        print(
            f"[shard] {args.doc_range}: {len(samples)} generations in {elapsed:.0f}s "
            f"-- NOT a score, merge before reading one",
            flush=True,
        )
    _shutdown(world_size)
    return 0


def _replay(args, model, tokenizer, world_size, rank, out_dir, tag, log) -> int:
    """Re-run named documents at a chosen batch size, and nothing else.

    The canary's 32 requests all go into one batch, and the reference's
    ``generate`` right-pads a batch and runs every row to the batch's own
    ``max(prompt) + max_new_tokens``: the shortest-prompt row therefore
    free-runs hundreds of tokens past the 256 that get scored, and every row
    shares one ``start_pos``. Whether a degenerate generation is the model's
    behaviour under this protocol or an artefact of that packing is not
    decidable by staring at the output -- it needs the same document driven
    on its own. This mode is that control; it writes the generations only,
    with no scoring, so nothing here can flatter the comparison.
    """
    req_path = Path(args.requests) if args.requests else out_dir / "requests.jsonl"
    wanted = [int(x) for x in args.replay_docs.split(",") if x.strip()]
    by_doc = {}
    with open(req_path) as f:
        f.readline()  # __meta__
        for line in f:
            r = json.loads(line)
            if r["doc_id"] in wanted:
                by_doc[r["doc_id"]] = r
    missing = [d for d in wanted if d not in by_doc]
    assert not missing, f"doc_ids not in {req_path}: {missing}"

    prompts = [by_doc[d]["prompt"] for d in wanted]
    until = list(by_doc[wanted[0]]["gen_kwargs"].get("until") or [])
    record: dict = {}
    _generate_batched(
        model,
        tokenizer,
        prompts,
        keys=list(wanted),
        max_new_tokens=MAX_OUTPUT_LENGTH,
        until=until,
        batch_size=args.batch_size,
        record=record,
        log=log,
    )
    if rank == 0:
        payload = {
            "identity": _identity(args.hf_ckpt or "", args.ckpt_path),
            "software": _software(),
            "batch_size": args.batch_size,
            "world_size": world_size,
            "max_new_tokens": MAX_OUTPUT_LENGTH,
            "until": until,
            "requests": str(req_path),
            "command": " ".join(sys.argv),
            "replays": [
                {
                    "doc_id": d,
                    "target": by_doc[d]["target"],
                    "n_prompt_tokens": record[d]["n_prompt_tokens"],
                    "n_generated_tokens": record[d]["n_generated_tokens"],
                    "generated_token_ids": record[d]["generated_token_ids"],
                    "raw_text": record[d]["raw_text"],
                    "text": record[d]["text"],
                }
                for d in wanted
            ],
        }
        (out_dir / f"replay{tag}.json").write_text(json.dumps(payload, indent=2))
        for d in wanted:
            gold = by_doc[d]["target"].split("#### ")[-1].strip()
            print(
                f"[replay b={args.batch_size}] doc {d} gold={gold} -> {record[d]['text'][-180:]!r}",
                flush=True,
            )
    _shutdown(world_size)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    if "--selftest" in sys.argv:
        return _selftest(lambda m: print(m, flush=True))
    ap.add_argument("--ref-dir", required=True, help="the checkpoint's inference/ directory")
    ap.add_argument("--ckpt-path", required=True, help="convert.py output for this rank count")
    ap.add_argument("--hf-ckpt", default=None, help="the published checkpoint, for digests")
    ap.add_argument("--out", required=True, help="artifact directory")
    ap.add_argument("--num-samples", type=int, default=None, help="None = the full 1319")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=MAX_INPUT_LENGTH + MAX_OUTPUT_LENGTH)
    ap.add_argument(
        "--fixtures-only",
        action="store_true",
        help="run only the fixed greedy prompts, no benchmark",
    )
    ap.add_argument(
        "--probe-requests",
        action="store_true",
        help="build and dump the request set only: no model, no GPU, no collectives",
    )
    ap.add_argument(
        "--replay-docs",
        default="",
        help="comma-separated doc_ids from requests.jsonl to re-run at --batch-size; "
        "the control for whether a generation is a property of the model or of "
        "how the driver batched it",
    )
    ap.add_argument(
        "--requests",
        default=None,
        help="requests.jsonl to replay from (default: <out>/requests.jsonl)",
    )
    ap.add_argument(
        "--doc-range",
        default="",
        help="START:END over doc_id -- generate only that half-open range and write "
        "a shard. The reference is only trustworthy at batch size 1, so the full "
        "1319 do not fit one job's wall-clock.",
    )
    ap.add_argument(
        "--merge",
        default="",
        help="comma-separated shard json files; score their union through lm-eval's "
        "own filters in one pass. No model is built, so this needs no GPU.",
    )
    ap.add_argument(
        "--freeze-fixtures",
        default="",
        help="comma-separated fixtures json files; freeze the prompts that were "
        "observed to produce identical tokens in all of them. No model, no GPU.",
    )
    ap.add_argument(
        "--compare",
        default="",
        help="comma-separated scored json files; report how far apart two runs of "
        "this reference land on the documents they share -- the anchor's own "
        "noise floor. No model, no GPU, nothing re-scored.",
    )
    ap.add_argument(
        "--min-runs",
        type=int,
        default=4,
        help="runs a prompt must agree across before it can be frozen",
    )
    ap.add_argument(
        "--min-hosts",
        type=int,
        default=4,
        help="distinct hosts a prompt must agree across before it can be frozen; "
        "the observed instability is between nodes, so same-host repeats are "
        "the weaker evidence",
    )
    ap.add_argument("--min-fixtures", type=int, default=3, help="anchor record floor")
    ap.add_argument("--max-fixtures", type=int, default=5, help="anchor record ceiling")
    ap.add_argument("--tag", default="", help="suffix for the artifact file names")
    args = ap.parse_args()

    ref_dir = os.path.abspath(args.ref_dir)
    sys.path.insert(0, ref_dir)
    hf_ckpt = args.hf_ckpt or str(Path(ref_dir).parent)

    if args.probe_requests:
        return _probe_requests(args, hf_ckpt)

    rank = int(os.getenv("RANK", "0"))

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    if args.compare:
        return _compare(args, log)

    if args.freeze_fixtures:
        return _freeze_fixtures(args, log)

    if args.merge:
        return _merge(args, hf_ckpt, log)

    model, tokenizer, world_size, rank = _build_reference(
        ref_dir, args.ckpt_path, args.batch_size, args.max_seq_len, log
    )

    out_dir = Path(args.out)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""

    if args.replay_docs:
        return _replay(args, model, tokenizer, world_size, rank, out_dir, tag, log)

    # --- the fixed greedy fixtures, always -------------------------------
    # Always at FIXTURE_BATCH_SIZE, never at --batch-size: a fixture is read as
    # evidence about the checkpoint, and this driver only produces trustworthy
    # generations one row at a time.
    fixtures = {}
    _generate_batched(
        model,
        tokenizer,
        FIXTURE_PROMPTS,
        keys=list(range(len(FIXTURE_PROMPTS))),
        max_new_tokens=FIXTURE_MAX_NEW_TOKENS,
        until=[],
        batch_size=FIXTURE_BATCH_SIZE,
        record=fixtures,
        log=log,
    )
    fixture_records = [dict(fixtures[i], index=i) for i in range(len(FIXTURE_PROMPTS))]
    if rank == 0:
        for r in fixture_records:
            print(f"[fixture] {r['prompt']!r}\n      -> {r['text']!r}", flush=True)
        (out_dir / f"fixtures{tag}.json").write_text(
            json.dumps(
                {
                    "record_version": FIXTURE_RECORD_VERSION,
                    "identity": _identity(hf_ckpt, args.ckpt_path),
                    "software": _software(),
                    "world_size": world_size,
                    "batch_size": FIXTURE_BATCH_SIZE,
                    # The reference model's row capacity, which is not the
                    # generation batch size. Recorded because it is the one
                    # remaining way this job's shape could reach the numbers.
                    "model_max_batch_size": args.batch_size,
                    "max_new_tokens": FIXTURE_MAX_NEW_TOKENS,
                    "decoding": "greedy (temperature 0.0)",
                    "torch_seed": TORCH_SEED,
                    "run": {
                        "host": os.uname().nodename,
                        "command": " ".join(sys.argv),
                    },
                    "fixtures": fixture_records,
                },
                indent=2,
            )
        )
    if args.fixtures_only:
        _shutdown(world_size)
        return 0

    if args.doc_range:
        return _shard(args, model, tokenizer, world_size, rank, out_dir, tag, log)

    # --- the benchmark ----------------------------------------------------
    import lm_eval

    task_dict = _build_task_dict(RANDOM_SEED)
    record: dict = {}
    lm = _make_lm(model, tokenizer, args.batch_size, record, log)
    t0 = time.time()
    results = lm_eval.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=args.num_samples,
        apply_chat_template=APPLY_CHAT_TEMPLATE,
        fewshot_as_multiturn=FEWSHOT_AS_MULTITURN,
        system_instruction=SYSTEM_PROMPT,
        log_samples=True,
    )
    elapsed = time.time() - t0

    scores = {
        k: (v * 100 if isinstance(v, (int, float)) else v)
        for k, v in results["results"][TASK].items()
    }
    score = scores[SCORES_FILTER]

    if rank == 0:
        # lm-eval emits one ``samples`` entry per (doc, filter), so fold the
        # filters back together per document and join on doc_id rather than on
        # list position.
        by_doc: dict = {}
        for s in results["samples"][TASK]:
            doc_id = s.get("doc_id")
            entry = by_doc.setdefault(
                doc_id, {"doc_id": doc_id, "target": s.get("target"), "filters": {}}
            )
            entry["filters"][s.get("filter", "?")] = {
                "filtered_resps": s.get("filtered_resps"),
                "exact_match": s.get("exact_match"),
            }
        per_sample = []
        for doc_id in sorted(by_doc):
            gen = record.get((doc_id, 0), {})
            assert gen, f"no generation recorded for doc_id {doc_id}"
            per_sample.append(
                {
                    "doc_id": doc_id,
                    "prompt": gen.get("prompt"),
                    "prompt_token_ids": gen.get("prompt_token_ids"),
                    "n_prompt_tokens": gen.get("n_prompt_tokens"),
                    "generated_token_ids": gen.get("generated_token_ids"),
                    "n_generated_tokens": gen.get("n_generated_tokens"),
                    "text": gen.get("text"),
                    "target": by_doc[doc_id]["target"],
                    "filters": by_doc[doc_id]["filters"],
                }
            )
        digest = hashlib.sha256()
        for r in per_sample:
            digest.update(json.dumps(r["prompt_token_ids"]).encode())
        payload = {
            "kind": "deepseek-v41-flash-reference-scored/1",
            "identity": _identity(hf_ckpt, args.ckpt_path),
            "software": _software(),
            "protocol": {
                "task": TASK,
                "driver": "checkpoint reference implementation (inference/generate.py)",
                "model_parallel": world_size,
                "n_samples": len(per_sample),
                "num_fewshot": task_dict[TASK].config.num_fewshot,
                "random_seed": RANDOM_SEED,
                "apply_chat_template": APPLY_CHAT_TEMPLATE,
                "fewshot_as_multiturn": FEWSHOT_AS_MULTITURN,
                "max_input_length": MAX_INPUT_LENGTH,
                "max_output_length": MAX_OUTPUT_LENGTH,
                "decoding": "greedy (temperature 0.0)",
                "scores_filter": SCORES_FILTER,
                "lm_eval_version": lm_eval.__version__,
                "torch_seed": TORCH_SEED,
                "batch_size": args.batch_size,
                "max_seq_len": args.max_seq_len,
            },
            "run": {
                "host": os.uname().nodename,
                "command": " ".join(sys.argv),
                "elapsed_seconds": elapsed,
            },
            "prompt_tokens_sha256": digest.hexdigest(),
            "scores": scores,
            "score": score,
            "samples": per_sample,
        }
        (out_dir / f"reference{tag}.json").write_text(json.dumps(payload, indent=2))
        print(
            f"[anchor] {TASK} {SCORES_FILTER} = {score:.4f} over "
            f"{len(per_sample)} samples in {elapsed:.0f}s",
            flush=True,
        )
        print(f"[anchor] all filters: {json.dumps(scores)}", flush=True)

    _shutdown(world_size)
    return 0


def _validate_shards(
    paths, frozen_by_doc: dict, req_path, requests_digest: str, exhaustive: bool, log
) -> tuple:
    """Decide whether a set of shard files is one measurement, before scoring it.

    Pure: no lm-eval, no tokenizer, no model, so every guard below is unit
    testable rather than only reachable through hours of GPU work. Returns
    ``(replay, shard_meta, common_identity, common_software, common_protocol)``,
    or raises with the specific reason this set is not an anchor.

    ``exhaustive`` demands that the shards cover the frozen request set exactly.
    It is off only for a deliberate subset stitch, such as the merge self-test.
    """
    assert paths, "--merge named no files"
    loaded = [(p, json.loads(Path(p).read_text())) for p in paths]

    # 1. Each file must be a shard: generations only, one declared range, and a
    #    batch-1 measurement.
    ranges = []
    for path, d in loaded:
        kind = d.get("kind")
        # ``None`` is accepted, and it is the one loose edge here, so say what it
        # buys: the six generation shards of this anchor were produced before the
        # marker existed, by the driver that still rebuilt the lm-eval task
        # in-process. Nothing about their *data* is weaker for it -- every sample
        # is checked against the frozen request set below, prompt and token ids --
        # but the binding to that request set is established at merge time rather
        # than at generation time, and ``shard_meta`` records which of the two a
        # given shard got. A shard that names a *different* request set is
        # rejected outright.
        assert kind in (SHARD_KIND, None), f"{path}: kind {kind!r} is not a shard artifact"
        assert d.get("score") is None and d.get("scores") is None, (
            f"{path} carries a score of its own, so it is a completed measurement "
            f"and not a shard. Merging one relabels its protocol as the merge's: "
            f"that is how job 724183 turned the batch-32 canary into a file "
            f"describing itself as 'batch size 1, generated in shards'."
        )
        assert d.get("doc_range"), f"{path} declares no doc_range"
        assert int(d["protocol"]["batch_size"]) == ANCHOR_BATCH_SIZE, (
            f"{path} was generated at batch size {d['protocol']['batch_size']}. "
            f"Measured on this stack (jobs 723978/724075/724100/724124), the "
            f"reference's batched generate() corrupts rows near the batch's shared "
            f"prefill boundary: the same 32 documents score 43.75 at batch 32 and "
            f"78.1250 at batch 1. Only batch {ANCHOR_BATCH_SIZE} is an anchor."
        )
        assert int(d["protocol"]["model_parallel"]) == ANCHOR_MODEL_PARALLEL, (
            f"{path} was generated at model-parallel size "
            f"{d['protocol']['model_parallel']}, not {ANCHOR_MODEL_PARALLEL}; the "
            f"reference is sharded per rank count, so this is different weights"
        )
        declared = d["protocol"].get("requests_sha256")
        assert declared in (None, requests_digest), (
            f"{path} was generated against request set {declared}, not the "
            f"{req_path} this merge scores ({requests_digest}); those are two "
            f"different protocols and their union is not one measurement"
        )
        lo, hi = (int(x) for x in d["doc_range"].split(":"))
        ranges.append((lo, hi, path))

    # 2. Every shard must describe the same run as the first one.
    ref_path, ref = loaded[0]
    common_identity = ref["identity"]
    common_software = ref["software"]
    common_protocol = {k: v for k, v in ref["protocol"].items() if k not in _PER_RUN_PROTOCOL_KEYS}
    absent = [k for k in _REQUIRED_PROTOCOL_KEYS if k not in common_protocol]
    assert not absent, f"{ref_path} records no {absent} in its protocol"
    for path, d in loaded[1:]:
        _assert_same(
            d["identity"],
            common_identity,
            "identity",
            path,
            ref_path,
            ignore=_PER_RUN_IDENTITY_KEYS,
        )
        _assert_same(d["software"], common_software, "software", path, ref_path)
        _assert_protocol_compatible(
            {k: v for k, v in d["protocol"].items() if k not in _PER_RUN_PROTOCOL_KEYS},
            common_protocol,
            path,
            ref_path,
        )

    # 3. Every generation must come from the frozen request set.
    replay: dict = {}
    shard_meta = []
    for (path, d), (lo, hi, _) in zip(loaded, ranges):
        shard_meta.append(
            {
                "file": path,
                "kind": d.get("kind"),
                "doc_range": d["doc_range"],
                "n": len(d["samples"]),
                "model_parallel": d["protocol"]["model_parallel"],
                "batch_size": d["protocol"]["batch_size"],
                "host": d.get("run", {}).get("host", d["identity"].get("host", "?")),
                "run": d.get("run", {}),
                # How this shard's generations are known to belong to the frozen
                # request set. Not decoration: it is the difference between a
                # driver that read the frozen prompts and one whose prompts were
                # only *checked* against them afterwards.
                "requests_bound": (
                    "at generation time"
                    if d["protocol"].get("requests_sha256")
                    else "at merge time"
                ),
            }
        )
        for s in d["samples"]:
            key = (s["doc_id"], 0)
            assert key not in replay, f"doc_id {s['doc_id']} appears in two shards"
            assert lo <= s["doc_id"] < hi, (
                f"{path}: doc_id {s['doc_id']} is outside its own declared range {d['doc_range']}"
            )
            fr = frozen_by_doc.get(s["doc_id"])
            assert fr is not None, (
                f"{path}: doc_id {s['doc_id']} is not in the frozen request set "
                f"{req_path}; this generation belongs to a different protocol"
            )
            assert s["prompt"] == fr["prompt"], (
                f"{path}: doc_id {s['doc_id']} was generated from a prompt that is "
                f"not the frozen one"
            )
            assert s["prompt_token_ids"] == fr["prompt_token_ids"], (
                f"{path}: doc_id {s['doc_id']} was generated from token ids that are "
                f"not the frozen ones ({len(s['prompt_token_ids'])} vs "
                f"{len(fr['prompt_token_ids'])})"
            )
            replay[key] = {
                "prompt": s["prompt"],
                "prompt_token_ids": s["prompt_token_ids"],
                "n_prompt_tokens": s["n_prompt_tokens"],
                "generated_token_ids": s["generated_token_ids"],
                "n_generated_tokens": s["n_generated_tokens"],
                "raw_text": s.get("raw_text", s["text"]),
                "text": s["text"],
            }

    # 4. Disjoint ranges, and -- for a full anchor -- exactly the request set.
    ranges.sort()
    for (a_lo, a_hi, a_path), (b_lo, b_hi, b_path) in zip(ranges, ranges[1:]):
        assert a_hi <= b_lo, (
            f"shard ranges overlap: {a_path} [{a_lo}:{a_hi}) and {b_path} [{b_lo}:{b_hi})"
        )
    covered = {k[0] for k in replay}
    if exhaustive:
        expected = set(frozen_by_doc)
        assert covered == expected, (
            f"the shards cover {len(covered)} of {len(expected)} documents; "
            f"missing {sorted(expected - covered)[:8]}, "
            f"unexpected {sorted(covered - expected)[:8]}"
        )
    log(
        f"merging {len(shard_meta)} shards, {len(replay)} generations from "
        f"{req_path}, ranges {[m['doc_range'] for m in shard_meta]}"
    )
    return replay, shard_meta, common_identity, common_software, common_protocol


def _merge(args, hf_ckpt: str, log) -> int:
    """Score the union of a validated shard set in one lm-eval pass. No GPU.

    Each shard contributed generations for its own ``--doc-range``; scoring
    happens once, here, over all of them, so the filters and aggregation are
    identical to what an unsharded run would have computed.
    """
    import lm_eval
    from transformers import AutoTokenizer

    req_path = Path(args.requests) if args.requests else Path(args.out) / "requests.jsonl"
    meta, frozen = _load_requests(req_path, log)
    frozen_by_doc = {r["doc_id"]: r for r in frozen}

    replay, shard_meta, common_identity, common_software, common_protocol = _validate_shards(
        [p.strip() for p in args.merge.split(",") if p.strip()],
        frozen_by_doc,
        req_path,
        meta["prompt_tokens_sha256"],
        exhaustive=args.num_samples is None,
        log=log,
    )
    covered = {k[0] for k in replay}

    # The shards agree on an identity; that identity also has to be the
    # checkpoint that is on disk now, or the record names weights this score was
    # not produced from.
    _assert_same(
        _identity(hf_ckpt, args.ckpt_path),
        common_identity,
        "checkpoint identity on disk now versus at generation time",
        f"{hf_ckpt} / {args.ckpt_path}",
        "the shards",
        ignore=_PER_RUN_IDENTITY_KEYS,
    )
    log("checkpoint identity: config/index/tokenizer digests and shard sizes unchanged")

    # 5. This environment must still reproduce the frozen token ids, or the
    #    prompts it is about to rebuild for scoring are not the protocol's.
    tok = AutoTokenizer.from_pretrained(args.ckpt_path)
    assert tok is not None, f"no tokenizer at {args.ckpt_path}"
    for doc_id in sorted(covered):
        fr = frozen_by_doc[doc_id]
        assert tok.encode(fr["prompt"]) == fr["prompt_token_ids"], (
            f"doc_id {doc_id}: this environment's tokenizer does not reproduce the "
            f"frozen prompt token ids"
        )
    log(f"tokenizer check: {len(covered)} prompts reproduce the frozen ids")

    task_dict = _build_task_dict(RANDOM_SEED)
    record: dict = {}
    lm = _make_lm(None, None, 1, record, log, replay=replay)
    results = lm_eval.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=args.num_samples,
        apply_chat_template=APPLY_CHAT_TEMPLATE,
        fewshot_as_multiturn=FEWSHOT_AS_MULTITURN,
        system_instruction=SYSTEM_PROMPT,
        log_samples=True,
    )
    scores = {
        k: (v * 100 if isinstance(v, (int, float)) else v)
        for k, v in results["results"][TASK].items()
    }
    score = scores[SCORES_FILTER]

    by_doc: dict = {}
    for s in results["samples"][TASK]:
        doc_id = s.get("doc_id")
        entry = by_doc.setdefault(
            doc_id, {"doc_id": doc_id, "target": s.get("target"), "filters": {}}
        )
        entry["filters"][s.get("filter", "?")] = {
            "filtered_resps": s.get("filtered_resps"),
            "exact_match": s.get("exact_match"),
        }
    per_sample, digest = [], hashlib.sha256()
    for doc_id in sorted(by_doc):
        gen = replay[(doc_id, 0)]
        digest.update(json.dumps(gen["prompt_token_ids"]).encode())
        per_sample.append(
            {
                "doc_id": doc_id,
                "prompt": gen["prompt"],
                "prompt_token_ids": gen["prompt_token_ids"],
                "n_prompt_tokens": gen["n_prompt_tokens"],
                "generated_token_ids": gen["generated_token_ids"],
                "n_generated_tokens": gen["n_generated_tokens"],
                "text": gen["text"],
                "target": by_doc[doc_id]["target"],
                "filters": by_doc[doc_id]["filters"],
            }
        )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    assert task_dict[TASK].config.num_fewshot == common_protocol["num_fewshot"], (
        f"this pass builds {task_dict[TASK].config.num_fewshot}-shot prompts but the "
        f"shards were generated {common_protocol['num_fewshot']}-shot"
    )
    payload = {
        "kind": "deepseek-v41-flash-reference-merged/1",
        # Generation and scoring are two different runs on two different stacks,
        # so they are recorded as two blocks. Re-stamping the merge's own host,
        # software and command as the measurement's is what made the 724183
        # output claim a batch size its data never had.
        "generation": {
            "identity": common_identity,
            "software": common_software,
            "protocol": common_protocol,
            "shards": shard_meta,
        },
        "scoring": {
            "software": _software(),
            "host": os.uname().nodename,
            "command": " ".join(sys.argv),
            "lm_eval_version": lm_eval.__version__,
            "scores_filter": SCORES_FILTER,
            "num_fewshot": task_dict[TASK].config.num_fewshot,
            "random_seed": RANDOM_SEED,
            "requests_file": str(req_path),
            "requests_sha256": meta["prompt_tokens_sha256"],
        },
        "identity": common_identity,
        "n_samples": len(per_sample),
        "prompt_tokens_sha256": digest.hexdigest(),
        "scores": scores,
        "score": score,
        "samples": per_sample,
    }
    (out_dir / f"reference{tag}.json").write_text(json.dumps(payload, indent=2))
    print(
        f"[anchor] {TASK} {SCORES_FILTER} = {score:.4f} over {len(per_sample)} samples", flush=True
    )
    print(f"[anchor] all filters: {json.dumps(scores)}", flush=True)
    return 0


def _compare(args, log) -> int:
    """How far apart are two scored runs of this reference on the documents they share?

    The anchor is one number, and the gate is that number minus a fixed 5.0.
    Whether a later difference of half a point means anything depends on how
    much this reference moves when nothing changes -- and it does move: greedy
    decoding here is not bitwise reproducible across physical nodes, which
    ``--freeze-fixtures`` already measured on the fixed prompts (4 of 9 candidate
    prompts produced 2-11 distinct continuations across 13 hosts). This is the
    same question asked of the benchmark instead of the fixtures.

    Nothing is re-scored. Each artifact already carries lm-eval's own per-sample
    ``exact_match`` verdict, and GSM8K aggregates ``exact_match`` by plain mean,
    so restricting to the shared documents is an average of recorded verdicts,
    not a second implementation of the metric.
    """
    loaded = []
    for path in args.compare.split(","):
        path = path.strip()
        if not path:
            continue
        d = json.loads(Path(path).read_text())
        assert d.get("score") is not None, f"{path} carries no score -- it is not a scored run"
        loaded.append((path, d))
    assert len(loaded) >= 2, "--compare needs at least two scored artifacts"

    by_path = {}
    for path, d in loaded:
        by_path[path] = {s["doc_id"]: s for s in d["samples"]}
    shared = set.intersection(*(set(v) for v in by_path.values()))
    assert shared, "the runs share no documents"

    filt = SCORES_FILTER.split(",", 1)[1]
    rows, identical = [], 0
    for doc_id in sorted(shared):
        samples = [by_path[p][doc_id] for p, _ in loaded]
        assert len({s["prompt"] for s in samples}) == 1, (
            f"doc_id {doc_id}: the runs scored different prompts, so they are "
            f"not two runs of one protocol"
        )
        toks = {tuple(s["generated_token_ids"]) for s in samples}
        if len(toks) == 1:
            identical += 1
        rows.append((doc_id, [s["filters"][filt]["exact_match"] for s in samples]))

    scores = []
    for i, (path, _) in enumerate(loaded):
        hits = sum(1 for _, ms in rows if ms[i] == 1.0)
        scores.append(100.0 * hits / len(rows))
    spread = max(scores) - min(scores)
    flipped = sum(1 for _, ms in rows if len(set(ms)) > 1)

    # Counted from the artifact rather than read from an ``n_samples`` field:
    # the scored-run writer gained that key over this campaign, and the earlier
    # canary artifact records its own count inside ``protocol`` instead.
    for (path, d), sc in zip(loaded, scores):
        log(
            f"[compare] {Path(path).name}: {sc:.4f} over the {len(rows)} shared "
            f"documents (whole run {d['score']:.4f} over {len(d['samples'])})"
        )
    log(
        f"[compare] {identical}/{len(rows)} generations token-identical, "
        f"{flipped} documents scored differently, spread {spread:.4f} points"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    (out_dir / f"compare{tag}.json").write_text(
        json.dumps(
            {
                "kind": "deepseek-v41-flash-reference-compare/1",
                "scores_filter": SCORES_FILTER,
                "runs": [
                    {
                        "file": path,
                        "whole_run_score": d["score"],
                        "whole_run_n": len(d["samples"]),
                        "score_over_shared": sc,
                    }
                    for (path, d), sc in zip(loaded, scores)
                ],
                "n_shared": len(rows),
                "n_token_identical": identical,
                "n_scored_differently": flipped,
                "spread_points": spread,
                "documents_scored_differently": [d for d, ms in rows if len(set(ms)) > 1],
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    return 0


def _freeze_fixtures(args, log) -> int:
    """Pick the frozen fixture set from measured cross-run agreement. No GPU.

    Greedy decoding through this reference is *not* reproducible across physical
    nodes. Nine batch-1 runs of five prompts on eight distinct GB300 nodes
    produced 3, 2, 2, 1 and 7 distinct 48-token continuations. Nothing is wrong
    with the checkpoint: a near-tie at one position forks the continuation, and
    the four-rank MoE and all-reduce reduction order is not bitwise stable
    between nodes. So a fixture cannot be chosen by writing down a prompt whose
    answer looks obvious -- it has to be one that *was observed* to produce the
    same tokens every time, and the count of times is part of the record.

    A prompt is frozen when it is token-identical across every artifact that
    contains it, over at least ``--min-runs`` runs on at least ``--min-hosts``
    distinct hosts. Everything rejected is written out too, with its competing
    continuations, because "these five prompts are stable" and "these five were
    the only ones tried" are different claims.
    """
    runs = []
    for path in args.freeze_fixtures.split(","):
        path = path.strip()
        if not path:
            continue
        d = json.loads(Path(path).read_text())
        assert d.get("record_version") == FIXTURE_RECORD_VERSION, (
            f"{path} is record_version {d.get('record_version')!r}, not "
            f"{FIXTURE_RECORD_VERSION}: it predates the fixture artifact recording "
            f"its own batch size, so it cannot be shown to be a batch-1 run"
        )
        assert int(d["batch_size"]) == FIXTURE_BATCH_SIZE, (
            f"{path} was generated at batch size {d['batch_size']}; only batch "
            f"{FIXTURE_BATCH_SIZE} generations are read as evidence here"
        )
        assert int(d["max_new_tokens"]) == FIXTURE_MAX_NEW_TOKENS, (
            f"{path} generated {d['max_new_tokens']} tokens, not {FIXTURE_MAX_NEW_TOKENS}; "
            f"continuations of different lengths cannot be compared for equality"
        )
        runs.append((path, d))
    assert runs, "--freeze-fixtures named no files"
    for path, d in runs[1:]:
        _assert_same(
            d["identity"],
            runs[0][1]["identity"],
            "identity",
            path,
            runs[0][0],
            ignore=_PER_RUN_IDENTITY_KEYS,
        )
        _assert_same(d["software"], runs[0][1]["software"], "software", path, runs[0][0])

    #  prompt text -> [(path, host, tuple(token ids), text), ...]
    seen: dict = {}
    for path, d in runs:
        host = d.get("run", {}).get("host", "?")
        for f in d["fixtures"]:
            seen.setdefault(f["prompt"], []).append(
                (path, host, tuple(f["generated_token_ids"]), f["text"])
            )

    frozen, rejected = [], []
    for prompt, obs in seen.items():
        variants: dict = {}
        for path, host, ids, text in obs:
            variants.setdefault(ids, {"text": text, "runs": [], "hosts": set()})
            variants[ids]["runs"].append(path)
            variants[ids]["hosts"].add(host)
        hosts = sorted({host for _, host, _, _ in obs})
        entry = {
            "prompt": prompt,
            "n_runs": len(obs),
            "n_hosts": len(hosts),
            "hosts": hosts,
            "n_variants": len(variants),
        }
        if len(variants) == 1 and len(obs) >= args.min_runs and len(hosts) >= args.min_hosts:
            ids, info = next(iter(variants.items()))
            frozen.append(
                dict(
                    entry,
                    generated_token_ids=list(ids),
                    n_generated_tokens=len(ids),
                    text=info["text"],
                )
            )
        else:
            entry["reason"] = (
                f"{len(variants)} distinct continuations"
                if len(variants) > 1
                else f"only {len(obs)} runs on {len(hosts)} hosts, "
                f"below --min-runs {args.min_runs} / --min-hosts {args.min_hosts}"
            )
            entry["variants"] = [
                {"n_runs": len(v["runs"]), "hosts": sorted(v["hosts"]), "text": v["text"]}
                for v in variants.values()
            ]
            rejected.append(entry)

    frozen.sort(key=lambda e: (-e["n_runs"], e["prompt"]))
    rejected.sort(key=lambda e: e["prompt"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    payload = {
        "kind": "deepseek-v41-flash-reference-fixtures/1",
        "identity": runs[0][1]["identity"],
        "software": runs[0][1]["software"],
        "decoding": "greedy (temperature 0.0)",
        "batch_size": FIXTURE_BATCH_SIZE,
        "max_new_tokens": FIXTURE_MAX_NEW_TOKENS,
        "torch_seed": TORCH_SEED,
        "criterion": (
            f"token-identical across every artifact containing the prompt, over at "
            f"least {args.min_runs} runs on at least {args.min_hosts} distinct hosts"
        ),
        "inputs": [
            {"file": p, "host": d.get("run", {}).get("host", "?"), "n_prompts": len(d["fixtures"])}
            for p, d in runs
        ],
        "command": " ".join(sys.argv),
        "frozen": frozen[: args.max_fixtures],
        "not_frozen": rejected,
        "frozen_beyond_max": frozen[args.max_fixtures :],
    }
    # One digest over the frozen (prompt, token ids) pairs, so the accuracy
    # record can pin all of them in a line instead of carrying five token
    # arrays, and so an edited artifact stops matching the record that cites it.
    fixtures_digest = hashlib.sha256()
    for e in payload["frozen"]:
        fixtures_digest.update(json.dumps([e["prompt"], e["generated_token_ids"]]).encode())
    payload["fixtures_sha256"] = fixtures_digest.hexdigest()
    (out_dir / f"fixtures-frozen{tag}.json").write_text(json.dumps(payload, indent=2))
    for e in payload["frozen"]:
        print(
            f"[frozen] {e['n_runs']} runs / {e['n_hosts']} hosts  "
            f"{e['prompt'][:60]!r} -> {e['text'][:90]!r}",
            flush=True,
        )
    for e in rejected:
        print(f"[not frozen] {e['prompt'][:60]!r}: {e['reason']}", flush=True)
    kept = len(payload["frozen"])
    print(f"[fixtures] fixtures_sha256 {payload['fixtures_sha256']}", flush=True)
    print(f"[fixtures] {kept} frozen of {len(seen)} candidates, from {len(runs)} runs", flush=True)
    assert args.min_fixtures <= kept <= args.max_fixtures, (
        f"{kept} fixtures survived the stability criterion; the anchor record needs "
        f"between {args.min_fixtures} and {args.max_fixtures}. Run more fixture jobs "
        f"or add candidate prompts -- do not relax the criterion."
    )
    return 0


def _selftest(log) -> int:
    """Exercise every guard the anchor depends on. No GPU, no model, seconds.

    The guards in ``_validate_shards`` and ``_freeze_fixtures`` are the whole
    reason a stitched anchor can be believed, and their failure mode is silence:
    a merge that skips a check still prints a plausible number. So they are
    driven here against synthetic artifacts built to violate one rule each, and
    each one has to be *observed* rejecting its case. A guard that stops working
    fails this in seconds instead of being discovered in the record.
    """
    import tempfile
    import traceback

    def req(doc_id, prompt, ids, target="#### 6"):
        return {
            "index": doc_id,
            "doc_id": doc_id,
            "prompt": prompt,
            "prompt_token_ids": ids,
            "n_prompt_tokens": len(ids),
            "gen_kwargs": {"until": ["Question:"], "do_sample": False, "temperature": 0.0},
            "target": target,
        }

    frozen = [req(i, f"prompt {i}", [1, 2, i]) for i in range(4)]
    frozen_by_doc = {r["doc_id"]: r for r in frozen}
    ident = {"host": "nodeA", "hf_checkpoint": "/ckpt", "sha256:config.json": "abc"}
    soft = {"python": "3.12.3", "torch": "2.9.0"}

    def proto(**over):
        p = {
            "task": TASK,
            "driver": "checkpoint reference implementation (inference/generate.py)",
            "model_parallel": 4,
            "batch_size": ANCHOR_BATCH_SIZE,
            "max_seq_len": 4352,
            "num_fewshot": 5,
            "random_seed": RANDOM_SEED,
            "apply_chat_template": APPLY_CHAT_TEMPLATE,
            "fewshot_as_multiturn": FEWSHOT_AS_MULTITURN,
            "max_input_length": MAX_INPUT_LENGTH,
            "max_output_length": MAX_OUTPUT_LENGTH,
            "decoding": "greedy (temperature 0.0)",
            "scores_filter": SCORES_FILTER,
            "torch_seed": TORCH_SEED,
            "lm_eval_version": "0.4.10",
        }
        p.update(over)
        return p

    def shard(doc_ids, doc_range, *, identity=None, software=None, protocol=None, **over):
        d = {
            "kind": SHARD_KIND,
            "identity": dict(identity or ident),
            "software": dict(software or soft),
            "protocol": protocol or proto(),
            "doc_range": doc_range,
            "run": {"host": "nodeA", "command": "anchor.py", "elapsed_seconds": 1.0},
            "scores": None,
            "score": None,
            "samples": [
                {
                    "doc_id": i,
                    "prompt": frozen_by_doc[i]["prompt"],
                    "prompt_token_ids": list(frozen_by_doc[i]["prompt_token_ids"]),
                    "n_prompt_tokens": 3,
                    "generated_token_ids": [9, 9],
                    "n_generated_tokens": 2,
                    "raw_text": " 6 trees.Question: next",
                    "text": " 6 trees.",
                    "target": frozen_by_doc[i]["target"],
                }
                for i in doc_ids
            ],
        }
        d.update(over)
        return d

    passed, failed = [], []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def write(name, obj):
            (tmp / name).write_text(json.dumps(obj))
            return str(tmp / name)

        digest = "d" * 64

        def expect_reject(name, paths, *, exhaustive=True, want=""):
            try:
                _validate_shards(
                    paths, frozen_by_doc, tmp / "requests.jsonl", digest, exhaustive, lambda _: None
                )
            except AssertionError as exc:
                if want and want not in str(exc):
                    failed.append(f"{name}: rejected for the wrong reason: {exc}")
                else:
                    passed.append(name)
                return
            except Exception as exc:  # noqa: BLE001 - any raise is a rejection, but say which
                passed.append(f"{name} ({type(exc).__name__})")
                return
            failed.append(f"{name}: ACCEPTED a shard set that violates the rule")

        good = [write("a.json", shard([0, 1], "0:2")), write("b.json", shard([2, 3], "2:4"))]

        # the happy path first: if this does not pass, every rejection below is
        # meaningless because the validator might simply reject everything.
        replay, meta, cid, csoft, cproto = _validate_shards(
            good, frozen_by_doc, tmp / "requests.jsonl", digest, True, lambda _: None
        )
        if len(replay) == 4 and len(meta) == 2 and cproto["batch_size"] == 1:
            passed.append("accepts a complete, consistent batch-1 shard set")
        else:
            failed.append(f"happy path returned {len(replay)} generations, {len(meta)} shards")
        if cid == ident and csoft == soft:
            passed.append("preserves the shards' own identity and software")
        else:
            failed.append("happy path did not preserve generation provenance")
        if [m["requests_bound"] for m in meta] == ["at merge time"] * 2:
            passed.append("records a shard that names no request set as merge-time bound")
        else:
            failed.append(
                f"mis-recorded the request binding: {[m['requests_bound'] for m in meta]}"
            )

        # identity may differ by host and only by host.
        other_host = write("host.json", shard([2, 3], "2:4", identity=dict(ident, host="nodeB")))
        try:
            _validate_shards(
                [good[0], other_host],
                frozen_by_doc,
                tmp / "requests.jsonl",
                digest,
                True,
                lambda _: None,
            )
            passed.append("accepts shards generated on different hosts")
        except AssertionError as exc:
            failed.append(f"rejected a legitimate cross-host shard set: {exc}")

        # A shard that read the frozen request set says so, and that claim is
        # checked rather than trusted: naming the right set is accepted and
        # upgrades the recorded binding; naming another one is rejected.
        bound = write("bound.json", shard([2, 3], "2:4", protocol=proto(requests_sha256=digest)))
        try:
            _, bmeta, _, _, _ = _validate_shards(
                [good[0], bound],
                frozen_by_doc,
                tmp / "requests.jsonl",
                digest,
                True,
                lambda _: None,
            )
            if bmeta[1]["requests_bound"] == "at generation time":
                passed.append(
                    "records a shard that read the frozen request set as generation-bound"
                )
            else:
                failed.append(
                    f"mis-recorded a generation-bound shard: {bmeta[1]['requests_bound']}"
                )
        except AssertionError as exc:
            failed.append(f"rejected a shard bound to the merge's own request set: {exc}")
        expect_reject(
            "rejects a shard generated against a different request set",
            [
                good[0],
                write("req.json", shard([2, 3], "2:4", protocol=proto(requests_sha256="0" * 64))),
            ],
            want="was generated against request set",
        )

        expect_reject(
            "rejects a scored artifact",
            [write("scored.json", shard([0, 1], "0:2", score=43.75, scores={"x": 43.75}))],
            want="carries a score of its own",
        )
        expect_reject(
            "rejects batch > 1",
            [write("b32.json", shard([0, 1], "0:2", protocol=proto(batch_size=32)))],
            want="batch size 32",
        )
        no_range = shard([0, 1], "0:2")
        no_range["doc_range"] = None
        expect_reject(
            "rejects a different model-parallel size",
            [write("mp8.json", shard([0, 1], "0:2", protocol=proto(model_parallel=8)))],
            want="model-parallel size 8",
        )
        expect_reject(
            "rejects a missing doc_range",
            [write("nor.json", no_range)],
            want="declares no doc_range",
        )
        expect_reject(
            "rejects overlapping ranges",
            [good[0], write("ov.json", shard([1, 2, 3], "1:4"))],
            want="appears in two shards",
        )
        expect_reject(
            "rejects a gap in coverage",
            [good[0]],
            want="cover 2 of 4 documents",
        )
        expect_reject(
            "rejects a doc outside its own declared range",
            [write("out.json", shard([0, 1, 2, 3], "0:2"))],
            want="outside its own declared range",
        )
        bad_prompt = shard([2, 3], "2:4")
        bad_prompt["samples"][0]["prompt"] = "a different prompt"
        expect_reject(
            "rejects a prompt that is not the frozen one",
            [good[0], write("bp.json", bad_prompt)],
            want="not the frozen one",
        )
        bad_ids = shard([2, 3], "2:4")
        bad_ids["samples"][0]["prompt_token_ids"] = [7, 7, 7, 7]
        expect_reject(
            "rejects token ids that are not the frozen ones",
            [good[0], write("bi.json", bad_ids)],
            want="not the frozen ones",
        )
        expect_reject(
            "rejects a drifted software stack",
            [good[0], write("sw.json", shard([2, 3], "2:4", software=dict(soft, torch="2.8.0")))],
            want="disagree on software",
        )
        expect_reject(
            "rejects a different checkpoint",
            [
                good[0],
                write(
                    "id.json",
                    shard([2, 3], "2:4", identity=dict(ident, **{"sha256:config.json": "zzz"})),
                ),
            ],
            want="disagree on identity",
        )
        expect_reject(
            "rejects a different few-shot count",
            [good[0], write("fs.json", shard([2, 3], "2:4", protocol=proto(num_fewshot=8)))],
            want="disagree on protocol",
        )
        thin = proto()
        thin.pop("torch_seed")
        expect_reject(
            "rejects a protocol missing a required key",
            [write("thin.json", shard([0, 1], "0:2", protocol=thin))],
            want="records no ['torch_seed']",
        )
        expect_reject(
            "rejects a foreign artifact kind",
            [write("kind.json", shard([0, 1], "0:2", kind="something-else/1"))],
            want="is not a shard artifact",
        )
        expect_reject("rejects an empty merge", [], want="named no files")

        # --- the frozen request set -------------------------------------
        def write_requests(name, records, digest=None):
            d = hashlib.sha256()
            for r in records:
                d.update(json.dumps(r["prompt_token_ids"]).encode())
            meta = {
                "n_requests": len(records),
                "prompt_tokens_sha256": digest or d.hexdigest(),
                "num_fewshot": 5,
                "lm_eval_version": "0.4.10",
            }
            with open(tmp / name, "w") as f:
                f.write(json.dumps({"__meta__": meta}) + "\n")
                for r in records:
                    f.write(json.dumps(r) + "\n")
            return tmp / name

        m, recs = _load_requests(write_requests("req.jsonl", frozen), lambda _: None)
        if len(recs) == 4 and m["n_requests"] == 4:
            passed.append("reads a frozen request set and re-derives its digest")
        else:
            failed.append("frozen request set did not round-trip")
        try:
            _load_requests(write_requests("bad.jsonl", frozen, digest="deadbeef"), lambda _: None)
            failed.append("accepted a request set whose digest does not match its contents")
        except AssertionError:
            passed.append("rejects a request set that does not hash to its own digest")

        # --- fixture freezing -------------------------------------------
        def fixt(name, host, texts, *, version=FIXTURE_RECORD_VERSION, batch=FIXTURE_BATCH_SIZE):
            obj = {
                "record_version": version,
                "identity": dict(ident, host=host),
                "software": dict(soft),
                "world_size": 4,
                "batch_size": batch,
                "max_new_tokens": FIXTURE_MAX_NEW_TOKENS,
                "decoding": "greedy (temperature 0.0)",
                "torch_seed": TORCH_SEED,
                "run": {"host": host, "command": "anchor.py --fixtures-only"},
                "fixtures": [
                    {
                        "index": i,
                        "prompt": f"fixture {i}",
                        "prompt_token_ids": [i],
                        "n_prompt_tokens": 1,
                        "generated_token_ids": list(t),
                        "n_generated_tokens": len(t),
                        "raw_text": str(t),
                        "text": str(t),
                    }
                    for i, t in enumerate(texts)
                ],
            }
            return write(name, obj)

        stable, drifting = (5, 5, 5), (6, 6, 6)
        runs = [
            fixt("f0.json", "n0", [stable, stable, drifting]),
            fixt("f1.json", "n1", [stable, stable, (7, 7, 7)]),
            fixt("f2.json", "n2", [stable, stable, (8, 8, 8)]),
            fixt("f3.json", "n3", [stable, stable, (9, 9, 9)]),
        ]
        ns = argparse.Namespace(
            freeze_fixtures=",".join(runs),
            out=str(tmp),
            tag="st",
            min_runs=4,
            min_hosts=4,
            min_fixtures=2,
            max_fixtures=5,
        )
        _freeze_fixtures(ns, lambda _: None)
        out = json.loads((tmp / "fixtures-frozen-st.json").read_text())
        if [e["prompt"] for e in out["frozen"]] == ["fixture 0", "fixture 1"]:
            passed.append("freezes only the prompts observed identical in every run")
        else:
            failed.append(f"froze {[e['prompt'] for e in out['frozen']]}")
        if out["not_frozen"] and out["not_frozen"][0]["n_variants"] == 4:
            passed.append("records the rejected prompt's competing continuations")
        else:
            failed.append("did not record why a prompt was rejected")
        recomputed = hashlib.sha256()
        for e in out["frozen"]:
            recomputed.update(json.dumps([e["prompt"], e["generated_token_ids"]]).encode())
        if out.get("fixtures_sha256") == recomputed.hexdigest():
            passed.append("pins the frozen fixtures with a digest over their own token ids")
        else:
            failed.append(
                f"fixtures_sha256 does not match its own frozen set: {out.get('fixtures_sha256')}"
            )

        def expect_fixture_reject(name, namespace, want=""):
            try:
                _freeze_fixtures(namespace, lambda _: None)
            except AssertionError as exc:
                if want and want not in str(exc):
                    failed.append(f"{name}: rejected for the wrong reason: {exc}")
                else:
                    passed.append(name)
                return
            failed.append(f"{name}: ACCEPTED")

        expect_fixture_reject(
            "rejects a batch > 1 fixture artifact",
            argparse.Namespace(
                **{**vars(ns), "freeze_fixtures": fixt("fb.json", "n4", [stable], batch=5)}
            ),
            want="batch size 5",
        )
        expect_fixture_reject(
            "rejects a fixture artifact that predates the batch-size record",
            argparse.Namespace(
                **{**vars(ns), "freeze_fixtures": fixt("fv.json", "n5", [stable], version=1)}
            ),
            want="record_version",
        )
        expect_fixture_reject(
            "rejects too few surviving fixtures",
            argparse.Namespace(**{**vars(ns), "min_fixtures": 3}),
            want="fixtures survived the stability criterion",
        )
        expect_fixture_reject(
            "rejects evidence from too few hosts",
            argparse.Namespace(**{**vars(ns), "min_hosts": 9, "min_fixtures": 1}),
            want="fixtures survived the stability criterion",
        )

        # --- run-to-run comparison ---------------------------------------
        filt = SCORES_FILTER.split(",", 1)[1]

        def scored(name, marks, *, tokens=None, prompts=None, score=50.0, n=None):
            samples = [
                {
                    "doc_id": i,
                    "prompt": (prompts or {}).get(i, f"prompt {i}"),
                    "generated_token_ids": (tokens or {}).get(i, [1, 2, 3]),
                    "n_generated_tokens": 3,
                    "text": " 6",
                    "target": "#### 6",
                    "filters": {filt: {"exact_match": mk, "filtered_resps": ["6"]}},
                }
                for i, mk in marks.items()
            ]
            obj = {"score": score, "samples": samples}
            if n is not None:  # the older scored writer carries no n_samples
                obj["n_samples"] = n
            return write(name, obj)

        a = scored("ra.json", {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0}, score=75.0)
        b = scored(
            "rb.json",
            {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0},
            tokens={1: [9, 9, 9]},
            score=60.0,
        )
        cns = argparse.Namespace(compare=f"{a},{b}", out=str(tmp), tag="cmp")
        _compare(cns, lambda _: None)
        cmp_out = json.loads((tmp / "compare-cmp.json").read_text())
        if (
            cmp_out["n_shared"] == 4
            and cmp_out["n_token_identical"] == 3
            and cmp_out["n_scored_differently"] == 1
            and cmp_out["documents_scored_differently"] == [1]
            and abs(cmp_out["spread_points"] - 25.0) < 1e-9
        ):
            passed.append("measures run-to-run spread over only the shared documents")
        else:
            failed.append(f"compare returned {json.dumps(cmp_out)[:200]}")
        try:
            _compare(
                argparse.Namespace(
                    compare=f"{a},{scored('rp.json', {0: 1.0}, prompts={0: 'other'})}",
                    out=str(tmp),
                    tag="cmpbad",
                ),
                lambda _: None,
            )
            failed.append("compare ACCEPTED two runs that scored different prompts")
        except AssertionError:
            passed.append("refuses to compare runs that scored different prompts")
        try:
            _compare(
                argparse.Namespace(
                    compare=write("unscored.json", shard([0, 1], "0:2")),
                    out=str(tmp),
                    tag="cmpun",
                ),
                lambda _: None,
            )
            failed.append("compare ACCEPTED an unscored shard")
        except AssertionError:
            passed.append("refuses to compare an artifact that carries no score")

    # --- stop-string semantics ------------------------------------------
    for text, stops, want in (
        ("6 trees.\nQuestion: next", ["Question:"], "6 trees.\n"),
        ("answer</s>tail", ["Question:", "</s>"], "answer"),
        ("no stop here", ["Question:"], "no stop here"),
        ("", ["Question:"], ""),
        ("aQuestion:b</s>c", ["</s>", "Question:"], "a"),
    ):
        got = _truncate_at_stop(text, stops)
        (passed if got == want else failed).append(f"_truncate_at_stop({text!r}) -> {got!r}")

    for name in passed:
        log(f"  ok   {name}")
    for name in failed:
        print(f"  FAIL {name}", flush=True)
    print(f"[selftest] {len(passed)} passed, {len(failed)} failed", flush=True)
    if failed:
        traceback.print_stack(limit=0)
    return 1 if failed else 0


def _shutdown(world_size: int) -> None:
    import torch.distributed as dist

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
