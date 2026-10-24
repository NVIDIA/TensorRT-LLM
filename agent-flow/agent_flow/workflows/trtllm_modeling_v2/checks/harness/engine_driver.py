#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a declared workload through the engine of the tree under test, and print structured records.

Usage (rank 0 under ``trtllm-llmapi-launch`` when the world size is above one)::

    <tree's python> engine_driver.py --spec spec.json [--quickstart <path>] [-- <quickstart argv>]

The engine is built by the TREE'S OWN ``examples/llm-api/quickstart_advanced.py``
from ``spec["quickstart_argv"]``, with what the gates state and the quickstart
has no flag for (:func:`engine_kwargs`). The spec's workload:

* ``requests``: ``{"id", "prompt", "max_tokens", "images", "render"}`` each,
  or ``{"id", "token_ids", "max_tokens"}``, or ``{"id", "probe_tokens",
  "max_tokens"}`` for a prompt of that many tokens of ordinary text.
  ``images`` are local files; ``render`` is ``loader`` (the default: the
  tree's ``default_multimodal_input_loader`` applies the model's chat template
  and image placeholders) or ``raw`` (``prompt`` is already rendered and the
  images go in ``multi_modal_data`` as they are).
* ``dataset``: ``{"source", "num", "text_field", "split", "config",
  "template", "skip", "max_tokens"}``, appended after ``requests``.
  Local, or a Hugging Face id read from the cache only; see
  :mod:`workload_io`.
* ``concurrency``: requests in flight at once (default 1: one at a time, so
  every request runs alone and the schedule does not depend on timing).
* ``ignore_eos``: generate ``max_tokens`` whatever the model emits.
* ``reuse_probe``: after the workload, send one prompt twice to see the
  engine resume the second from its KV cache (``engine_reuse.run_probe``).

Output, on stdout, each one JSON line behind a marker (see :mod:`workload_io`):
``MV2FLOW_DRIVER`` once, ``MV2FLOW_ENGINE`` once the engine is up,
``MV2FLOW_RECORD`` per request in order (:func:`record_of`; ``iterations``
counts streamed steps, the first being the context step), then
``MV2FLOW_REUSE`` once. Exits non-zero with a traceback when anything fails.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_facts  # noqa: E402 -- imported by path, see the package docstring
import engine_reuse  # noqa: E402
import workload_io as wio  # noqa: E402


def load_quickstart(path: str) -> Any:
    """Import the tree's ``quickstart_advanced.py`` as a module."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"quickstart not found: {source}")
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("mv2flow_quickstart", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The target the gates name, as the LLM-API argument ``modeling_v2`` takes it.
MODELING_V2_ENV = "TRTLLM_MODELING_V2"
#: A JSON object of ``kv_cache_config`` keys the quickstart has no flag for,
#: from the unit's options file (``gates.with_kv_cache_config``).
KV_CACHE_CONFIG_ENV = "MV2FLOW_KV_CACHE_CONFIG"


def kv_cache_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The ``kv_cache_config`` keys the gates hand over in :data:`KV_CACHE_CONFIG_ENV`."""
    text = (os.environ if environ is None else environ).get(KV_CACHE_CONFIG_ENV, "").strip()
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{KV_CACHE_CONFIG_ENV} is not a JSON object: {text[:200]!r}")
    return value


def merge_kv_cache_config(config: Any, overrides: Mapping[str, Any]) -> Any:
    """``config`` (the quickstart's ``KvCacheConfig``) with ``overrides`` added.

    A key the quickstart set keeps its value: its flags are the settings the
    gate argv states. The object is rebuilt through its class rather than
    assigned to, so a nested mapping such as ``mamba_state_config`` is
    validated into its model, as ``trtllm-eval`` validates the options file.
    """
    if not overrides:
        return config
    if config is None:
        return dict(overrides)
    if isinstance(config, Mapping):
        return {**overrides, **config}
    stated = getattr(config, "model_fields_set", None)
    if stated is None:
        raise TypeError(
            f"cannot add {sorted(overrides)} to a {type(config).__name__}: it is not a pydantic model"
        )
    return type(config)(**{**overrides, **{name: getattr(config, name) for name in stated}})


def engine_kwargs(
    kwargs: Mapping[str, Any], *, mode: str | None, kv_cache: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The ``LLM`` constructor's keyword arguments with what the gates add to the quickstart's."""
    merged = dict(kwargs)
    if mode:
        merged.setdefault("modeling_v2", mode)
    if kv_cache:
        merged["kv_cache_config"] = merge_kv_cache_config(merged.get("kv_cache_config"), kv_cache)
    return merged


def quickstart_args(quickstart: Any, argv: Sequence[str]) -> Any:
    """The quickstart's ``parse_arguments()`` over ``argv``."""
    saved = sys.argv
    sys.argv = [str(getattr(quickstart, "__file__", "quickstart_advanced.py")), *argv]
    try:
        return quickstart.parse_arguments()
    finally:
        sys.argv = saved


def build_llm(quickstart: Any, argv: Sequence[str]) -> tuple[Any, Any]:
    """``(llm, sampling_params)`` exactly as the quickstart builds them from ``argv``."""
    args = quickstart_args(quickstart, argv)
    mode = os.environ.get(MODELING_V2_ENV)
    kv_cache = kv_cache_overrides()
    llm_class = getattr(quickstart, "LLM", None)
    if (mode or kv_cache) and llm_class is not None:

        def build(*positional: Any, **kwargs: Any) -> Any:
            return llm_class(*positional, **engine_kwargs(kwargs, mode=mode, kv_cache=kv_cache))

        quickstart.LLM = build
        try:
            return quickstart.setup_llm(args)
        finally:
            quickstart.LLM = llm_class
    return quickstart.setup_llm(args)


def sampling_for(base: Any, max_tokens: int, ignore_eos: bool) -> Any:
    """A copy of ``base`` with this request's budget."""
    params = copy.deepcopy(base)
    params.max_tokens = int(max_tokens)
    if ignore_eos:
        params.ignore_eos = True
    return params


def _model_dir(llm: Any) -> str:
    return str(getattr(llm, "_hf_model_dir", None) or getattr(llm, "model_dir", "") or "")


def make_input(llm: Any, request: dict[str, Any]) -> Any:
    """The ``generate`` input for one request: text, token ids, or text plus images."""
    if request.get("token_ids") is not None:
        return {"prompt_token_ids": [int(t) for t in request["token_ids"]]}
    if request.get("probe_tokens") is not None:
        tokens = int(request["probe_tokens"])
        return {"prompt_token_ids": engine_reuse.probe_prompt(llm.tokenizer, tokens)}
    prompt = str(request.get("prompt", ""))
    images = [str(p) for p in request.get("images") or ()]
    if not images:
        return prompt
    for image in images:
        if not Path(image).is_file():
            raise FileNotFoundError(f"image not found: {image}")
        if wio.is_lfs_pointer(image):
            raise RuntimeError(
                f"{image} is a git-lfs pointer, not an image: run `git lfs pull --include "
                f"{Path(image).name}` in the tree under test"
            )
    if request.get("render") == "raw":
        from PIL import Image

        return {
            "prompt": prompt,
            "multi_modal_data": {"image": [Image.open(p).convert("RGB") for p in images]},
        }
    from tensorrt_llm.inputs import default_multimodal_input_loader

    model_dir = _model_dir(llm)
    model_type = json.loads((Path(model_dir) / "config.json").read_text())["model_type"]
    inputs = default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=model_dir,
        model_type=model_type,
        modality="image" if len(images) == 1 else "multiple_image",
        prompts=[prompt],
        media=images if len(images) == 1 else [images],
        image_data_format="pt",
        device="cpu",
    )
    return inputs[0]


def record_of(index: int, request: dict[str, Any], output: Any, steps: int) -> dict[str, Any]:
    """The ``MV2FLOW_RECORD`` payload of one finished request."""
    sequence = output.outputs[0]
    token_ids = [int(t) for t in sequence.token_ids]
    prompt_ids = getattr(output, "prompt_token_ids", None)
    return {
        "index": index,
        "id": request.get("id", str(index)),
        "prompt_sha": wio.prompt_digest(str(request.get("prompt", request.get("token_ids", "")))),
        "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None,
        "cached_tokens": engine_reuse.cached_tokens(output),
        "token_ids": token_ids,
        "text": sequence.text,
        "tokens": len(token_ids),
        "iterations": steps,
        "finish_reason": str(getattr(sequence, "finish_reason", "")),
        "images": len(request.get("images") or ()),
    }


def resolve_requests(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The declared requests followed by the dataset's, each with its budget."""
    default_tokens = int(spec.get("max_tokens", 64))
    requests = []
    for i, entry in enumerate(spec.get("requests") or ()):
        request = dict(entry)
        request.setdefault("id", f"r{i}")
        request.setdefault("max_tokens", default_tokens)
        requests.append(request)
    dataset = spec.get("dataset")
    if dataset:
        rows = wio.load_rows(
            dataset["source"], split=dataset.get("split", "test"), config=dataset.get("config")
        )
        chosen = wio.select_prompts(
            rows,
            num=int(dataset.get("num", 32)),
            text_field=dataset.get("text_field"),
            template=dataset.get("template") or "{text}",
            skip=int(dataset.get("skip", 0)),
        )
        for i, prompt in enumerate(chosen):
            requests.append(
                {
                    "id": f"d{i}",
                    "prompt": prompt,
                    "max_tokens": int(dataset.get("max_tokens", default_tokens)),
                }
            )
    return requests


def run_requests(
    llm: Any, base: Any, spec: dict[str, Any], ledger: engine_reuse.ReuseLedger
) -> None:
    """Run every request, ``concurrency`` at a time, and print a record for each."""
    requests = resolve_requests(spec)
    concurrency = max(1, int(spec.get("concurrency", 1)))
    ignore_eos = bool(spec.get("ignore_eos", False))
    for start in range(0, len(requests), concurrency):
        window = requests[start : start + concurrency]
        streams = [
            llm.generate_async(
                make_input(llm, request),
                sampling_for(base, request["max_tokens"], ignore_eos),
                streaming=True,
            )
            for request in window
        ]
        for offset, (request, stream) in enumerate(zip(window, streams)):
            steps, output = 0, None
            for output in stream:
                steps += 1
            if output is None:
                raise RuntimeError("the engine streamed no output for a request")
            wio.emit(wio.RECORD_MARKER, record_of(start + offset, request, output, steps))
            ledger.add(output)


def probe_reuse(llm: Any, base: Any, granularity: engine_reuse.Granularity) -> dict[str, Any]:
    """Run the reuse probe (``engine_reuse.run_probe``) at the engine's own ``granularity``.

    A probe that raises is reported, not raised: the workload's records are
    already out.
    """
    tokens = None
    try:
        tokens = engine_reuse.probe_tokens(*granularity)
        args = getattr(llm, "args", None)
        parallel = getattr(args, "parallel_config", None)
        chunked = bool(getattr(args, "enable_chunked_prefill", False))
        return engine_reuse.run_probe(
            llm,
            sampling_for(base, 1, True),
            tokens,
            max_seq_len=getattr(args, "max_seq_len", None),
            max_prompt=None if chunked else getattr(args, "max_num_tokens", None),
            attention_dp=bool(getattr(parallel, "enable_attention_dp", False)),
        )
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        return {"tokens": tokens, "error": f"{type(exc).__name__}: {exc}"[:300]}


def run_workload(llm: Any, base: Any, spec: Mapping[str, Any]) -> None:
    """Run ``spec``'s workload on an engine that is up, and print what it prints."""
    ledger = engine_reuse.ReuseLedger()
    run_requests(llm, base, dict(spec), ledger)
    granularity = engine_reuse.engine_granularity(llm)
    probe = probe_reuse(llm, base, granularity) if spec.get("reuse_probe") else None
    engine_reuse.emit(
        ledger,
        granularity,
        concurrency=max(1, int(spec.get("concurrency", 1))),
        probe=probe,
    )


def main(argv: list[str] | None = None) -> int:
    """Read the spec, build the engine, run the workload, print records."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, help="the workload spec, a JSON file")
    parser.add_argument(
        "--quickstart", help="the tree's quickstart_advanced.py (default: the spec's)"
    )
    parser.add_argument(
        "quickstart_argv",
        nargs=argparse.REMAINDER,
        help="after `--`: the quickstart's argv (default: the spec's)",
    )
    options = parser.parse_args(argv)
    spec = json.loads(Path(options.spec).read_text(encoding="utf-8"))
    if options.quickstart:
        spec["quickstart"] = options.quickstart
    words = [w for w in options.quickstart_argv if w != "--"] if options.quickstart_argv else []
    if words:
        spec["quickstart_argv"] = words

    quickstart = load_quickstart(spec["quickstart"])
    import tensorrt_llm

    wio.emit(
        wio.DRIVER_MARKER,
        {
            "tensorrt_llm": str(getattr(tensorrt_llm, "__file__", "")),
            "quickstart": spec["quickstart"],
        },
    )
    observer = engine_facts.WorkerObserver()
    observer.install()
    argv = list(spec.get("quickstart_argv") or ())
    llm, base = build_llm(quickstart, argv)
    try:
        engine_facts.emit(llm, observer, argv=argv)
        run_workload(llm, base, spec)
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
