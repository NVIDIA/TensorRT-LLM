# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ChatGLM3-6B GSM8K accuracy gate (HF reference vs TRT-LLM via trtllm-eval).

Selectors (mapped to acceptance criteria)::

    pytest -q test_chatglm3_gsm8k.py -k 'smoke or accuracy_canary'    # LLM-API smoke + GSM8K canary
    pytest -q test_chatglm3_gsm8k.py -k 'full_trtllm_eval and baseline'    # full GSM8K, cuda_graph=false
    pytest -q test_chatglm3_gsm8k.py -k 'full_trtllm_eval and cuda_graph'  # full GSM8K, cuda_graph=true

Both the HF reference and the TRT-LLM run consume the same GSM8K evaluator
(``tensorrt_llm.evaluate.GSM8K`` -> lm-eval ``gsm8k`` task) with identical
num_samples / seed / decoding, and each full-gate run writes a score artifact
recording that shared config, both scores, and the CUDA-graph hard-path flag.
"""

import hashlib
import json
import os
from dataclasses import dataclass

import pytest
import torch

import tensorrt_llm.evaluate
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SamplingParams

CHATGLM3_CKPT = os.environ.get(
    "CHATGLM3_CKPT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/hf_data/chatglm3-6b",
)
GSM8K_DATASET_PATH = os.environ.get("CHATGLM3_GSM8K_PATH") or None
ARTIFACT_DIR = os.environ.get(
    "CHATGLM3_ARTIFACT_DIR", os.path.join(os.getcwd(), "chatglm3_gsm8k_artifacts")
)
CANARY_N = int(os.environ.get("CHATGLM3_GSM8K_CANARY_N", "20"))
FULL_N = int(os.environ.get("CHATGLM3_GSM8K_N", "1319"))
MAX_INPUT_LEN = 4096
MAX_OUTPUT_LEN = 256
GAP_TOL = 2.0  # absolute points

skip_no_ckpt = pytest.mark.skipif(
    not os.path.isdir(CHATGLM3_CKPT), reason=f"ChatGLM3 checkpoint not found at {CHATGLM3_CKPT}"
)
skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def _force_single_process_worker(monkeypatch):
    # The CUDA-graph hard-path assertions introspect the in-process CUDAGraphRunner,
    # reachable only when the worker runs in-process.
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")


def _patch_chatglm3_hf_tied_compat():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(
        "modeling_chatglm.ChatGLMForConditionalGeneration", CHATGLM3_CKPT
    )
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}


@dataclass(repr=False)
class RuntimeCfg:
    cuda_graph: bool
    overlap_scheduler: bool

    def __repr__(self) -> str:
        return f"cuda_graph:{self.cuda_graph}-overlap:{self.overlap_scheduler}"


BASELINE = RuntimeCfg(cuda_graph=False, overlap_scheduler=False)
ENABLED = RuntimeCfg(cuda_graph=True, overlap_scheduler=True)
CFGS = [pytest.param(BASELINE, id="baseline"), pytest.param(ENABLED, id="cuda_graph")]

# Process-cached HF reference scores keyed by num_samples (HF pass is the slow part).
_HF_SCORE_CACHE = {}


def _make_evaluator(num_samples: int):
    return tensorrt_llm.evaluate.GSM8K(
        dataset_path=GSM8K_DATASET_PATH, num_samples=num_samples, random_seed=0
    )


def _shared_eval_config(num_samples: int) -> dict:
    """The eval config both the HF reference and the TRT-LLM run share verbatim."""
    return {
        "benchmark": "gsm8k",
        "harness": "lm-eval gsm8k task via tensorrt_llm.evaluate.GSM8K",
        "dataset": ("local:" + GSM8K_DATASET_PATH) if GSM8K_DATASET_PATH else "hub:openai/gsm8k",
        "dataset_split": "test",
        "num_samples": num_samples,
        "random_seed": 0,
        "num_fewshot": "lm_eval_gsm8k_default(5-shot CoT)",
        "prompt_template": "lm_eval_gsm8k_default(Question:/Answer:)",
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "answer_extraction": "lm_eval_gsm8k_default(flexible-extract / strict-match exact_match)",
        "tokenizer_path": CHATGLM3_CKPT,
        "max_input_len": MAX_INPUT_LEN,
        "max_output_tokens": MAX_OUTPUT_LEN,
        "stop_strings": "lm_eval_gsm8k_default(until markers)",
        "temperature": 0.0,
        "top_k": 1,
        "sampling": False,
    }


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _pick_metric_key(scores: dict) -> str:
    numeric = {
        m: v for m, v in scores.items() if isinstance(v, (int, float)) and "_stderr" not in m
    }
    for pref in ("exact_match,flexible-extract", "exact_match,strict-match", "exact_match"):
        if pref in numeric:
            return pref
    em = sorted(m for m in numeric if m.startswith("exact_match"))
    if em:
        return em[0]
    # Fail loudly: gating on a non-exact_match metric would compare unrelated numbers.
    raise AssertionError(f"no exact_match GSM8K metric in {list(scores)}; cannot gate accuracy")


def _build_llm(cfg: RuntimeCfg) -> LLM:
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.5, enable_block_reuse=False, use_kv_cache_manager_v2=True
    )
    return LLM(
        model=CHATGLM3_CKPT,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        dtype="float16",
        kv_cache_config=kv_cache_config,
        cuda_graph_config=CudaGraphConfig() if cfg.cuda_graph else None,
        disable_overlap_scheduler=not cfg.overlap_scheduler,
        max_batch_size=32,
        max_num_tokens=8192,
    )


def _find_cuda_graph_runner(llm: LLM):
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner

    engine = getattr(getattr(llm, "_executor", None), "engine", None)
    runner = getattr(getattr(engine, "model_engine", None), "cuda_graph_runner", None)
    return runner if isinstance(runner, CUDAGraphRunner) else None


def _assert_cuda_graph_hard_path(llm: LLM, cfg: RuntimeCfg) -> bool:
    """Returns True iff the enabled config actually captured >=1 CUDA graph."""
    runner = _find_cuda_graph_runner(llm)
    if cfg.cuda_graph:
        assert runner is not None, (
            "enabled config: could not reach the in-process CUDAGraphRunner "
            "(TLLM_WORKER_USE_SINGLE_PROCESS=1 is required for this introspection)"
        )
        graphs = runner.graphs
        print(
            f"[cuda_graph_hard_path] cfg={cfg} enabled={runner.enabled} "
            f"num_captured_graphs={len(graphs)} keys={list(graphs)[:8]}"
        )
        assert runner.enabled, "enabled config: CUDAGraphRunner.enabled is False (silent fallback)"
        assert len(graphs) >= 1, (
            "enabled config captured 0 CUDA graphs -> silent eager fallback, not the hard path"
        )
        assert all(isinstance(g, torch.cuda.CUDAGraph) for g in graphs.values())
        return True
    if runner is not None:
        assert not runner.enabled, "baseline config: CUDAGraphRunner.enabled unexpectedly True"
        assert len(runner.graphs) == 0, "baseline config unexpectedly captured CUDA graphs"
    return False


def _assert_v2_and_backend(llm: LLM, cfg: RuntimeCfg) -> bool:
    args = llm.args
    assert args.kv_cache_config.use_kv_cache_manager_v2 is True
    assert str(getattr(args, "attn_backend", "TRTLLM")).upper() == "TRTLLM"
    if cfg.cuda_graph:
        assert args.cuda_graph_config is not None
    assert args.disable_overlap_scheduler is (not cfg.overlap_scheduler)
    return _assert_cuda_graph_hard_path(llm, cfg)


def _trtllm_score(num_samples: int, cfg: RuntimeCfg, metric_key: str):
    """Returns (score_0_100, cuda_graph_hard_path_bool)."""
    evaluator = _make_evaluator(num_samples)
    llm = _build_llm(cfg)
    try:
        sampling_params = SamplingParams(
            max_tokens=MAX_OUTPUT_LEN,
            truncate_prompt_tokens=MAX_INPUT_LEN,
            temperature=0.0,
            top_k=1,
        )
        score = evaluator.evaluate(llm, sampling_params, scores_filter=metric_key)
        hard_path = _assert_v2_and_backend(llm, cfg)
        return score, hard_path
    finally:
        llm.shutdown()


def _hf_reference_scores(num_samples: int) -> dict:
    if num_samples in _HF_SCORE_CACHE:
        return _HF_SCORE_CACHE[num_samples]

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    class _GreedyHFLM(HFLM):
        """lm-eval HF wrapper with ChatGLM's legacy tuple-cache greedy loop."""

        @torch.inference_mode()
        def _model_generate(self, context, max_length=None, stop=None, **generation_kwargs):
            model = self.model
            device = context.device
            bsz, ctx_len = context.shape
            pad_id = self.tokenizer.pad_token_id
            eos_id = self.tokenizer.eos_token_id
            stops = [s for s in (stop or []) if s]
            if max_length is not None:
                max_new = max(1, int(max_length) - ctx_len)
            elif "max_gen_toks" in generation_kwargs:
                max_new = int(generation_kwargs["max_gen_toks"])
            else:
                max_new = self.max_gen_toks

            attn = generation_kwargs.get("attention_mask")
            if attn is None:
                attn = (context != pad_id).long()
            attn = attn.to(device)
            positions = (attn.long().cumsum(-1) - 1).clamp(min=0)

            out = model(
                input_ids=context,
                position_ids=positions,
                attention_mask=attn,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            last_pos = positions[:, -1:]

            finished = torch.zeros(bsz, dtype=torch.bool, device=device)
            cols = []
            emitted = [[] for _ in range(bsz)]
            for _ in range(max_new):
                pad_col = torch.full((bsz,), pad_id, dtype=torch.long, device=device)
                nxt = torch.where(finished, pad_col, logits.argmax(-1))
                cols.append(nxt)
                for b in range(bsz):
                    if not finished[b]:
                        emitted[b].append(int(nxt[b]))
                finished = finished | (nxt == eos_id)
                if stops:
                    for b in range(bsz):
                        if not finished[b] and any(s in self.tok_decode(emitted[b]) for s in stops):
                            finished[b] = True
                if bool(finished.all()):
                    break
                last_pos = last_pos + 1
                attn = torch.cat([attn, attn.new_ones((bsz, 1))], dim=-1)
                out = model(
                    input_ids=nxt.unsqueeze(1),
                    position_ids=last_pos,
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()

            gen = torch.stack(cols, dim=1) if cols else context.new_zeros((bsz, 0))
            return torch.cat([context, gen], dim=1)

    evaluator = _make_evaluator(num_samples)

    cfg = AutoConfig.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
    # Compatibility aliases for the 2023 ChatGLM remote code under transformers >=5.x.
    if getattr(cfg, "max_length", None) is None:
        cfg.max_length = 8192
    if getattr(cfg, "num_hidden_layers", None) is None:
        cfg.num_hidden_layers = cfg.num_layers
    _patch_chatglm3_hf_tied_compat()
    tok = AutoTokenizer.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            CHATGLM3_CKPT, config=cfg, trust_remote_code=True, torch_dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    try:
        hflm = _GreedyHFLM(
            pretrained=model, tokenizer=tok, trust_remote_code=True, batch_size=16, max_length=8192
        )
        results = lm_eval.evaluate(
            lm=hflm,
            task_dict=evaluator.task_dict,
            limit=evaluator.num_samples,
            apply_chat_template=evaluator.apply_chat_template,
            fewshot_as_multiturn=False,
            system_instruction=None,
            log_samples=False,
        )
        scores = results["results"]["gsm8k"]
        _HF_SCORE_CACHE[num_samples] = scores
        return scores
    finally:
        del model
        torch.cuda.empty_cache()


def _write_artifact(name: str, payload: dict) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[gsm8k_artifact] wrote {path}")
    return path


def _run_config(n: int, label: str, cfg: RuntimeCfg, artifact_name: str = None) -> dict:
    shared_cfg = _shared_eval_config(n)
    src = shared_cfg["dataset"]
    hf_scores = _hf_reference_scores(n)
    metric_key = _pick_metric_key(hf_scores)
    hf = float(hf_scores[metric_key]) * 100  # lm-eval native 0-1 -> 0-100
    print(f"[{label}] HF reference GSM8K [{metric_key}] = {hf:.2f} (samples={n}, dataset={src})")

    trt, hard_path = _trtllm_score(n, cfg, metric_key)
    gap = abs(trt - hf)
    payload = {
        "label": label,
        "metric_key": metric_key,
        "num_samples": n,
        "hf_reference_score": round(hf, 4),
        "trtllm_score": round(float(trt), 4),
        "absolute_delta_points": round(gap, 4),
        "gap_tolerance": GAP_TOL,
        "cuda_graph": cfg.cuda_graph,
        "overlap_scheduler": cfg.overlap_scheduler,
        "cuda_graph_hard_path": hard_path,
        "shared_eval_config": shared_cfg,
        "shared_eval_config_hash": _config_hash(shared_cfg),
    }
    if artifact_name:
        _write_artifact(artifact_name, payload)
    print(
        f"[{label}] cfg={cfg} TRT-LLM(via trtllm-eval)[{metric_key}] score={trt:.2f} "
        f"hf={hf:.2f} |gap|={gap:.2f} (tol={GAP_TOL}) cuda_graph_hard_path={hard_path}"
    )
    if cfg.cuda_graph:
        assert hard_path, "enabled config did not exercise the CUDA-graph hard path"
    assert gap <= GAP_TOL, (
        f"{label} {cfg}: TRT-LLM {trt:.2f} vs HF {hf:.2f} gap {gap:.2f} > {GAP_TOL}"
    )
    return payload


# --------------------------------------------------------------------------- #
# Criterion: smoke  (LLM-API generation is deterministic and non-empty)
# --------------------------------------------------------------------------- #
@skip_no_cuda
@skip_no_ckpt
@pytest.mark.parametrize("cfg", CFGS)
def test_chatglm3_gsm8k_smoke(cfg: RuntimeCfg):
    llm = _build_llm(cfg)
    try:
        _assert_v2_and_backend(llm, cfg)
        params = SamplingParams(max_tokens=16, temperature=0.0, top_k=1)
        prompts = ["Question: What is 2 + 3?\nAnswer:", "Question: What is 10 - 4?\nAnswer:"]
        out1 = llm.generate(prompts, params)
        for o in out1:
            assert o.outputs[0].text.strip() != "", "empty generation"
            assert len(o.outputs[0].token_ids) > 0
        out2 = llm.generate(prompts, params)
        for a, b in zip(out1, out2):
            assert list(a.outputs[0].token_ids) == list(b.outputs[0].token_ids), "non-deterministic"
    finally:
        llm.shutdown()


# --------------------------------------------------------------------------- #
# Criterion: accuracy_canary  (short deterministic GSM8K slice, both configs)
# --------------------------------------------------------------------------- #
@skip_no_cuda
@skip_no_ckpt
@pytest.mark.parametrize("cfg", CFGS)
def test_chatglm3_gsm8k_accuracy_canary(cfg: RuntimeCfg):
    _run_config(CANARY_N, "accuracy_canary", cfg)


# --------------------------------------------------------------------------- #
# Criterion: full_trtllm_eval  (full GSM8K, writes score artifact; delta <= 2 pts)
# --------------------------------------------------------------------------- #
@skip_no_cuda
@skip_no_ckpt
@pytest.mark.parametrize("cfg", CFGS)
def test_chatglm3_gsm8k_full_trtllm_eval(cfg: RuntimeCfg):
    name = "cuda_graph" if cfg.cuda_graph else "baseline"
    _run_config(FULL_N, "full_trtllm_eval", cfg, artifact_name=f"chatglm3_gsm8k_{name}.json")
