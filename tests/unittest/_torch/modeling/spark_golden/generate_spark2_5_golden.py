# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the Spark2_5 golden parity artifact (spark2_5_golden.pt).

The golden pins the TRT-LLM Spark2_5 implementation against the HF reference
implementation shipped in the XHToken checkpoints (``modeling_spark.py`` /
``configuration_spark.py``, e.g. from ``XHToken/Spark-X2.6-4B`` or
``XHToken/Spark-X2.5-1.7B`` on HuggingFace), so the parity test in
``tests/unittest/_torch/modeling/test_modeling_spark.py`` runs in CI without
network access or the reference sources.

Usage:
    SPARK_HF_REFERENCE_DIR=/path/to/reference/files \
        python3 tests/unittest/_torch/modeling/spark_golden/generate_spark2_5_golden.py

The artifact stores the tiny config, the reference-initialized weights (in
HF checkpoint naming, including the fused ``q_k_v_proj``), the input ids, and
the reference fp32 logits. Regenerate only when the reference implementation
changes; the test does not depend on RNG stability because the weights are
committed.
"""

import hashlib
import importlib.util
import inspect
import os
import sys
import types

import torch

# Deterministic ultra-tiny variant of the real config. Two layers cover both
# layer types; seq len (12) exceeds the sliding window (4) so windowed and
# full attention both matter; head_dim 32 with partial factor 0.25 exercises
# an 8-dim rotary slice.
GOLDEN_CONFIG = {
    "architectures": ["Spark2_5ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "gate_attn_act_mode": "sigmoid",
    "head_dim": 32,
    "headwise_attn_output_gate": True,
    "hidden_act": "gelu",
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 128,
    "layer_types": ["sliding_attention", "full_attention"],
    "max_position_embeddings": 4096,
    "mlp_bias": False,
    "model_type": "spark2_5",
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "num_key_value_heads": 1,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 5000000,
        },
        "sliding_attention": {
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000,
        },
    },
    "sliding_window": 4,
    "tie_word_embeddings": True,
    "use_cache": False,
    "vocab_size": 64,
    "dtype": "float32",
}

INPUT_IDS = [5, 17, 3, 42, 8, 61, 2, 33, 11, 27, 4, 19]
SEED = 20261104


def load_reference(ref_dir: str):
    """Import the checkpoint's reference modules as a synthetic package."""
    pkg = types.ModuleType("spark_ref_gen")
    pkg.__path__ = [ref_dir]
    sys.modules["spark_ref_gen"] = pkg
    modules = {}
    for name in ("configuration_spark", "modeling_spark"):
        path = os.path.join(ref_dir, f"{name}.py")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"reference file not found: {path}")
        full_name = f"spark_ref_gen.{name}"
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        modules[name] = module

    modeling = modules["modeling_spark"]
    import transformers

    # The reference targets transformers 4.57: _tied_weights_keys is a list
    # (5.x requires the {target: source} dict) and create_causal_mask takes
    # `input_embeds` / `cache_position` (5.x renamed / dropped them).
    tied = getattr(modeling.Spark2_5ForCausalLM, "_tied_weights_keys", None)
    if int(transformers.__version__.split(".")[0]) >= 5 and isinstance(tied, (list, tuple)):
        modeling.Spark2_5ForCausalLM._tied_weights_keys = {
            "lm_head.weight": "model.embedding.weight"
        }

    def _compat_mask_fn(fn):
        sig = inspect.signature(fn)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        def wrapped(**kwargs):
            if "input_embeds" not in params and "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            elif "inputs_embeds" not in params and "inputs_embeds" in kwargs:
                kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
            if not accepts_var_kw:
                kwargs = {k: v for k, v in kwargs.items() if k in params}
            return fn(**kwargs)

        return wrapped

    for fn_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        if hasattr(modeling, fn_name):
            setattr(modeling, fn_name, _compat_mask_fn(getattr(modeling, fn_name)))

    return modules["configuration_spark"], modeling


def main() -> None:
    ref_dir = os.environ.get("SPARK_HF_REFERENCE_DIR")
    if not ref_dir:
        raise SystemExit(
            "SPARK_HF_REFERENCE_DIR must point at a directory containing the "
            "checkpoint's modeling_spark.py and configuration_spark.py"
        )
    cfg_mod, modeling_mod = load_reference(ref_dir)

    # CPU + fixed seed; the weights are committed so tests never re-roll them.
    torch.manual_seed(SEED)
    ref_config = cfg_mod.Spark2_5Config(**GOLDEN_CONFIG)
    ref_model = modeling_mod.Spark2_5ForCausalLM(ref_config).to(torch.float32).eval()

    with torch.no_grad():
        logits = ref_model.forward(
            input_ids=torch.tensor([INPUT_IDS], dtype=torch.long),
            use_cache=False,
        ).logits[0]
    assert torch.isfinite(logits).all(), "reference logits contain non-finite values"

    # Real checkpoints carry no lm_head weight (tied embeddings).
    state_dict = {
        k: v.detach().clone()
        for k, v in ref_model.state_dict().items()
        if not k.startswith("lm_head")
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spark2_5_golden.pt")
    torch.save(
        {
            "config": GOLDEN_CONFIG,
            "state_dict": state_dict,
            "input_ids": INPUT_IDS,
            "logits": logits.detach().clone(),
            "provenance": {
                "seed": SEED,
                "reference_dir": os.path.basename(ref_dir.rstrip("/")),
                # Plain str: TorchVersion is a str subclass that
                # torch.load(weights_only=True) refuses to unpickle.
                "torch_version": str(torch.__version__),
                "transformers_version": str(__import__("transformers").__version__),
            },
        },
        out_path,
    )
    size = os.path.getsize(out_path)
    digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    print(f"wrote {out_path} ({size} bytes, sha256={digest[:16]}...)")
    if size > 500 * 1024:
        raise SystemExit("golden exceeds the 500KB check-added-large-files limit")


if __name__ == "__main__":
    main()
