# 🔥🚀⚡ LLM Compiler

**LLM Compiler** (Python package: `llmc`) is the standalone, lightweight distribution of the AutoDeploy graph-transformation pipeline. It exposes the same `ModelFactory` and `InferenceOptimizer` building blocks as TensorRT-LLM's AutoDeploy backend, but with a minimal dependency footprint (PyTorch + Triton + FlashInfer), so you can prototype optimization pipelines or run inference without a full TRT-LLM install.

The `llmc` package is **generated** from the AutoDeploy source tree inside the [TensorRT-LLM repo](https://github.com/NVIDIA/TensorRT-LLM). The standalone repo is **read-only** — see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for how to land changes.

For general AutoDeploy documentation (motivation, support matrix, feature overview), see the [official docs](https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/auto-deploy.html).

______________________________________________________________________

## Install

Install with [uv](https://docs.astral.sh/uv/) directly from the latest commit:

```bash
# https
uv pip install "git+https://github.com/NVIDIA/llm-compiler.git"

# ssh
uv pip install "git+ssh://git@github.com/NVIDIA/llm-compiler.git"
```

For development, clone the repo and do an editable install:

```bash
# https
git clone https://github.com/NVIDIA/llm-compiler.git
# or ssh
git clone git@github.com:NVIDIA/llm-compiler.git

cd llm-compiler
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Sanity check

```python
from llmc._compat import TRTLLM_AVAILABLE
print(f"TRT-LLM available: {TRTLLM_AVAILABLE}")  # False in standalone mode
```

In standalone mode the package uses the PyTorch, Triton, and FlashInfer kernel paths. TRT-LLM-only kernels (custom CUDA, optimized all-reduce, MoE fused kernels, the `pyexecutor` runtime) are skipped at registration time.

### Run the bundled tests

```bash
pytest tests/
```

______________________________________________________________________

## Building a custom inference pipeline

The two core abstractions in `llmc` are:

- **`ModelFactory`** — wraps a HuggingFace checkpoint (or any other source) and produces an initialized `nn.Module` plus dynamic-shape metadata for export.
- **`InferenceOptimizer`** — runs a configured pipeline of graph transforms (export → fuse → quantize → shard → compile → cache) over a model produced by a factory, against a `CachedSequenceInterface` that defines the input contract.

The code below builds a custom optimization pipeline end-to-end, without going through the higher-level `LLM` API. It's the same machinery `LLM(...)` uses internally — direct access is useful when you want to insert your own transforms, target ONNX/MLIR export, or drive a non-standard runtime.

```python
import torch

from llmc.models.factory import ModelFactoryRegistry
from llmc.shim.interface import CachedSequenceInterface
from llmc.transform.optimizer import InferenceOptimizer
from llmc.utils.dist_config import DistConfig

# 1. Build a ModelFactory.
#
#    The registry is populated at import time. Built-in factories include
#    "AutoModelForCausalLM" (text generation) and
#    "AutoModelForImageTextToText" (multimodal). Factories know how to
#    resolve a HF model id, instantiate the architecture, optionally load
#    weights, and report dynamic shapes for export.
factory_cls = ModelFactoryRegistry.get("AutoModelForCausalLM")
factory = factory_cls(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={"torch_dtype": "float16"},
    skip_loading_weights=False,  # set True to inspect graph without weights
    max_seq_len=2048,
)

# 2. Define the runtime input contract via a CachedSequenceInterface.
#
#    This object owns the dummy inputs (input_ids, position_ids, ...) used
#    during export and tracing, and — when running with a TRT-LLM-style
#    KV cache — manages cache tensors. In standalone mode the cache
#    manager is a no-op; CachedSequenceInterface still drives the
#    transform pipeline.
cache_seq = CachedSequenceInterface(
    max_seq_len=2048,
    max_batch_size=4,
    device="cuda",
    kv_cache_config=None,           # or llmc._compat.KvCacheConfig(...)
    max_num_tokens=4 * 2048,
    vocab_size_padded=factory.vocab_size_padded,
)

# 3. Describe the transform pipeline.
#
#    Each entry maps a registered transform name to its config.
#    `stage` controls ordering across the pipeline; transforms are
#    re-sorted by stage at construction time. The names below are
#    illustrative — see `llmc.transform.library` for the full set.
transforms_config = {
    "export_to_gm":         {"stage": "export"},
    "fuse_gemms":           {"stage": "post_export"},
    "fuse_rmsnorm":         {"stage": "post_export"},
    "shard_attention":      {"stage": "sharding", "world_size": 1},
    "compile_model":        {"stage": "compile",  "backend": "torch-simple"},
    "initialize_cache":     {"stage": "cache_init"},
}

# 4. Build the optimizer and run it.
#
#    `dist_config` is required when sharding across devices; for a single
#    GPU the default (world_size=1) is fine.
optimizer = InferenceOptimizer(
    factory=factory,
    config=transforms_config,
    dist_config=DistConfig(world_size=1, rank=0, tp_size=1),
)
optimized_model = optimizer(cache_seq)

# 5. Run the optimized model.
#
#    The exact call signature depends on the runtime your pipeline
#    targets — for prefill-style cached attention, the inputs are
#    (input_ids, position_ids) plus the cache state owned by `cache_seq`.
input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
position_ids = torch.arange(input_ids.size(1), device="cuda").unsqueeze(0)
with torch.inference_mode():
    logits = optimized_model(input_ids, position_ids)
print(logits.shape)
```

### Where to look next

| Topic | Path |
|-------|------|
| Available transforms | `llmc/transform/library/` |
| Available factories  | `llmc/models/`, `llmc/models/custom/` |
| Custom ops & backends | `llmc/custom_ops/` |
| Compile backends | `llmc/compile/backends/` |
| Type compatibility shims | `llmc/_compat.py` |
| Configuration data classes | `llmc/transform/interface.py`, `llmc/llm_args.py` |

For higher-level usage, the `llmc.LLM` class (available when running inside TRT-LLM) wraps all of the above behind a familiar generate API. In pure-standalone mode use `InferenceOptimizer` directly as shown.
