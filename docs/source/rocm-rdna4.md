# RDNA4 / ROCm backend

This fork provides a separate Linux ROCm inference backend for **gfx1200 and gfx1201**,
plus process-wide, opt-in performance reporting. It does **not** make TensorRT engine
plans, NVIDIA CUDA/PTX kernels, or the upstream distributed executor run on AMD.
The original CUDA build remains available; its installation instructions are not
ROCm installation instructions.

## Status and support boundary

The HIP sources and Python integration are implemented. CPU reference/API tests run
without downloading a model. **Native HIP compilation, GPU correctness and performance
have not been verified in the development sandbox.** Run the local qualification steps
below on your RDNA4 host. CPU CI is not evidence of GPU correctness, and this is not a
claim of complete NVIDIA feature parity.

| Capability | ROCm implementation / limitation |
| --- | --- |
| GPU/runtime | Linux, real gfx1200/gfx1201, RDNA4-compatible HIP PyTorch. Other architectures and `HSA_OVERRIDE_GFX_VERSION` are rejected. |
| Model loading | Unquantized Hugging Face `AutoModelForCausalLM` checkpoints or an existing model/tokenizer. Remote model code is disabled by default. |
| Execution | Single GPU; batched causal generation; ordinary HF cache; explicit context/batch limits; FP32, FP16, BF16 GPU storage. |
| Python entry points | `tensorrt_llm.LLM`, `tensorrt_llm.llmapi.LLM`, `tensorrt_llm._torch.LLM` and portable `SamplingParams`/outputs. The portable API is smaller than the upstream executor API. |
| Default kernels | ROCm PyTorch operators and ROCm BLAS; SDPA or eager attention. No NVIDIA-only dependencies are required. |
| Native HIP kernels | RMSNorm, add+RMSNorm, one-dimensional affine LayerNorm, SiLU/GELU gating, partial rotary embedding and causal/noncausal/GQA attention. FP32 reductions/attention accumulation; output storage rounding. |
| Native model integration | `kernels="hip"` replaces compatible Llama/Mistral/Qwen2/Qwen3/Phi3 RMSNorms and affine 1D LayerNorms. `attn_backend="hip"` additionally uses native attention for Llama/Mistral/Qwen2/Qwen3 with head dimensions ≤256. |
| Rotary / gating primitives | Available through `tensorrt_llm.rocm.ops`; not automatically substituted into every HF model implementation. |
| Sampling | Greedy, temperature/top-p/top-k, repetition penalty, seed, multiple completions, HF beam search, EOS and decoded stop strings. No logprob materialization. |
| Serving | Non-streaming OpenAI-compatible completions/chat, health and model discovery. Generation is serialized; this is not an in-flight batching scheduler. |
| Benchmarks | Real serial latency/throughput, excluding warmup; includes tokenization, transfers, generation and detokenization. Not TTFT or streaming token latency. |
| Profiling | Sherlock-style component table, per-file/function host data, HIP/CUDA event spans, CPU/RAM/GPU/VRAM sampling, JSON/text/Chrome/cProfile outputs. |
| CPU mode | Explicit `device="cpu"` reference/testing mode, FP32 by default. It is never a fallback for an unavailable GPU. Half-precision model execution depends on your CPU PyTorch build. |
| Not ported | TensorRT plans/plugins/builders; CUDA graphs; FlashInfer/CUTLASS/PTX/CDNA-specific fused code; quantized engines; tensor/pipeline/expert parallelism; RCCL/disaggregated cache exchange; paged/native KV manager; speculative decoding; LoRA; VisualGen/multimodal; upstream AsyncLLM, streaming, guided decoding and evaluation CLI. Unsupported options fail rather than being silently ignored. |

Models must fit in the selected device's VRAM. An HF model relying on custom CUDA
extensions or unsupported SDPA is not made portable merely by loading its weights.
Select `attn_backend="eager"` for models without SDPA; custom remote code and model
families outside the native integration list need their own qualification.

## Dependency review and trusted checkpoints

The 2026-10-08 review resolved `requirements-rocm.txt` with `pip-audit` using the
PyPI vulnerability service. It scanned 31 packages and reported **9 findings
covering 6 distinct advisory IDs in Transformers 4.57.6** (some PyPI records are
duplicated). No other package in that resolution was flagged. This is a snapshot
of one resolution, not a claim that every version allowed by the requirements is
safe. User-installed ROCm PyTorch and its native libraries require a separate
review; the ROCm requirements deliberately do not resolve or replace Torch.

| Advisory ID | Affected Transformers path |
| --- | --- |
| PYSEC-2025-217 / CVE-2025-14929 | X-CLIP checkpoint conversion deserialization |
| PYSEC-2026-2288 / CVE-2026-1839 | Trainer RNG-state checkpoint loading |
| PYSEC-2026-2289 / CVE-2026-4372 | Model configuration selecting a remote attention kernel, bypassing remote-code consent |
| PYSEC-2026-2290 / CVE-2026-5241 | LightGlue nested configuration overriding remote-code consent |
| PYSEC-2026-3929 / CVE-2026-9856 | Tokenizer/processor chat-template filename traversal on save |
| PYSEC-2026-4174 / CVE-2026-80047 | Custom generation module written to cache before remote-code consent |

**The Transformers 4.x dependency is not vulnerability-free.** The fixed versions
listed by the service for several advisories are in Transformers 5.x (including
5.3.0 and 5.10.0); some records list no fixed version. The backend's attention and
mask integration is currently qualified only against 4.x. A 5.x migration needs
separate API/parity qualification rather than silently loosening that major-version
constraint to make an audit appear green.

Use only trusted model weights, configuration and tokenizer files, prefer
safetensors, and isolate inference from credentials and other sensitive files.
`trust_remote_code=False` and `--local-files-only` **do not establish a security
boundary** against the published configuration/cache vulnerabilities. Do not load
untrusted checkpoints with this 4.x dependency; local-only loading does not make an
already malicious local checkpoint safe. Trainer, X-CLIP conversion and LightGlue
are not used by this causal-inference backend, but that does not dismiss the
model-loading and tokenizer advisories.

Resolved PyPI license metadata declares MIT, BSD/0BSD, Apache-2.0, PSF-2.0,
CNRI-Python, Zlib, CC0-1.0 and MPL-2.0 terms. In particular, certifi declares MPL-2.0
and tqdm declares MPL-2.0/MIT; redistributors must retain notices and comply with
applicable component-license obligations. Metadata review is not legal approval.
To repeat the vulnerability review in a disposable environment:

```bash
python -m pip install pip-audit
python -m pip_audit -r requirements-rocm.txt --vulnerability-service pypi
```

## Install without CUDA/TensorRT build stages

1. Install AMD's Linux driver/ROCm stack and **a HIP PyTorch wheel explicitly supporting
   your RDNA4 architecture**, following AMD's current Radeon support matrix. A matching
   ROCm 7.x SDK/runtime and its recommended PyTorch wheel are the intended native-HIP
   environment. `torch.version.hip` alone does not establish that a wheel contains
   gfx1200/gfx1201 operator code. Do not replace it with PyPI's default CUDA Torch wheel.
2. Ensure the user has permission to access `/dev/kfd` and `/dev/dri` (commonly the
   `render` and `video` groups). Remove architecture-spoofing overrides.
3. In this repository and that Python environment:

   ```bash
   python -m pip install -r requirements-rocm.txt
   TRTLLM_BUILD_BACKEND=rocm python -m pip install --no-deps -e .
   trtllm-rdna4 doctor
   ```

The ROCm packaging path ships Python code and HIP source files, **not compiled
binaries**. It bypasses the upstream TensorRT/CUDA/CUTLASS/native-binding build and
never installs Torch itself. To package the same source for distribution:

```bash
python -m pip install build
TRTLLM_BUILD_BACKEND=rocm python -m build --wheel
```

`TRTLLM_BACKEND=auto` selects ROCm when PyTorch reports HIP, otherwise CUDA.
`TRTLLM_BACKEND=rocm` can explicitly select the portable backend for CPU tests with
CPU-only or CUDA Torch. It does not authorize NVIDIA GPU execution as ROCm.
`trtllm-rdna4` selects ROCm before importing the library, so its doctor command can
also diagnose a wrongly installed Torch wheel without loading TensorRT bindings.

PyTorch deliberately calls HIP devices **`cuda:N`** and exposes stream/event APIs
under `torch.cuda`. These names do not mean NVIDIA code is used. On a mixed-architecture
host, restrict visible devices to RDNA4 using `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`;
doctor's ready flag checks all visible devices. Logical device indices can differ
from physical DRM/PCI identities in utilization reports.

## Generate

```bash
trtllm-rdna4 generate --model /path/to/hf-checkpoint \
  --local-files-only --device cuda:0 --dtype bfloat16 \
  --prompt 'Explain wavefront execution briefly.' --max-tokens 64
```

Python API:

```python
from tensorrt_llm import LLM, SamplingParams

with LLM(model="/path/to/hf-checkpoint", max_batch_size=2,
         device="cuda:0", dtype="bfloat16", kernels="torch") as llm:
    results = llm.generate(
        ["Explain attention.", "Explain a wavefront."],
        SamplingParams(temperature=0, max_tokens=64),
    )
    for result in results:
        print(result.outputs[0].text)
```

To request native HIP norm and attention integration explicitly:

```bash
MAX_JOBS=2 trtllm-rdna4 generate --model /path/to/llama-checkpoint \
  --local-files-only --kernels hip --attn-backend hip \
  --prompt 'Hello' --max-tokens 32
```

Only an explicit `kernels="hip"`/`--kernels hip` request invokes the local JIT compiler.
Use a matching ROCm SDK/`hipcc`, host C++ compiler and Ninja; set `ROCM_HOME` if needed.
Builds use PyTorch's extension cache and compile for both gfx1200 and gfx1201 with
wave32 and no fast-math. No SDK/compiler is needed for the default `kernels="torch"`
path. Native attention is correctness-first online softmax, **not a tuned flash/WMMA
attention implementation**; use SDPA as the default performance path. Native ops are
inference-only and do not implement autograd or dropout.

## Serve and benchmark

```bash
# Optional bearer authentication. Set a real, private key in your environment.
export TRTLLM_API_KEY='your-private-key'
trtllm-serve /path/to/hf-checkpoint --local-files-only \
  --host 0.0.0.0 --port 8000 --served-model-name local-rdna4

trtllm-bench --model /path/to/hf-checkpoint throughput --local-files-only \
  --prompt 'Explain ROCm.' --max-tokens 64 --warmup 2 --iterations 5 \
  --output profiles/benchmark.json
```

Endpoints: `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`.
Chat requires a tokenizer chat template. Unknown request fields, streaming and
unsupported sampling/logprob features are rejected. The server binds to loopback
(`127.0.0.1`) by default; use `--host 0.0.0.0` only when external access is intended.
It has no TLS and no
production scheduler; put authentication/TLS/rate limiting in front of it when
exposing it outside a trusted machine. Without `TRTLLM_API_KEY`, it is unauthenticated.

Benchmark prompts may also come from JSONL `--dataset` records. Token throughput
uses the returned completion token IDs (terminal EOS is excluded); benchmark JSON
records the scope, elapsed time, output tokens and latency distribution. Seeds
are isolated from the caller's RNG state. Compare unprofiled runs for headline
performance: profiling intentionally adds substantial overhead.

## Profile any executed Python source file

For entry points/scripts importing `tensorrt_llm` before application argument parsing,
append `--profile`. The bootstrap consumes profiler arguments before the application's
own parser runs; instrumentation is off by default.

```bash
trtllm-rdna4 generate --model /path/to/hf-checkpoint --local-files-only \
  --prompt 'Hello' --max-tokens 32 \
  --profile --profile-output profiles/generation --profile-interval 0.1

trtllm-bench --model /path/to/hf-checkpoint latency --local-files-only \
  --prompt 'Hello' --iterations 3 --profile --profile-output profiles/bench
```

For an unmodified script, including one that does not import this library, use the
standalone launcher. Its `--` separator protects application arguments:

```bash
python scripts/profile.py examples/rocm/llm_inference.py \
  --profile --profile-output profiles/example -- --model /path/to/hf-checkpoint
# Installed equivalent:
trtllm-profile path/to/your_script.py --profile --profile-output profiles/custom \
  -- --your-application-option value
```

Profiling cannot make an otherwise CUDA-only example execute on ROCm: use only
examples/options compatible with the support matrix above. The launcher preserves
script arguments, exceptions and exit codes. Direct-import bootstrapping does not
know the application's exit code; its report labels that status as unknown.

Options:

| Flag | Meaning |
| --- | --- |
| `--profile` | Enable recording for this process/run. |
| `--profile-output PREFIX` | Write `PREFIX.json`, `.txt`, `.trace.json`, `.prof`; default is a timestamp/PID prefix under `profiles/`. |
| `--profile-interval SECONDS` | CPU/RAM/GPU/VRAM sampling interval, minimum 0.01 s; default 0.1 s. |
| `--profile-max-ops COUNT` | Bound stored operation/event spans; default 100,000. Truncation is explicitly reported. Per-file host profiling and full-run telemetry aggregates continue. |
| `--profile-no-tensors` | Keep source-file/function host profiling and telemetry; disable tensor-dispatch tracing/device spans. |

Reports show the requested component columns:

```text
component                ops      %dev          dev us         idle us         host us
```

Components include attention, projections, recurrent, attention-mix, head+sample,
bias, transfer, norms, residual, ffn-activate, embed and other. HF module scopes and
operator-name heuristics attribute operations; custom code can add scopes:

```python
from trtllm_profile import component, trace_active

with component("attention-mix"):
    # Your existing tensor operations.
    ...

# Use inside each worker thread whose execution should be included:
with trace_active():
    ...
```

The supplied server enrolls its generation worker. Python profiling/Torch dispatch
contexts are thread-local: arbitrary application threads must enroll explicitly.
Subprocesses need independent activation and output prefixes; a parent profiler
inherited by fork never writes a child's records into the parent's output. For
GPU multiprocessing use the usual PyTorch-supported spawn start method. Join
workers before ending the profiling session. Unexecuted files have no measurements;
there are no invasive instrumentation edits to every source file.

### Timing and memory semantics

- Host per-file/function data uses cProfile wall time. JSON and `.prof` contain all
  observed source files/functions; text shows the top 20. Self and inclusive times
  are separate; inclusive times overlap and must not be summed as run time.
- `ops` counts **Torch dispatch operations**, not ISA instructions or necessarily
  individual kernels. Device time is a begin/end event span on the current HIP/CUDA
  stream. Views, host enqueue delay, transfers and backend-internal work can occur
  within spans. Work on unobserved side streams/native APIs cannot automatically
  be reconstructed as a kernel trace.
- `%dev` is the share of recorded device-span time, **not GPU utilization**. Concurrent
  stream spans can overlap. `idle us` is the nonnegative gap between consecutive
  recorded spans on a stream, not a hardware-wide idle counter. Device clock alignment
  to the host Chrome timeline is an event-marker approximation.
- The instrumentation floor executes **256 empty operations through the same
  begin/end path** and reports host/device microseconds per operation. It is diagnostic,
  hardware-dependent and **never subtracted** from measurements. No example timing
  values are hard-coded. CPU device times are `N/A`, not invented GPU timings.
- Native RDNA4 operators are attributed to `rocm/kernels/csrc/rdna4Ops.hip` at line 0
  (file-level attribution). This is not native line/instruction profiling or separate
  timing of inlined headers. ROCm library-generated shader internals remain library
  work attributed to the dispatch call.
- CPU system utilization, process CPU utilization, process RSS/RAM percentage and
  system RAM used/total are sampled via psutil, with Linux procfs fallbacks. Process
  CPU can exceed 100% when multiple cores are used.
- AMD GPU busy percentage and physical VRAM used/total/percentage come from readable
  amdgpu DRM sysfs counters; optional NVML covers NVIDIA cards. Physical PCI/DRM or
  UUID identities are retained instead of guessing their correspondence to HIP
  logical indices. These are whole-device/system counters, not exclusive to this process.
- Sampled mean/peak aggregates cover the full run even when raw samples are capped.
  Sampling can miss brief peaks. Missing driver counters/permissions are `N/A`, never
  zero-filled. Memory sizes in text are GiB; JSON sizes are bytes.
- Logical-device PyTorch allocated/reserved memory is reported separately. Its
  allocator peak is a **process-lifetime high-water mark**, not a reset per-run peak.
  Hardware VRAM usage is not the same as allocator allocation/reservation.
- GPU errors discard unavailable device timings and retain host/telemetry diagnostics
  where possible. A failing command is not converted into success by profiling.
  Report files can contain local paths; review them before sharing.

## RDNA4 ISA design

The design follows AMD's [RDNA4 ISA Reference Guide, 7 April 2025](https://gpuopen.com/download/rdna4-instruction-set-architecture.pdf),
not NVIDIA warp/PTX layouts or CDNA MFMA assumptions:

- `wave32.h` fixes the shader wavefront to **32 lanes**. HIP shuffle reductions use
  width 32 and full participation, never a 64-lane ballot or a NVIDIA `__shfl_sync`
  mask. Compile flags enforce wave32 and the header rejects a conflicting known
  wavefront-size macro. RDNA4's wave32-only VOPD may be selected by the compiler;
  code does not incorrectly apply it to wave64.
- Blocks contain **256 threads / eight complete wave32s**, below the ISA's 1024-item
  workgroup limit. Cross-wave sums use eight FP32 LDS slots and uniform workgroup
  barriers. Out-of-range data contributes zero; no lane skips a reduction barrier.
  This is far below the 64-KiB workgroup LDS limit (a WGP has 128 KiB across 64 banks).
- Reductions, centered variance and online-softmax attention accumulate in FP32;
  FP16/BF16 are storage formats. Fused residual addition rounds to storage before
  normalization. RMSNorm preserves the storage-rounding boundary before weighting.
  Positive finite FP32 epsilon/finite FP32 scale, operand shape/device/dtype, GQA
  divisibility, head limits and inference-only behavior are checked.
- Native attention streams keys through an online max/sum update, avoiding a
  materialized score matrix. Its mask contract is bool `True=allowed` or additive
  logits bias; causal positions are right-aligned for cached decode. Fully masked
  `-inf` rows return zero. Other nonfinite input behavior is not silently repaired.
- Kernel launches use the **actual PyTorch current HIP stream** with a device guard
  and HIP launch-error checks, not the default stream. Tensor memory ownership
  remains with PyTorch. Native source registration uses the CUDA dispatch key because
  that is HIP PyTorch's documented device convention.
- Matrix projections are delegated to ROCm PyTorch/BLAS. RDNA4 WMMA/SWMMAC fragment
  maps differ from CDNA MFMA and from later gfx1250 instruction shapes. This code
  deliberately does not invent an unverified mapping or reuse NVIDIA MMA fragments.
  See the [ROCm matrix-instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator).
- HIP/LLVM generates code objects, register allocation, hazard waits, LDS barriers
  and required shader padding. These are compiler/loader responsibilities, not
  manually emitted machine-code bytes. Source-level ISA-aware design is not a
  substitute for checking the emitted code on your chosen compiler/runtime.

## Local qualification (no downloaded model required)

Reference/API checks on any CPU host:

```bash
python -m pip install pytest httpx
TRTLLM_BACKEND=rocm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/rocm --confcutdir=tests/rocm -q
TRTLLM_BACKEND=rocm python -m tensorrt_llm.rocm.validation \
  --device cpu --dtype all --output profiles/cpu-validation.json
```

`--confcutdir` isolates this suite from upstream CUDA/MPI-only pytest plugins.
Native HIP tests are skipped when no real RDNA4 GPU is available. The dedicated
GitHub Actions CPU job consumes `l0_rdna4_cpu.yml`; its optional, manually requested
GPU job needs a preconfigured self-hosted `rdna4` runner with HIP PyTorch and the SDK.

On the target RDNA4 machine:

```bash
trtllm-rdna4 doctor
MAX_JOBS=2 trtllm-rdna4 validate --device cuda:0 --kernels hip --dtype all \
  --profile --profile-output profiles/native-validation \
  --output profiles/native-validation-results.json
MAX_JOBS=2 python -m pytest tests/rocm/test_hip.py --confcutdir=tests/rocm -q
```

The validator checks dtype-specific primitive tolerances, non-multiple-of-wave
widths, normalization/residual storage rounding, partial/interleaved rotary,
masked/GQA/cached-decode attention, projections, and a seeded offline tiny Llama's
prefill logits and greedy-token agreement against the CPU reference. Its model
comparison is FP32; half-storage coverage is in the primitive cases. It reports
whether native kernels/model norm replacements actually executed and returns a
failure exit code on mismatch. The separate stream test executes on a nondefault
stream. These are qualification starting points, not coverage of every large model,
compiler version or deployment workload. Save doctor output, qualification reports
and your real-model results before relying on native HIP inference.
