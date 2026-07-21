# Kimi K3 ("golden prairie") in TensorRT-LLM — setup & reproduction

Runs the Kimi K3 model (93-layer hybrid KDA/MLA text core, 896 MXFP4 routed
experts with SiTU activation, ~1.5 TB HF checkpoint) end-to-end through the
TRT-LLM PyTorch backend, including a GSM8K eval.

## The two external inputs

Everything is parameterized by two paths (host paths; the sbatch scripts
mount them into the container at identical paths):

| Env var | Contents |
|---|---|
| `KIMI_K3_OPT_WORK_DIR` | the `exisiting_optimization_work` checkout: prior NVIDIA/Moonshot kernels, including `trtllmgen_MOE/` — the private FlashInfer snapshot + SiTU MXFP4 cubin pool used for the fused MoE |
| `KIMI_K3_MODEL_DIR` | the complete HF checkpoint (`goldenprairie-final-weights_vv1`): config + tokenizer + 96 safetensors shards |

Each sbatch script sets up the runtime env inline on every rank (before any
python import): persistent JIT caches and a node-local per-rank
`TRITON_CACHE_DIR` (the shared NFS `~/.triton` races across ranks). The
default fused MoE path (`KIMI_K3_FUSED_MOE=native`) uses the in-tree
TRTLLM-Gen SiTU cubins — no private flashinfer snapshot or
`FLASHINFER_PRIVATE_CUBIN_DIR` is needed; those are only required for the
legacy `KIMI_K3_FUSED_MOE=1` path (see the commented-out lines in the
sbatch scripts).

## Prerequisites

1. Build this repo for aarch64 (GB200/GB300) into `.venv-3.12` — standard
   editable build inside the 26.05 sbsa PyTorch container.
2. `pip install fla-core einops` into the venv (KDA prefill/decode kernels
   use FLA's Triton implementation).
3. (Legacy path only) When running `KIMI_K3_FUSED_MOE=1`, set up the
   flashinfer snapshot 3rdparty deps per
   `$KIMI_K3_OPT_WORK_DIR/trtllmgen_MOE/SNAPSHOT_SETUP.md` (clone the pinned
   `3rdparty/{cutlass,spdlog,cccl}` and create the `flashinfer/data/*`
   symlinks). Not needed for the default `native` path.

## Deployment shape

* **16 GPUs (4× GB300 trays), EP-only parallelism**: `tensor_parallel_size=16`
  is reinterpreted by the model as expert-parallel width — each rank holds
  the full ~70 GB bf16 non-expert model plus 896/16 = 56 experts per MoE
  layer (~90 GB MXFP4), with an allreduce of the routed latent partial sums.
* Fused MoE (`KIMI_K3_FUSED_MOE=native`, default): in-tree TRTLLM-Gen SiTU
  op (`mxe4m3_mxe2m1_block_scale_moe_runner`, W4A8 MXFP4 weights × MXFP8
  activations) consuming the checkpoint's MXFP4 weights via a one-time
  load-time shuffle. `KIMI_K3_FUSED_MOE=1` selects the legacy private
  flashinfer `trtllm_fp4_block_scale_routed_moe` path (W4A16, requires the
  snapshot env). Both are ~0.5–0.8 ms/layer vs ~60–140 ms for the
  reference dequant loop.
* Required LLM args (see `eval_extra_llm_options.yaml`): overlap scheduler
  off, CUDA graphs off, chunked prefill off, KV block reuse off,
  `tokens_per_block=64`. Keep `max_batch_size` ≤ ~32 (each KDA state slot
  costs ~455 MB across the 69 KDA layers).
* Unsupported: pipeline parallel, speculative decoding, disagg.

## Run

Preferred entry point: `examples/kimi_k3/run_kimi.py`, which resolves the
two paths + `IMAGE` from `~/.config/kimi-bringup.ini`. One-time setup:

```bash
cp examples/kimi_k3/kimi-bringup.ini.example ~/.config/kimi-bringup.ini
$EDITOR ~/.config/kimi-bringup.ini   # point `workspace` at your directory
```

Then:

```bash
examples/kimi_k3/run_kimi.py sanity        # 4-GPU / 4-layer pipeline check (~10 min)
examples/kimi_k3/run_kimi.py sanity-full   # 16-GPU full sanity (~40 min, mostly weight loading)
examples/kimi_k3/run_kimi.py gsm8k         # 16-GPU GSM8K, full 1319-problem test set (~3 hr)
```

Extra args are forwarded to sbatch; `--dry-run` prints the resolved inputs
and command without submitting. See `run_kimi.py --help` for the env-var /
config-key / workspace resolution rules.

The sbatch scripts can also be submitted directly, exporting
`KIMI_K3_OPT_WORK_DIR`, `KIMI_K3_MODEL_DIR`, and `IMAGE` yourself (they
have no defaults and fail fast if unset) — see each script's header. Both
default `REPO` to the submit directory — submit from the repo root, or set
`REPO` explicitly. Partition/account in the `#SBATCH` headers are cluster
defaults; override on the command line as needed.

## Debug knobs

* `KIMI_K3_NUM_LAYERS_OVERRIDE=<N>` — truncate to the first N decoder layers
  (loads only those shards; output is gibberish by construction, pipeline
  checks only).
* `KIMI_K3_FUSED_MOE=0` — fall back to the in-tree MXFP4 dequant reference
  MoE (bit-parity oracle for the fused paths, ~100× slower).
* `KIMI_K3_FUSED_MOE=1` — legacy private-flashinfer SiTU path (W4A16);
  requires the snapshot env and `FLASHINFER_PRIVATE_CUBIN_DIR`.
* `KIMI_K3_MLA_MAX_POSITIONS` (default 65536) — identity-RoPE table bound
  (K3 MLA is NoPE); raise for longer sequences.
* `TLLM_LOG_LEVEL_BY_MODULE="debug:_torch"` — verbose model-side logging.
