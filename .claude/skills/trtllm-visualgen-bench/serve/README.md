# Bringing up the server

Step 2 of `/trtllm-visualgen-bench`. Assumes GPUs and environment already exist. A server is worth starting on its own too — nothing here needs a workload. Script paths are relative to the skill directory, one level up.

| var | needed by | default |
|---|---|---|
| `LLM_MODELS_ROOT` | `--find` | **none — set it** to the checkpoint store to search |
| `TRTLLM_REPO` | `--find` | **none — set it.** Unset, the pipeline column degrades to the checkpoint's raw `_class_name` and says so |
| `LOGDIR` | steps B–C | `tmp` — parent of the per-run directory |

`--fields` needs an environment where `import tensorrt_llm` works; nothing else does.

```
A. resolve the model   →  checkpoint path
B. generate the config, into a fresh run directory
C. launch + wait for server ready
D. shut down
```

## A. Resolve the model

**`--model_path` is the answer when it was given.** Skip to step B.

Otherwise, **resolve it with `--find`, not with `ls` or a glob.** A path is only part of the answer: the
same walk reports the pipeline the checkpoint dispatches to and whether its weights are real,
and a directory listing tells you neither.

```bash
scripts/serve_check.py --find wan 14b fp4
# $LLM_MODELS_ROOT/Wan2.2-T2V-A14B-Diffusers-NVFP4   WanPipeline   ok
```

- More than one row → **ask the user which**. No words → list all.
- No rows → the words are wrong, not the store. They match the path, which spells the version as the checkpoint does: `wan22` finds nothing where `Wan2.2` is the directory. Widen a token; do not fall back to `ls`.
- The path may be a **file** inside a checkpoint dir (LTX-2 ships `-fp8`/`-fp4` alongside bf16).
- `LFS-STUB` → weights are pointer files; unusable.
- Words match the path relative to the store, so `--find cosmos3 fp8` reaches `Cosmos3-Nano-FP8/<release>/`. The walk stops three levels in; a store nesting deeper needs the path passed to `trtllm-serve` directly.
- Checkpoints outside `LLM_MODELS_ROOT` are not listed and serve fine.

## B. Generate the config

One directory per run, holding the config and the log:

```bash
RUN=${LOGDIR:-tmp}/vg-serve-$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
CFG=$RUN/<model>_<n>gpu.yaml        # e.g. wan22-t2v-a14b-nvfp4_8gpu.yaml
```

```
$LOGDIR/                              # tmp unless set
└── vg-serve-<stamp>/                 # one per launch
    ├── wan22-t2v-a14b-nvfp4_8gpu.yaml    # the config as served
    └── server.log                         # step C reads this to verify the launch
```

The directory name carries the timestamp; the config name carries the model and GPU count. The bench flow sets `LOGDIR` to nest this whole tree inside the bench run it belongs to; served on its own, it lands under `tmp`.

**`--server_config`, or a config the user named — an `examples/` yaml, their own file — is served as-is; go to step C.** Do not regenerate it, and do not add a `compilation_config` they did not ask for. A config carrying no warmup shapes compiles the first request of every shape inside the measured latency, which step C reports as a warning.

`--visual_gen_args` takes a YAML of `VisualGenArgs`. It is strict — an unknown or misspelled top-level key raises at load. Every field, with type, default and description:

```bash
scripts/serve_check.py --fields                      # every block
scripts/serve_check.py --fields parallel_config      # one block
scripts/serve_check.py --fields pipeline_config --model <checkpoint>
```

`pipeline_config` needs the checkpoint: its keys belong to the architecture, not to
`VisualGenArgs`, and differ per model — LTX-2 declares three, Cosmos3 none.

| block | what it sets |
|---|---|
| `model` · `revision` | checkpoint path or hub id (usually the CLI positional instead) |
| `parallel_config` | distributed parallelism: `cfg_size`, `ulysses_size`, `ring_size`, `attn2d_size`, `tp_size`, `parallel_vae_size` |
| `compilation_config` | warmup shapes — `resolutions` × `num_frames`, `skip_warmup` |
| `attention_config` | the attention backend, plus quantized- and sparse-attention recipes |
| `quant_config` | quantize at load; a ModelOpt-format dict or `QuantConfig` |
| `torch_compile_config` | torch.compile + `enable_autotune` |
| `cuda_graph_config` | CUDA graph capture/replay |
| `cpu_offload_config` | offload components to CPU — memory for latency |
| `cache_config` | step caching: `teacache` or `cache_dit` |
| `runtime_lora_config` | LoRA fused into the transformer at startup |
| `pipeline_config` | per-architecture knobs (LTX-2 text encoder, two-stage paths). Strict; unknown keys raise |
| `enable_layerwise_nvtx_marker` | per-layer NVTX hooks for nsys. **Never for a perf run** |

Empty config is valid for every model except FastWan; the rules below are what you actually have to decide.

**Parallel and attention — ask the user, never infer.** Both decide what the run measures, and a value nobody chose still ends up in the result. Take `cfg_size` / `ulysses_size` / `ring_size` / `attn2d_size` / `tp_size` / `parallel_vae_size` and the attention backend as given. Ask from `scripts/serve_check.py --fields parallel_config` and `--fields attention_config`, which print every field with its default, rather than from this sentence, which ages.

```
n_workers = cfg_size × cp_size × ulysses_size × tp_size
cp_size   = attn2d[0] × attn2d[1]  if > 1  |  ring_size if > 1  |  1     # mutually exclusive
```

`parallel_vae_size` is absent from that product: it splits the VAE across ranks
`n_workers` already launched.

**Warmup — derive from the bench workload, never from the pipeline default.**

```bash
scripts/serve_check.py --warmup-from <workload.yaml> >> $CFG
```

**Switches:**

| block | rule |
|---|---|
| `attention_config` | **Ask, as for parallel.** The backend decides what the run measures and the default is `VANILLA` (SDPA), which nothing overrides per model, so leaving it unset measures SDPA. `scripts/serve_check.py --fields attention_config` prints the backends and the `quant_attention_config` recipe beneath them. |
| `quant_config` | **Omit on a pre-quantized checkpoint.** Set it only to quantize at load from a BF16 base; omitting `dynamic` there defaults to `true`. |
| `cuda_graph_config` | Off by default. Enable **for LTX-2 only**. Leave `torch_compile` on (already on). |
| `cache_config` | Opt-in. `cache_dit` on LTX-2 silently forces the one-stage variant. |
| `parallel_config.async_ulysses` | Off by default. **On for Wan and LTX-2.** Needs `ulysses_size > 1`, and rejects `ring_size > 1`. |

**Per-model, required or startup fails.** None of these three failures names its own fix:

- FastWan → `compilation_config.skip_warmup: true`. It inherits Wan's 2-step warmup while its `forward()` takes exactly 3, so it dies on a step-count mismatch.
- LTX-2 → `pipeline_config.text_encoder_path: <local Gemma-3>`; the registry default is a hub id, useless offline. Both `spatial_upsampler_path` and `distilled_lora_path` are globbed out of the checkpoint dir, so a full LTX-2 checkout serves `LTX2TwoStagesPipeline` with an empty `pipeline_config`. Force one-stage with `TLLM_LTX2_FORCE_ONE_STAGE_PIPELINE=1`; the log's `Running warmup for <class>` line is what tells you which you got.
- Cosmos3 → `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` in the environment, or `cosmos_guardrail` installed. Installing the package also uninstalls opencv.

Apply them yourself when writing the config — no script does it for you.

```yaml
attention_config: {backend: TRTLLM}
parallel_config: {cfg_size: 2, ulysses_size: 4, parallel_vae_size: 8}
compilation_config:                                  # from --warmup-from
  resolutions: [[720, 1280]]
  num_frames: [81]
```

## C. Launch + wait for server ready

**Show what you derived and wait for a yes.** The checkpoint `--find` resolved and the config you
wrote in step B, both in full. What arrived as `--model_path` or `--server_config` is the user's
answer already and needs no round trip.

`serve_wait.py` checks the server against the config. Nothing checks the config against what the
user meant, so a checkpoint resolved from fuzzy words and a field nobody chose both verify clean
and reach the result. One `--find` row means the words were unambiguous in this store, not that
they were the words you wanted: where a single checkpoint carries `fp8`, `wan fp8` resolves to it
whichever mode the user meant.

**Single node — one process, any `n_workers` including 8 GPUs.** Do not wrap in `torchrun` or `srun --ntasks=N`.

`parallel_config` decides how many GPUs; `CUDA_VISIBLE_DEVICES` decides which. On a node you have to yourself, leave it unset. To run **several servers on one node**, set both it and `--port` per server — otherwise the second server answers on the first one's port and every measurement lands on the wrong model.

```bash
PORT=${PORT:-8000}
setsid trtllm-serve <checkpoint> --host 0.0.0.0 --port $PORT --visual_gen_args $CFG \
  > $RUN/server.log 2>&1 &
SERVER_PID=$!

scripts/serve_wait.py --log $RUN/server.log --config $CFG --port $PORT --pid $SERVER_PID
```

`setsid` is what step D needs: it makes the server lead its own process group, so
one signal reaches the workers it spawns and nothing else. Started without it the
server joins the caller's group, and signalling that group kills the caller too.

**Multi node — hold the allocation, drive it from the login shell.** Slurm below; on another launcher what carries over is that the task count equals `n_workers` and only rank 0 serves HTTP.

```bash
JID=<your allocation's job id>
N_WORKERS=<the product above>
export MASTER_ADDR=$(scontrol show hostnames "$(squeue -j $JID -h -o %N)" | head -n 1)
export MASTER_PORT=29500                        # rendezvous, not the HTTP port
srun -l --jobid=$JID --overlap --export=ALL \
  --ntasks=$N_WORKERS --ntasks-per-node=<gpus_per_node> \
  trtllm-serve <checkpoint> --host 0.0.0.0 --port ${PORT:-8000} --visual_gen_args $CFG \
  > $RUN/server.log 2>&1 &
SERVER_PID=$!

scripts/serve_wait.py --log $RUN/server.log --config $CFG \
  --host $MASTER_ADDR --port ${PORT:-8000} --pid $SERVER_PID
```

- `--jobid --overlap` mandatory; without `--jobid`, `srun` queues a new job.
- `--ntasks` = `n_workers`. `--ntasks-per-node` = GPUs per node is a run-time layout knob (the allocation only reserved the nodes): it pins task→node placement, and thus each rank's parallel-group co-location. A tight alloc (`n_workers`/gpus-per-node nodes) already block-distributes evenly; a looser one needs the flag.
- Only rank 0 serves HTTP, at `$MASTER_ADDR`.

`serve_wait.py` blocks until `/health` is 200, then checks the log against the config: world size, quantization algo, attention backend, and the shapes the server actually warmed against the ones `compilation_config` asked for. Non-zero on mismatch, server death, or a silent log (`--stall`, default 300s). Never poll the port — it opens before the model loads.

Then send one request at the warmed shape. Latency far above the rest → warmup missed it; fix step B.

A workload asking for `format: mp4` needs ffmpeg on this host; without it the request
comes back 400 naming the missing dependency.

## D. Shut down

`kill $SERVER_PID` leaves the `spawn` workers holding GPUs and the port, and `pgrep -f trtllm-serve` does not match them. Signal the group the `setsid` in step C created, which is the server and its workers and nothing else:

```bash
kill -TERM -"$SERVER_PID" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL -"$SERVER_PID" 2>/dev/null

nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ss -lptn "sport = :$PORT"
```

Both must come back empty before the next launch.

Do not select what to kill by who holds a GPU: on a shared node, or one running a
second server, that reaches processes this run never started.

A server left running answers `/health` instantly, so the next run's readiness check passes against the **previous** model and every number it reports belongs to that one.

Serving several models on one node at once: give each its own port **and** GPU set — two servers that pick the same GPUs contend or OOM, and two that pick the same port make the second one measure the first. Decide the split up front rather than having each launch choose: a server is absent from `nvidia-smi --query-compute-apps` until it builds a CUDA context, tens of seconds after launch, so two launches started together both see the same GPU free and both take it.

## Serving under nsys

`TLLM_PROFILE_VISUAL_GEN_START_STOP` scopes the capture window; the pipeline calls
`cudaProfilerStart`/`Stop` at its edges, which an attached collector honours.

| value | window |
|---|---|
| `A-B` · `A-B,C-D` · `A,B` | denoise steps; several ranges toggle the profiler per range |
| `predenoise` | request start through denoise-loop setup, so text encoding and latent prep. Fires once |
| `postdenoise` | end of a denoise loop to request completion, so VAE decode. Fires once |
| `all` | the whole request, text encoding through VAE decode, skipping warmup |
| unset | no profiler API calls; a plain `nsys profile` captures everything, warmup included |

```bash
TLLM_PROFILE_VISUAL_GEN_START_STOP=10-12 nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
  -o $RUN/serve/report trtllm-serve <checkpoint> --port $PORT --visual_gen_args $CFG
```

---

Example configs: `examples/visual_gen/configs/`, `examples/visual_gen/serve/configs/`, `tests/scripts/perf-sanity/visual_gen/`.
