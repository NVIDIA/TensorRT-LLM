# Benchmarking a VisualGen server

`tensorrt_llm.serve.scripts.benchmark_visual_gen` drives a running `trtllm-serve` VisualGen
server over its OpenAI-compatible routes and reports latency and throughput.

```bash
python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
    --workload workload.yaml --port 8000 \
    --max-concurrency 1 --save-result --save-detailed
```

This page repeats no field description. Each lives with its definition:

| what | where |
|---|---|
| every flag, with its default | `--help` |
| what a generation parameter means | `VisualGenParams` in [`visual_gen/params.py`](../../visual_gen/params.py) |
| which route validates which field | `ImageGenerationRequest` · `ImageEditRequest` · `VideoGenerationRequest` in [`openai_protocol.py`](../openai_protocol.py) |
| what a result-JSON key holds | `VisualGenBenchResult` and `VisualGenRequestRecord` in [`benchmark_visual_gen.py`](benchmark_visual_gen.py) |

## Benchmarking args

`--help` lists them under five groups, each answering one question:

* **Connection** — where the server is, and what it serves.
* **Workload** — what the run sends. The document arrives as `--workload` or spelled out as
  field flags, never both.
* **Traffic** — when requests are issued.
* **Execution** — how the client drives the run, and how the media comes back.
* **Results** — what the run writes down.

Four of them are their flags and nothing more. Workload is the one with a shape of its own.

### Workload

A YAML or JSON file, the same document inline (starting with `{` or `[`), or a bare list of
requests, named by `--workload`.

```yaml
backend: openai-videos                    # openai-videos | openai-images | openai-image-edits

common_params:                            # applies to every request
  width: 1280
  height: 720
  num_frames: 81
  extra_params:                           # per-pipeline knobs
    output_type: video

requests:
  - prompt: A red fox trotting across a snowy field at dawn
  - prompt_file: prompts/aerial.json      # instead of prompt
    image_reference: ../media/frame.png
    width: 720                            # overrides common_params for this request
    height: 1280
```

`backend` sits at the top level and selects the route, and so what the run measures.
Everything else is a `VisualGenParams` field, `prompt`, `prompt_file`, or `extra_params`.

#### Route matrix

Derived at load from the request model the route validates against, so naming a field the
route cannot carry is an error rather than a request the server ignores.

| field | `openai-videos` | `openai-images` | `openai-image-edits` |
|---|:---:|:---:|:---:|
| `width` · `height` · `seed` | ✓ | ✓ | ✓ |
| `num_inference_steps` · `guidance_scale` · `max_sequence_length` | ✓ | ✓ | ✓ |
| `negative_prompt` | ✓ | ✓ | ✓ |
| `num_frames` · `frame_rate` | ✓ | — | — |
| `num_images_per_prompt` | — | ✓ | ✓ |
| `image_reference` | ✓ | — | required |
| `video_reference` · `audio_reference` | ✓ | — | — |

Each reference conditions one generation, so it goes in a request and `common_params`
rejects it.

#### Resolution order

Each request is `common_params`, then the request's own keys. `extra_params` merges per key
rather than being replaced whole.

* `width` and `height` are judged on key presence: setting exactly one is rejected before the
  run.
* `--num-requests` cycles or truncates the resulting list.

#### References and prompt files

A path is read and encoded when the document loads, so a missing file fails before the run
starts. Relative paths resolve from the document, and `~` expands; there is no variable
expansion. A reference may also be given in the wire form `MediaRef` declares — `{content,
format}` — which is passed through untouched.

`extra_params.action_file` names a JSON `[T, D]` action trajectory, read into
`extra_params.action` and dropped; setting both is an error. `_resolve_prompt_file` documents
the shapes a prompt file is read in.

## Result

`--save-result` writes the JSON, `--save-detailed` adds the server-side series and the
per-request records, and the run prints the summary scalars with a table of `e2e_latency` and
`gen_latency`. Every series is `{mean, median, std, min, max, percentiles}` over the run's
requests — **one sample per request**.

### Series relationships

```
gen_latency = server_gen  + network + client_poll_interval
e2e_latency = gen_latency + server_media_encode + client_fetch_result
```

`e2e_latency` and `gen_latency` are client-measured and `server_gen` comes from the
`Server-Timing` header; the other terms name spans nothing measures on its own, and their
prefix says which side spends them.

Under `--response-format path` the fetch returns a path rather than the bytes, so
`e2e_latency` minus `gen_latency` is essentially `server_media_encode`. The two being equal
means the boundary was not observed: the encode finished within one `--poll-interval`.

`server_pre_denoise`, `server_denoise` and `server_post_denoise` sum to the pipeline's time on
the device, and `server_gen` minus that sum is its host-side work.

`server_post_denoise` and the `postprocessing` job status are different phases either side of
the `gen_latency` boundary: the first is GPU work inside `server_gen`, the second is the encode
that follows it, and `gen_latency` stops where the second begins. The encode's cost belongs to
`--format` and to whether ffmpeg is installed, so folding it into `server_gen` would move the
engine's number when only the output container changed.

### Validity checks

The run exits non-zero when `completed` differs from `total_requests`, after writing the
result.

A completed request says nothing about the media it produced. Confirm the artifacts decode and
match the requested shape.
