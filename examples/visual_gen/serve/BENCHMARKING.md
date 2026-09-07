# Benchmarking a VisualGen server

`tensorrt_llm.serve.scripts.benchmark_visual_gen` drives a running `trtllm-serve` VisualGen
server over its OpenAI-compatible routes and reports latency and throughput.

```bash
python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
    --workload workload.yaml --port 8000 \
    --max-concurrency 1 --save-result --save-detailed
```

A run has two kinds of input: **the workload**, which is what to generate, and **the run
settings**, which are how to send it and where to put the results. This page is split the same
way: [Input](#input), then [Output](#output).

`--help` lists every flag.

## Input

### The workload

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

#### Fields

A route accepts in the document what it accepts on the wire. The client derives each backend's
schema from the request model that route validates against — `ImageGenerationRequest`,
`ImageEditRequest` and `VideoGenerationRequest` in
[`tensorrt_llm/serve/openai_protocol.py`](../../../tensorrt_llm/serve/openai_protocol.py) — so
those models are the authority when this table falls behind.

`backend` sits at the top level and selects the route, and so what the run measures. It is
required there or as `--backend`, and disagreeing with it is an error.

The other keys this document defines go in a request or in `common_params`:

| key | meaning |
|---|---|
| `prompt` | The prompt text. |
| `prompt_file` | Path to a prompt file. Mutually exclusive with `prompt`. |
| `extra_params` | Per-pipeline parameters. Shallow-merged, so a request overriding one key keeps the others. `action_file` is read here and sent as `action`. |

Everything else is a `VisualGenParams` field, and which ones a route takes is the route's own
schema. Naming one it does not take is an error.

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

Each request is `common_params`, then the request's own keys.
`extra_params` merges per key rather than being replaced whole.

* `width` and `height` are judged on key presence: setting exactly one is rejected before the
  run.
* `--num-requests` cycles or truncates the resulting list.

#### References and prompt files

A path is read and encoded when the document loads, so a missing file fails before the run
starts. Relative paths resolve from the document, and `~` expands;
there is no variable expansion.

A reference may also be given in the wire form `MediaReferenceItem` declares — `{content,
format}` with `format` one of `path`, `url`, `base64` — which is passed through untouched.

`extra_params.action_file` names a JSON `[T, D]` action trajectory, read into `extra_params.action`
and dropped. The server takes only the keys its pipeline declares, and a trajectory is hundreds of
literals; setting both it and `action` is an error.

A prompt file is read in three shapes:

| file contents | prompt sent |
|---|---|
| JSON object with a `prompt` key | that field |
| JSON object without one | the whole object, serialized |
| anything that is not JSON | the text |

#### Spelling the document on the CLI

The same document, without a file: each `common_params` field above is a flag of the same
name, and `--requests` carries the list. References have no flag, being request-only.

```bash
python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
    --backend openai-videos --width 1280 --height 720 --num-frames 81 \
    --requests '[{"prompt": "A red fox"}, {"prompt": "A cat", "seed": 7}]' --port 8000
```

Entries override the fields per key, exactly as they override `common_params` in a file,
which also has to state its `requests` list.

`--workload` and this spelling are alternatives; the scalar flags are generated from
`VisualGenParams`, so a flag cannot name a field differently from the document.

### The run settings

Everything the document does not decide. These are run-level by construction: varying one
within a run would make its own aggregate incomparable.

#### Connection

| flag | meaning |
|---|---|
| `--host` · `--port` | Where the server is. Default `127.0.0.1:8000`. |
| `--num-gpus` | GPUs the server runs on — the product of its parallel sizes. Recorded in the result, and `request_throughput` is divided by it into `per_gpu_throughput`. The same model and workload at 1, 4 or 8 GPUs are different measurements, and the server reports no topology of its own. |
| `--backend` | Supplies `backend` when the document omits it. Required from one of the two — it selects the route, and a checkpoint serving both modes answers the wrong one without complaining. |
| `--model` | Sent as the request's `model` field and used to label results. Default: the id from `GET /v1/models`, so a stored result names the checkpoint that produced it. Passing it cross-checks against that id rather than replacing it. |

#### Traffic

| flag | meaning |
|---|---|
| `--num-requests` | Resize the workload to exactly this many requests, cycling in order or truncating. Default: send the document as written. |
| `--max-concurrency` | Maximum requests in flight. Default: unbounded. |
| `--request-rate` | Arrival rate in req/s, which paces when a request is created; `--max-concurrency` caps how many run. Default `inf`, which creates them all at once. |
| `--burstiness` | Spread of the arrival intervals, in effect while `--request-rate` is finite. Default `1.0`, an exponential interval. Below 1 the arrivals come in bursts; above 1 they even out. |
| `--request-timeout` | Per-request timeout in seconds. Default 6 hours. |
| `--response-format` | How the server returns media, which is inside the measured window: `path` returns a locator, the others return the bytes. Default `path`; the routes otherwise accept `file` (video, its own default) and `url` / `b64_json` (images, default `url`). |
| `--format` | Encoding the server writes: `mp4`/`avi`/`auto` for video, `png`/`webp`/`jpeg` for images. Default: the server's own, which for video is `auto` — without ffmpeg that is AVI/MJPEG, a different encode inside the measured window. |
| `--poll-interval` | Status poll interval for `openai-videos`, default `0.1`. It is the granularity of `gen_latency` and `e2e_latency`; the image routes are synchronous and ignore it. |
| `--no-test-input` | Skip the single probe request sent before the measured run. It is not counted, and it fails fast on a workload the server rejects. |
| `--disable-tqdm` | No progress bar. |

#### Where results go

| flag | meaning |
|---|---|
| `--save-result` | Write the result JSON. Without it the run only prints. |
| `--save-detailed` | Add `timings.server_*` and the per-request records. A heterogeneous run cannot be attributed without them. |
| `--result-dir` · `--result-filename` | Where to write, and under what name. |
| `--output-media-dir` | Write each successful request's media here as `{index}_{i}{ext}`, and point `output_paths` at it. Under `--response-format path` the file is copied from the server, so this client has to be able to read it; `url` costs one fetch per request. Writing stays outside the measured window. |
| `--metric-percentiles` | Comma-separated percentiles, default `50,90,99`. |
| `--metadata KEY=VALUE ...` | Free-form pairs copied into the result JSON for record keeping. |

## Output

### Metrics

Printed after the run, and written to the result JSON by `--save-result`. Each is
`{mean, median, std, min, max, percentiles}` over the run's requests — **one sample per
request**. A series nothing reported is `null`.

#### Client-side

What a caller of the endpoint experiences, network and this client's own reads included, so
these move with the deployment as much as with the engine.

| metric | measures |
|---|---|
| `e2e_latency` | From sending the request until the result has been fully read from the server. |
| `gen_latency` | From sending the request until the job first reports `postprocessing` or `completed`. Video only — the image routes are synchronous, so the measurement does not exist and the key is absent. |
| `request_throughput` | Completed requests over the benchmark duration. |
| `per_gpu_throughput` | `request_throughput` per GPU, present when `--num-gpus` says how many. |
| `frames_per_second` · `images_per_second` | Produced output over the duration; the first for video, the second for the image routes. |

#### Server-side

Read from the `Server-Timing` response header, and present only with `--save-detailed`.

| metric | measures |
|---|---|
| `timings.server_e2e` | Request arrival to the finished artifact, measured by the server: the encoded MP4 saved on the video route, the encoded image in the response body on the image routes. |
| `timings.server_gen` | The engine's inference call, what `VisualGen.generate()` costs, before any encoding or persistence. Excludes network and poll granularity, which makes it the series to watch for regressions. |
| `timings.server_pre_denoise` | Text encoding, latent prep and conditioning, on the GPU stream. |
| `timings.server_denoise` | One request's whole denoise loop. `p99` is across requests. |
| `timings.server_post_denoise` | VAE decode, format conversion and audio decode, on the GPU stream. |

The three GPU-stream phases sum to the pipeline's time on the device, and
`server_gen` minus that sum is its host-side work. A pipeline that does not time a
phase reports it as `null`.

`server_post_denoise` and the `postprocessing` job status are different phases either side of
the `gen_latency` boundary: the first is GPU work inside `server_gen`, the second is the
encode that follows it, and `gen_latency` stops where the second begins. The encode's cost
belongs to `--format` and to whether ffmpeg is installed, so folding it into `server_gen`
would move the engine's number when only the output container changed.

Not in the schema: per-step denoising latency. `server_denoise / steps` is a mean.

#### Reading the differences

```
gen_latency = server_gen  + network + client_poll_interval
e2e_latency = gen_latency + server_media_encode + client_fetch_result
```

`gen_latency`, `e2e_latency` and `server_gen` are reported; the other terms name spans
nothing measures on its own, and their prefix says which side spends them.

Under `--response-format path` the fetch returns a path rather than the bytes, so
`e2e_latency` minus `gen_latency` is essentially `server_media_encode`.

`gen_latency` equal to `e2e_latency` means the boundary was not observed: the encode
finished within one `--poll-interval`.

### Result JSON

`--save-result` writes the printed metrics plus `date`, `duration`, and `config` — the run's
`num_requests`, `num_gpus`, `max_concurrency`, `request_rate`, `burstiness`, `response_format`,
`format`, `output_media_dir` and, for video, `poll_interval`.

`--save-detailed` adds `timings.server_*` and a `requests[]` record per request:

| key | contents |
|---|---|
| `index` · `prompt` | Position and prompt text. |
| `prompt_file` · `action_file` · the `*_reference` slots | Present when the document set them, by locator — a path, a URL, or `<base64>`. |
| `params` | The merged parameters as sent, which is what a run actually measured. |
| `success` · `error` | Outcome. A run with any failure is not a result. |
| `start` · `end` | Wall-clock bounds. |
| `client_e2e` · `client_gen` · `server_e2e` · `server_gen` · `server_pre_denoise` · `server_denoise` · `server_post_denoise` | The seven timings; `null` where undefined for the backend or not reported. |
| `poll_count` | Status polls, video only. |
| `output_paths` | Always a list; an image request with `n > 1` has several. Server-side paths under `--response-format path`, and local files under `--output-media-dir`. |

A completed request says nothing about the media it produced. Confirm the artifacts decode
and match the requested shape.

The run exits non-zero when `completed` differs from `total_requests`, after writing the
result.
