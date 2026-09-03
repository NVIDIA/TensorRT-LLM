# Benchmarking a VisualGen server

`tensorrt_llm.serve.scripts.benchmark_visual_gen` drives a running `trtllm-serve` VisualGen
server over its OpenAI-compatible routes and reports latency and throughput.

```bash
python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
    --workload workload.yaml --port 8000 \
    --max-concurrency 1 --save-result --save-detailed
```

`--help` lists every flag.

## Spelling the document on the CLI

The same document, without a file: each `common_params` field below is a flag of the same
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

## The `--workload` document

A YAML or JSON file, the same document inline (starting with `{` or `[`), or a bare list of
requests.

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
  - prompt_file: prompts/aerial.json      # instead of prompt, not as well as
    image_reference: ../media/frame.png
    width: 720                            # overrides common_params for this request
    height: 1280
```

### Fields

| key | where | meaning |
|---|---|---|
| `backend` | top level | Selects the route, and so what the run measures. Required here or as `--backend`; disagreeing with it is an error. |
| `prompt` | request or `common_params` | The prompt text. |
| `prompt_file` | request or `common_params` | Path to a prompt file. Mutually exclusive with `prompt`. |
| `image_reference` · `video_reference` | request only | Reference media; it conditions one generation, so `common_params` rejects it. `video_reference` is video-only. |
| `extra_params` | request or `common_params` | Per-pipeline parameters. Shallow-merged, so a request overriding one key keeps the others. |
| everything else | request or `common_params` | `VisualGenParams` fields — `width`, `height`, `num_frames`, `frame_rate`, `num_inference_steps`, `guidance_scale`, `seed`, `max_sequence_length`, `negative_prompt`. |

### Resolution order

Each request is `common_params`, then the request's own keys.
`extra_params` merges per key rather than being replaced whole.

* `width` and `height` are judged on key presence: setting exactly one is rejected before the
  run rather than by a 422 from the server.
* `--num-requests` cycles or truncates the resulting list.

### References and prompt files

A path is read and encoded when the document loads, so a missing file fails before the run
rather than part-way through it. Relative paths resolve from the document, and `~` expands;
there is no variable expansion.

A reference may also be given in the wire form `MediaReferenceItem` declares — `{content,
format}` with `format` one of `path`, `url`, `base64` — which is passed through untouched.

A prompt file is read in three shapes:

| file contents | prompt sent |
|---|---|
| JSON object with a `prompt` key | that field |
| JSON object without one | the whole object, serialized |
| anything that is not JSON | the text |

## Metrics

Printed after the run, and written to the result JSON by `--save-result`. Each is reported as
`{mean, median, std, min, max, percentiles}` over the requests.

### Client-side

#### `e2e_latency`
  * From sending the request until the result has been fully read.

#### `gen_latency`
  * From sending the request until the job first reports `postprocessing` or `completed`.
  * Video only — image routes are synchronous, so the measurement does not exist and the row
    is absent.

#### Throughput
  * `request_throughput` is completed requests over the benchmark duration.
  * `frames_per_second` for video, `images_per_second` for image routes.

### Server-side

Read from the `Server-Timing` response header, and present only with `--save-detailed`.

#### `timings.server_e2e`
  * Request arrival to job completion, measured by the server.

#### `timings.server_gen`
  * Engine wall clock. Excludes network and poll granularity, which makes it the series to
    watch for regressions.

#### `timings.server_denoise`
  * The denoise loop alone.

### Reading the differences

$$
\text{e2e} - \text{gen} = \text{encode} + \text{fetching the result}
$$

$$
\text{gen} - \text{server\_gen} = \text{network} + \text{one poll interval}
$$

Under `--response-format path` the fetch returns a path rather than the bytes, so the first
gap is essentially the encode.

`gen_latency` equal to `e2e_latency` means the boundary was not observed, not that the encode
was free. The split is visible only while the server can answer during the encode, which is
not the case when `TRTLLM_VIDEO_ASYNC_ENCODE=0` puts the encode on the event loop, or when
the encode finishes within one `--poll-interval`.

## Result JSON

`--save-result` writes the printed metrics plus `date`, `duration`, and `config` — the run's
`num_requests`, `max_concurrency`, `response_format`, `format` and, for video,
`poll_interval`.

`--save-detailed` adds `timings.server_*` and a `requests[]` record per request:

| key | contents |
|---|---|
| `index` · `prompt` | Position and prompt text. |
| `prompt_file` · `image_reference` · `video_reference` | Present when the document set them, by locator — a path, a URL, or `<base64>` — never the encoded bytes. |
| `params` | The merged parameters as sent, which is what a run actually measured. |
| `success` · `error` | Outcome. A run with any failure is not a result. |
| `start` · `end` | Wall-clock bounds. |
| `client_e2e` · `client_gen` · `server_e2e` · `server_gen` · `server_denoise` | The five timings; `null` where undefined for the backend or not reported. |
| `poll_count` | Status polls, video only. |
| `output_paths` | Always a list; an image request with `n > 1` has several. |

A completed request says nothing about the media it produced. Confirm the artifacts decode
and match the requested shape.
