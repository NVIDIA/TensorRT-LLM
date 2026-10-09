# Online Serving Examples with `trtllm-serve`

We provide a CLI command, `trtllm-serve`, to launch a FastAPI server compatible with OpenAI APIs, here are some client examples to query the server, you can check the source code here or refer to the [command documentation](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html) and [examples](https://nvidia.github.io/TensorRT-LLM/examples/trtllm_serve_examples.html) for detailed information and usage guidelines.


## Scheduled batch size

The PyTorch executor publishes `trtllm_scheduled_batch_size` directly to the
`/prometheus/metrics` endpoint. Enable Prometheus using a configuration
file; iteration statistics and iteration logging can remain disabled:

```yaml
return_perf_metrics: true
enable_iter_perf_stats: false
print_iter_log: false
```

```bash
trtllm-serve <model> --config metrics.yaml
curl http://localhost:8000/prometheus/metrics
```

The gauge counts real context (prefill) plus generation (decode) requests in the
latest batch submitted to forward, after resource preparation. It excludes
attention-DP dummy requests, queued requests, encoder-only work, and requests
waiting for KV transfer. It becomes zero when no batch can be submitted, before blocking
at idle, and on normal executor exit or exceptions that reach cleanup. For
pipeline parallelism it describes the latest scheduling decision for one
microbatch, not the sum of outstanding microbatches. A zero can therefore be
scraped while other microbatches are still running; this metric is not a GPU
utilization or total outstanding-request gauge.

Labels are `model_name` (the served-model alias, local model directory basename,
or Hugging Face ID, matching the server's other metrics), `engine_type`, `rank`,
and the Prometheus multiprocess `pid`. Each executor process has its own series,
so disaggregated prefill and decode workers remain distinct. Tensor/pipeline
parallel ranks can report the same logical requests: do not sum replicas to
estimate worker batch size. Attention-DP ranks report their local batches.

The endpoint collects executor processes that share its
`PROMETHEUS_MULTIPROC_DIR`. With a node-local directory, it exposes only ranks on
that node; it does not gather remote ranks. An externally launched executor must
inherit the directory before importing `prometheus_client`. Use a fresh metrics
directory for each deployment, as required by Prometheus multiprocess mode.
The gauge uses `all` mode to retain one series per process. A worker killed
before cleanup can leave its last value in that directory; restarting only the
worker does not remove the old PID series. Do not treat retained series as a
worker-liveness signal.

For offline `LLM(return_perf_metrics=True)`, construction also initializes this
process-wide environment variable and a temporary directory if one was not
supplied. A supplied directory must already exist. An already-started external
MPI session must receive the variable before its launch; setting it in the
parent afterward cannot update those workers.

This is a sampled gauge, so a scrape can miss short prefill batches. Enable
`enable_iter_perf_stats: true` when a complete per-iteration history is needed.
Its `inflightBatchingStats.numScheduledRequests` uses the same pre-forward
boundary and dummy filtering. The existing `trtllm_num_scheduled_requests` gauge
still comes from that iteration-statistics history and can lag the direct gauge
while the server drains the history. `return_perf_metrics` still enables its
usual request timing metrics; the new gauge itself needs no CUDA timing or
iteration-statistics collection.
