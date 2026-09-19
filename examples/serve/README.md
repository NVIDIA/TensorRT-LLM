# Online Serving Examples with `trtllm-serve`

We provide a CLI command, `trtllm-serve`, to launch a FastAPI server compatible with OpenAI APIs, here are some client examples to query the server, you can check the source code here or refer to the [command documentation](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html) and [examples](https://nvidia.github.io/TensorRT-LLM/examples/trtllm_serve_examples.html) for detailed information and usage guidelines.


## In-flight batch size

The PyTorch executor publishes `trtllm_inflight_batch_size` directly to the
`/prometheus/metrics` endpoint. Enable Prometheus using an extra LLM API options
file; iteration statistics and iteration logging can remain disabled:

```yaml
return_perf_metrics: true
enable_iter_perf_stats: false
print_iter_log: false
```

```bash
trtllm-serve <model> --extra_llm_api_options metrics.yaml
curl http://localhost:8000/prometheus/metrics
```

The gauge counts real context (prefill) plus generation (decode) requests in the
latest batch submitted to forward, after resource preparation. It excludes
attention-DP dummy requests, queued requests, encoder-only work, and requests
waiting for KV transfer. It becomes zero when no batch can run, before blocking
at idle, and when the executor exits. For pipeline parallelism it describes the
latest microbatch, not the sum of outstanding microbatches.

Labels are `model_name` (the configured model path or ID), `engine_type`, `rank`,
and the Prometheus multiprocess `pid`. Each executor process has its own series,
so disaggregated prefill and decode workers remain distinct. Tensor/pipeline
parallel ranks can report the same logical requests: do not sum replicas to
estimate worker batch size. Attention-DP ranks report their local batches.

The endpoint collects executor processes that share its
`PROMETHEUS_MULTIPROC_DIR`. With a node-local directory, it exposes only ranks on
that node; it does not gather remote ranks. An externally launched executor must
inherit the directory before importing `prometheus_client`. Use a fresh metrics
directory for each deployment, as required by Prometheus multiprocess mode.

This is a sampled gauge, so a scrape can miss short prefill batches. Enable
`enable_iter_perf_stats: true` when a complete per-iteration history is needed.
Its `inflightBatchingStats.numScheduledRequests` uses the same pre-forward
boundary and dummy filtering. The existing `trtllm_num_scheduled_requests` gauge
still comes from that iteration-statistics history and can lag the direct gauge
while the server drains the history. `return_perf_metrics` still enables its
usual request timing metrics; the new gauge itself needs no CUDA timing or
iteration-statistics collection.
