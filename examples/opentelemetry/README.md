# OpenTelemetry Integration Guide

This guide explains how to setup OpenTelemetry tracing in TensorRT-LLM to monitor and debug your LLM inference services.

## Install OpenTelemetry

Install the required OpenTelemetry packages:

```bash
pip install \
  'opentelemetry-sdk' \
  'opentelemetry-api' \
  'opentelemetry-exporter-otlp' \
  'opentelemetry-semantic-conventions-ai'
```

## Start Jaeger

You can start Jaeger with Docker:

```bash
docker run --rm --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 14250:14250 \
  -p 14268:14268 \
  -p 14269:14269 \
  -p 9411:9411 \
  jaegertracing/all-in-one:1.57.0
```

Or run the jaeger-all-in-one(.exe) executable from [the binary distribution archives](https://www.jaegertracing.io/download/):

```bash
jaeger-all-in-one --collector.zipkin.host-port=:9411
```

## Setup environment variables and run TensorRT-LLM

Set up the environment variables:

```bash
export JAEGER_IP=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' jaeger)
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
export OTEL_SERVICE_NAME="trt-server"
```

Then run TensorRT-LLM with OpenTelemetry, and make sure to set `return_perf_metrics` to true in the model configuration:

```bash
trtllm-serve models/Qwen3-8B/ --otlp_traces_endpoint="$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
```

## Send requests and find traces in Jaeger

You can send a request to the server and view the traces in [Jaeger UI](http://localhost:16686/).
The traces should be visible under the service name "trt-server".

## Configuration for Disaggregated Serving

For disaggregated serving scenarios, the configuration for ctx server and gen server remains the same as the standalone model. For the proxy, you can configure it as follows:

```yaml
# disagg_config.yaml
hostname: 127.0.0.1
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
    - "127.0.0.1:8001"
generation_servers:
  num_instances: 1
  urls:
    - "127.0.0.1:8002"
observability_config:
  otlp_traces_endpoint: "grpc://0.0.0.0:4317"
```
