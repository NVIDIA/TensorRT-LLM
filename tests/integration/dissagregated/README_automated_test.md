# Automated Disaggregated Test

This tool automates testing of TensorRT-LLM's disaggregated serving capability with multiple context and generation servers.

## Overview

The `automated_disagg_test.py` script replaces the manual bash testing script with a more robust Python implementation that:

1. Starts CONTEXT and GENERATION servers on specified GPUs
2. Dynamically waits for servers to become healthy (instead of using fixed sleep times)
3. Properly tracks and manages all processes
4. Runs client tests and verifies results
5. Tests failover scenarios by killing servers
6. Ensures proper cleanup of all processes

## Requirements

Install the required Python packages:

```bash
pip install requests psutil
```

## Usage

Basic usage:

```bash
python automated_disagg_test.py
```

The script uses default values, but all parameters can be customized:

```bash
python automated_disagg_test.py \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --extra-llm-api-path TensorRT-LLM/examples/disaggregated/extra-llm-api-config.yml \
  --etcd-config-path TensorRT-LLM/examples/disaggregated/etcd_config.yaml \
  --disagg-config-path disagg_config.yaml \
  --client-script-path TensorRT-LLM/examples/disaggregated/clients/disagg_client.py \
  --prompts-path TensorRT-LLM/examples/disaggregated/clients/prompts.json
```

## How It Works

The test sequence:

1. Starts a CONTEXT server on GPU 0 (port 8001)
2. Starts a GENERATION server on GPU 1 (port 8002)
3. Waits for these servers to be healthy (checking /health endpoints)
4. Launches a disaggregated service
5. Starts a second CONTEXT server on GPU 2 (port 8003)
6. Waits for the second context server to be healthy
7. Runs the client test with both context servers available
8. Kills the first CONTEXT server (on port 8001)
9. Runs the client test again to verify failover to the second context server
10. Cleans up all processes

## Exit Codes

- `0`: Test successful
- `1`: Test failed

## Logs

The script provides detailed logging to help diagnose any issues:

- Server startup and health status
- Client test results
- Error conditions with stack traces
- Process cleanup information

## Customization

You can extend the `DisaggregatedTester` class to add more test scenarios or modify the existing ones to suit your specific testing needs.
