import os
import time

import pytest
import requests
from defs.conftest import llm_models_root
from defs.trt_test_alternative import popen, print_error, print_info


def check_spec_decoding_metrics(http_port="8000", expect_draft_latency=False, min_draft_tokens=1):
    print_info("Checking specDecodingStats in /metrics endpoint...")

    try:
        url = f"http://localhost:{http_port}/metrics"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            print_error(f"Failed to get metrics: HTTP {response.status_code}")
            return False

        metrics_list = response.json()
        if not metrics_list or len(metrics_list) == 0:
            print_error("No metrics returned")
            return False

        print_info(f"Retrieved {len(metrics_list)} iteration stats")

        # Find iterations with speculative decoding
        iterations_with_spec = []
        for metric in metrics_list:
            spec_stats = metric.get("specDecodingStats")
            if spec_stats and spec_stats.get("numDraftTokens", 0) > 0:
                iterations_with_spec.append((metric["iter"], spec_stats))

        if not iterations_with_spec:
            print_error("No iterations with specDecodingStats found")
            print_info("This might happen if all requests are in context phase")
            return False

        print_info(f"Found {len(iterations_with_spec)} iterations with spec decoding")

        # Validate specDecodingStats structure and values
        for iter_num, spec_stats in iterations_with_spec[:3]:  # Check first 3
            # Validate field types and constraints
            assert isinstance(spec_stats["numDraftTokens"], int)
            assert isinstance(spec_stats["numAcceptedTokens"], int)
            assert isinstance(spec_stats["numRequestsWithDraftTokens"], int)
            assert isinstance(spec_stats["acceptanceLength"], (int, float))
            assert isinstance(spec_stats["iterLatencyMS"], (int, float))
            assert isinstance(spec_stats["draftOverhead"], (int, float))

            # Validate value constraints
            assert spec_stats["numDraftTokens"] >= min_draft_tokens
            assert spec_stats["numAcceptedTokens"] >= 0
            assert spec_stats["numAcceptedTokens"] <= spec_stats["numDraftTokens"]
            assert spec_stats["numRequestsWithDraftTokens"] > 0
            assert spec_stats["acceptanceLength"] >= 1.0  # At least 1 token per request
            assert spec_stats["iterLatencyMS"] >= 0.0
            assert spec_stats["draftOverhead"] >= 0.0
            assert spec_stats["draftOverhead"] <= 1.0  # Can't exceed 100%

            # For 2-model mode, verify draft latency is tracked
            if expect_draft_latency:
                assert spec_stats["iterLatencyMS"] > 0.0, (
                    "Two-model mode should have non-zero draft latency"
                )
                assert spec_stats["draftOverhead"] > 0.0, (
                    "Two-model mode should have non-zero draft overhead"
                )

        return True

    except Exception as e:
        print_error(f"Error checking spec decoding metrics: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_spec_decoding_metrics_eagle3_one_model():
    model_path = f"{llm_models_root()}/Qwen3/Qwen3-8B"
    eagle3_path = f"{llm_models_root()}/Qwen3/qwen3_8b_eagle3"

    # Skip if models don't exist
    if not os.path.exists(model_path) or not os.path.exists(eagle3_path):
        pytest.skip(f"Required models not found: {model_path}, {eagle3_path}")

    extra_config = {
        "enable_iter_perf_stats": True,
        "disable_overlap_scheduler": True,
        "trust_remote_code": True,
        "speculative_config": {
            "decoding_type": "Eagle",
            "max_draft_len": 4,
            "speculative_model": eagle3_path,
            "eagle3_one_model": True,
        },
    }

    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as f:
        yaml.dump(extra_config, f)
        config_file = f.name

    port = 8010
    cmd = [
        "trtllm-serve",
        model_path,
        "--backend",
        "pytorch",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--max_batch_size",
        "8",
        "--trust_remote_code",
        "--config",
        config_file,
    ]

    try:
        print_info("Starting trtllm-serve with Eagle3 one-model mode...")
        with popen(cmd):
            # Wait for server
            from test_serve import check_server_ready

            check_server_ready(http_port=str(port), timeout_timer=600)

            # Send test requests
            print_info("Sending test requests...")
            for i in range(5):
                requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={
                        "model": "Qwen3-8B",
                        "messages": [{"role": "user", "content": f"Count to {i + 3}"}],
                        "max_tokens": 30,
                    },
                    timeout=30,
                )
                time.sleep(0.3)

            # Wait for metrics to accumulate
            time.sleep(1)

            # Check metrics (one-model mode has iterLatencyMS=0)
            assert check_spec_decoding_metrics(
                http_port=str(port),
                expect_draft_latency=False,  # One-model mode
                min_draft_tokens=1,
            )
    finally:
        if os.path.exists(config_file):
            os.remove(config_file)


def test_spec_decoding_metrics_eagle3_two_model():
    model_path = f"{llm_models_root()}/Qwen3/Qwen3-8B"
    eagle3_path = f"{llm_models_root()}/Qwen3/qwen3_8b_eagle3"

    # Skip if models don't exist
    if not os.path.exists(model_path) or not os.path.exists(eagle3_path):
        pytest.skip(f"Required models not found: {model_path}, {eagle3_path}")

    extra_config = {
        "enable_iter_perf_stats": True,
        "disable_overlap_scheduler": True,
        "trust_remote_code": True,
        "speculative_config": {
            "decoding_type": "Eagle",
            "max_draft_len": 4,
            "speculative_model": eagle3_path,
            "eagle3_one_model": False,  # Two-model mode
        },
    }

    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as f:
        yaml.dump(extra_config, f)
        config_file = f.name

    port = 8011
    cmd = [
        "trtllm-serve",
        model_path,
        "--backend",
        "pytorch",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--max_batch_size",
        "8",
        "--trust_remote_code",
        "--config",
        config_file,
    ]

    try:
        print_info("Starting trtllm-serve with Eagle3 two-model mode...")
        with popen(cmd):
            # Wait for server
            from test_serve import check_server_ready

            check_server_ready(http_port=str(port), timeout_timer=600)

            # Send test requests
            print_info("Sending test requests...")
            for i in range(5):
                requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={
                        "model": "Qwen3-8B",
                        "messages": [{"role": "user", "content": f"Count to {i + 3}"}],
                        "max_tokens": 30,
                    },
                    timeout=30,
                )
                time.sleep(0.3)

            # Wait for metrics to accumulate
            time.sleep(1)

            # Check metrics (two-model mode should have iterLatencyMS > 0)
            assert check_spec_decoding_metrics(
                http_port=str(port),
                expect_draft_latency=True,  # Two-model mode
                min_draft_tokens=1,
            )
    finally:
        if os.path.exists(config_file):
            os.remove(config_file)
