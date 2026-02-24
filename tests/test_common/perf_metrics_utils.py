import requests


def get_timing_metrics(server_url: str):
    response = requests.get(f"{server_url}/perf_metrics", timeout=10)
    assert response.status_code == 200
    perf_metrics = response.json()
    assert len(perf_metrics) > 0
    return perf_metrics[0]


def validate_timing_metrics(perf_metrics_item, request_context="", time_tolerance_seconds=0.005):
    """Helper function to validate timing metrics relationships.

    Args:
        perf_metrics_item: A single performance metrics item from the /perf_metrics endpoint
        request_context: String context for error messages (e.g., "request 1", "streaming")
    """
    # Validate basic structure
    required_keys = [
        "ctx_server",
        "gen_server",
        "ctx_perf_metrics",
        "gen_perf_metrics",
        "disagg_server_arrival_time",
        "disagg_server_first_token_time",
    ]
    for key in required_keys:
        assert key in perf_metrics_item, f"Missing key: {key} in {request_context}"

    assert (
        perf_metrics_item["ctx_perf_metrics"]["ctx_request_id"]
        == perf_metrics_item["gen_perf_metrics"]["ctx_request_id"]
    )

    # Extract timing metrics
    ctx_metrics = perf_metrics_item["ctx_perf_metrics"]["perf_metrics"]["timing_metrics"]
    gen_metrics = perf_metrics_item["gen_perf_metrics"]["perf_metrics"]["timing_metrics"]
    disagg_arrival = perf_metrics_item["disagg_server_arrival_time"]
    disagg_first_token = perf_metrics_item["disagg_server_first_token_time"]

    # Validate disaggregated server timing metrics
    assert disagg_arrival is not None, f"disagg_server_arrival_time is None in {request_context}"
    assert disagg_first_token is not None, (
        f"disagg_server_first_token_time is None in {request_context}"
    )
    assert isinstance(disagg_arrival, (int, float)), (
        f"disagg_server_arrival_time is not numeric in {request_context}"
    )
    assert isinstance(disagg_first_token, (int, float)), (
        f"disagg_server_first_token_time is not numeric in {request_context}"
    )
    assert disagg_arrival > 0, f"disagg_server_arrival_time is not positive in {request_context}"
    assert disagg_first_token > 0, (
        f"disagg_server_first_token_time is not positive in {request_context}"
    )
    assert disagg_arrival <= disagg_first_token, (
        f"disagg_server_arrival_time > disagg_server_first_token_time in {request_context}"
    )

    # Validate server-level timing metrics for context server
    ctx_server_arrival = ctx_metrics.get("server_arrival_time")
    ctx_server_first_token = ctx_metrics.get("server_first_token_time")
    assert ctx_server_arrival is not None, f"ctx server_arrival_time is None in {request_context}"
    assert ctx_server_first_token is not None, (
        f"ctx server_first_token_time is None in {request_context}"
    )
    assert isinstance(ctx_server_arrival, (int, float)), (
        f"ctx server_arrival_time is not numeric in {request_context}"
    )
    assert isinstance(ctx_server_first_token, (int, float)), (
        f"ctx server_first_token_time is not numeric in {request_context}"
    )
    assert ctx_server_arrival <= ctx_server_first_token, (
        f"ctx server_arrival_time > server_first_token_time in {request_context}"
    )
    assert ctx_metrics["last_token_time"] - ctx_server_first_token < 1e-3

    # Validate server-level timing metrics for generation server
    gen_server_arrival = gen_metrics.get("server_arrival_time")
    gen_server_first_token = gen_metrics.get("server_first_token_time")
    assert gen_server_arrival is not None, f"gen server_arrival_time is None in {request_context}"
    assert gen_server_first_token is not None, (
        f"gen server_first_token_time is None in {request_context}"
    )
    assert isinstance(gen_server_arrival, (int, float)), (
        f"gen server_arrival_time is not numeric in {request_context}"
    )
    assert isinstance(gen_server_first_token, (int, float)), (
        f"gen server_first_token_time is not numeric in {request_context}"
    )
    assert gen_server_arrival <= gen_server_first_token, (
        f"gen server_arrival_time > server_first_token_time in {request_context}"
    )

    # Validate timing relationships between different levels
    # Disaggregated server should receive request before individual servers
    # Allow some tolerance of a local network ping time when comparing the times from disagg and ctx/gen servers
    # by taking consideration of the error of NTP (1/2 ping time).
    assert disagg_arrival <= ctx_server_arrival + time_tolerance_seconds, (
        f"disagg_arrival {disagg_arrival} > ctx_server_arrival {ctx_server_arrival} in {request_context}"
    )
    assert disagg_arrival <= gen_server_arrival + time_tolerance_seconds, (
        f"disagg_arrival {disagg_arrival} > gen_server_arrival {gen_server_arrival} in {request_context}"
    )

    # Context should complete before generation starts
    assert ctx_server_first_token <= gen_server_arrival + time_tolerance_seconds, (
        f"ctx_server_first_token > gen_server_arrival in {request_context}"
    )

    # Validate internal timing consistency
    ctx_arrival_time = ctx_metrics["arrival_time"]
    ctx_first_token_time = ctx_metrics["first_token_time"]
    gen_arrival_time = gen_metrics["arrival_time"]
    gen_first_token_time = gen_metrics["first_token_time"]

    assert ctx_arrival_time <= ctx_first_token_time, (
        f"ctx arrival_time > first_token_time in {request_context}"
    )
    assert gen_arrival_time <= gen_first_token_time, (
        f"gen arrival_time > first_token_time in {request_context}"
    )

    # Test KV cache transfer timing (if present)
    if "kv_cache_transfer_start" in gen_metrics and "kv_cache_transfer_end" in gen_metrics:
        kv_start = gen_metrics["kv_cache_transfer_start"]
        kv_end = gen_metrics["kv_cache_transfer_end"]
        assert gen_metrics["kv_cache_size"] > 0
        assert kv_start <= kv_end, (
            f"kv_cache_transfer_start > kv_cache_transfer_end in {request_context}"
        )
        assert gen_arrival_time <= kv_start, (
            f"gen_arrival_time > kv_cache_transfer_start in {request_context}"
        )
        assert kv_end <= gen_metrics["first_scheduled_time"], (
            f"kv_cache_transfer_end > first_scheduled_time in {request_context}"
        )

    return True


def get_prometheus_metrics(server_url: str):
    response = requests.get(server_url + "/prometheus/metrics")
    assert response.status_code == 200
    # Parse Prometheus metrics lines into a dictionary of {metric_name: value}
    metrics = {}
    print(response.text)
    for line in response.text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric = parts[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        import re

        if bucket_match := re.match(r'(.+)_bucket\{le="([^"]+)"\}', metric):
            # Try to parse bucket boundaries out of metrics like ..._bucket{le="0.005"}
            base_metric, le_value = bucket_match.groups()
            if base_metric not in metrics:
                metrics[base_metric] = {}
            try:
                metrics[base_metric][float(le_value)] = value
            except ValueError:
                continue
        elif sum_match := re.match(r"(.+)_sum$", metric):
            base_metric = sum_match.groups()[0]
            if base_metric not in metrics:
                metrics[base_metric] = {}
            metrics[base_metric]["sum"] = value
        elif count_match := re.match(r"(.+)_count$", metric):
            base_metric = count_match.groups()[0]
            if base_metric not in metrics:
                metrics[base_metric] = {}
            metrics[base_metric]["count"] = value
        elif total_match := re.match(r"(.+)_total$", metric):
            base_metric = total_match.groups()[0]
            print(f"Total metric {metric}: {base_metric} = {value}")
            metrics[base_metric] = value
        else:
            # ignore prometheus built-in metrics
            pass
    return metrics
