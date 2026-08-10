import requests
from defs.trt_test_alternative import print_error, print_info


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
