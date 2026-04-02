#!/usr/bin/env python3
"""Extract coarse prefill/decode times from real trtllm-bench iteration log.

Usage:
    python3 calibrate_sim.py --iteration-log /tmp/real_iterations.jsonl \
                             --output /tmp/calibrated_sim.yaml
"""
import argparse
import json
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Extract prefill/decode times from iteration log")
    parser.add_argument("--iteration-log", required=True,
                        help="Path to iteration_log JSONL from real run")
    parser.add_argument("--output", required=True,
                        help="Output YAML path for calibrated sim config")
    args = parser.parse_args()

    prefill_times = []
    decode_times = []

    with open(args.iteration_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Iteration log may use Python repr format (single quotes, None)
            # Convert to valid JSON before parsing
            json_line = line.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
            rec = json.loads(json_line)
            latency_ms = rec.get("iterLatencyMS", rec.get("iter_latency_ms", 0))
            if latency_ms <= 0:
                continue
            # Classify: if there are context requests, it's a prefill iteration
            # Handle both camelCase (real log) and snake_case (sim log) keys
            num_ctx = rec.get("numNewActiveRequests",
                              rec.get("num_new_active_requests", 0))
            inflight = rec.get("inflightBatchingStats",
                               rec.get("inflight_batching_stats", {}))
            num_ctx_reqs = inflight.get("numContextRequests",
                                        inflight.get("num_context_requests", 0))
            if num_ctx_reqs > 0 or num_ctx > 0:
                prefill_times.append(latency_ms)
            else:
                decode_times.append(latency_ms)

    if not prefill_times:
        print("WARNING: No prefill iterations found, using default 10ms")
        mean_prefill = 10.0
    else:
        mean_prefill = sum(prefill_times) / len(prefill_times)

    if not decode_times:
        print("WARNING: No decode iterations found, using default 5ms")
        mean_decode = 5.0
    else:
        mean_decode = sum(decode_times) / len(decode_times)

    print(f"Extracted from {len(prefill_times)} prefill + "
          f"{len(decode_times)} decode iterations:")
    print(f"  Mean prefill: {mean_prefill:.2f}ms")
    print(f"  Mean decode:  {mean_decode:.2f}ms")

    config = {
        "predictor": {
            "name": "constant",
            "constant_prefill_time_ms": round(mean_prefill, 2),
            "constant_decode_time_ms": round(mean_decode, 2),
        }
    }

    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
