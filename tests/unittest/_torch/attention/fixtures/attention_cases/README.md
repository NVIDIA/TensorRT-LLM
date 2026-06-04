# Captured attention cases (replay fixtures)

Each `*.json` / `*.jsonl` here is a `BackendCase` spec (shapes + config, no tensor
values) captured from a real model run and minimized to a minimal reproducer.

To add a failing real-workload case:

```bash
# 1. Capture all attention forwards from a run:
TRTLLM_ATTN_CAPTURE_DIR=/tmp/attn_cap <your model run / pytest>
# 2. Minimize the failing (usually last) case and commit it here:
python -m minimize /tmp/attn_cap/cases_rank0.jsonl --backend TRTLLM \
    > tests/unittest/_torch/attention/fixtures/attention_cases/<name>.json
```

`test_attention_replay.py` auto-discovers and replays every file here.
