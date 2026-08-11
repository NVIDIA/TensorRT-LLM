# GVR top-k per-step perf grid (provenance for the PR numbers)

Per-step paired protocol: each (arm, model, ISL, B, layer) is one NVTX
range; the cold reps inside cycle every usable decode step (L2 evicted,
batch refilled before eviction), so the k-th kernel instance of a range
IS decode step k and arms pair index-by-index.

- `ab_steps.py` - the driver (env: ARM = pr|st|va|vb|wf, UNITS =
  "pro:64k,...", BS_LIST, OUT; GVR_CAP_ROOT points at the capture data,
  HARNESS_ROOT at the repo).
- `f58_grid.sh <gpu> "<units>"` - nsys wrapper per arm x unit
  (GVR_BENCH_OUT selects the output dir).
- `perstep.py` - per-step extraction from the nsys sqlite.
- `f58_an.py` - pairing + production routing (plan_emission) + the B x N
  mean/min tables. `f58_regr.py`-style regression stats derive from the
  same pickle.

Requirements: B200 (SM100), cutlass-dsl 4.5.x on PYTHONPATH, exclusive
GPU (no concurrent process - it pollutes event timing), nsys >= 2025.3.
