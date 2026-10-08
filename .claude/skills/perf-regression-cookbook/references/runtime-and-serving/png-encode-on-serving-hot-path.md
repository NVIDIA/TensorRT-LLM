---
id: case-png-encode-on-serving-hot-path
type: regression-case
family: execution-and-graph
module: runtime-and-serving
maturity: full
regression_class: [host-work-added]
signals: [host-time-increase, throughput-drop]
subsystems: [serve-endpoint]
introduced_via: [pre-existing-gap]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["6064029"]
commits: ["5653803a538e", "2ea0e6306f52"]
success_prs: [12903, 13074]
failed_prs: []
---

# Redundant + max-effort PNG encoding on the visual-gen serving hot path

> Part of the [Runtime & serving regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6064029` · commit `5653803a538e` · PR #12903 —
  Eliminate double PNG encoding in visual gen serving;
  related: nvbug `6064029` · commit `2ea0e6306f52` · PR #13074 —
  Use fast PNG compression for visual gen serving (same bug, remaining
  encode overhead after the double-encode was removed). From a customer
  report; closed as verified.
  **Filed as a functional bug, and this case is kept anyway.** The bug reports
  the visual-gen engine time and the end-to-end generation time as very
  different, which reads like two instruments disagreeing — and that *is*
  grounds for removal when the disagreement is the whole defect (contrast the
  Helix context-parallel case removed on 2026-08-12, where the two instruments
  measured the same quantity and nothing extra was actually spent). It is not
  the case here: the gap is real host work the user waits through — the logged
  total pipeline time was 4.66 s while the full end-to-end generation took
  6.1 s — which the bug traces to `MediaStorage.save_image` +
  `convert_image_to_bytes` + base64 in `openai_server.py`, and which the fixes
  cut from 0.97 s to 0.08 s of measured encode. See the ltx2-bf16-lora-restore
  sibling for the same severity-vs-body judgment written out at length.
- **Symptom:** ~1.5 s extra wall time per image generation/edit request for
  image-generating models (FLUX.1, FLUX.2) served via `trtllm-serve`
  (PR #12903); reported wall time was 4.6 s while actual latency was ~6.1 s.
  After #12903, PNG encode alone still cost 0.816 s per 1280x720 image;
  end-to-end `b64_json` requests took 12.71 s vs 10.78 s post-fix on
  FLUX.2-dev/B200 — 1.93 s (15.2%) per request (PR #13074 benchmarks).
- **Root cause:** two stacked host-side costs in the image response path of
  `tensorrt_llm/serve/openai_server.py` / `media_storage.py`: (1)
  `MediaStorage.save_image()` and `convert_image_to_bytes()` each
  independently ran the full tensor→PIL→PNG encode, so every `b64_json`
  response encoded the PNG twice plus an unneeded disk write; (2) both used
  PIL `optimize=True`, a CPU-expensive max-effort compression (~0.7–1 s per
  image) although PNG is lossless at every compression level.
- **How introduced:** unknown — not stated in the PRs; the encode path
  shipped this way with visual-gen serving (below-expectation from the
  start, not a regression from a faster state).
- **Fix mechanism:** #12903 skips the disk save entirely for `b64_json`
  responses (the common case) and only calls `save_image()` in URL mode,
  eliminating one full PNG encode and the disk I/O; #13074 replaces
  `optimize=True` with `compress_level=1` in `MediaStorage` PNG saves
  (10.9x faster encode, +8.8% file size, still lossless).
- **Detection signal:** per-request serving latency far above the
  model-reported generation wall time, with the gap in host-side
  post-processing (CPU-bound image encode, no GPU activity); check for
  expensive encodes on the response path with
  `grep -rn "optimize=True" tensorrt_llm/serve/` and for duplicate
  encode pipelines (`save_image` + `convert_image_to_bytes` on the same
  output).
- **Prevention/guard:** a *behavioural* guard exists and is easy to miss
  because it landed in a third PR: **#13372** ("[https://nvbugs/6064029][test]
  Visual gen b64 path regression tests", MERGED 2026-04-24) adds
  `tests/unittest/_torch/visual_gen/test_trtllm_serve_endpoints.py`, which pins
  the b64 path's *shape* — that the `b64_json` response does not also run the
  disk-save encode — so a re-introduction of the double encode fails a test
  rather than only moving a number. What is still missing is a **latency bar**:
  no perf CI test measures the server wall-time-vs-generation-time delta, and
  #13074's `compress_level` choice is unguarded, so a future PIL default or a
  revert to `optimize=True` would regress silently. Add the bar; also a review
  checklist item: response serialization must not re-run work already done for
  storage, and lossless encoders on the request path default to
  speed-oriented settings.
- **Generalizes to:** `pattern-host-work-on-hot-path` — hidden host-side
  work on the per-request path; carries to any serve-endpoint
  post-processing (base64/JSON serialization of large tensors, video
  encode paths, audio resampling), library default flags tuned for size
  or quality rather than latency (PIL/ffmpeg/zlib levels), and
  save-then-reencode duplication wherever a response is both persisted
  and returned inline.
