*Generated: 2026-10-31*

# MiniMax-H3 fused QK-norm + RoPE + FP8 cast: scale tables

Calibration files of the static-scale variant of `fused_fp8_qk.py` (board thread §657, lever 8, boxes 25), kept so the labelled alternative row of the stack post is reproducible. The composition uses the **dynamic** variant (`H3_FP8_SCALE_FUSION=dynamic`), which needs no table.

## Contents

1. [Flags](#1-flags)
2. [Tables](#2-tables)
3. [How a table is made](#3-how-a-table-is-made)
4. [Composition with Ulysses](#4-composition-with-ulysses)

## 1. Flags

| Variable | Dynamic (composition) | Static alternative row (job 4) |
|---|---|---|
| `H3_FP8_SCALE_FUSION` | `dynamic` | `static` |
| `H3_FP8_STATIC_SCALES` | – | `table-job4-heldout1-margin1.0.json` |
| `H3_FP8_STATIC_MARGIN` | – | `1.0` |
| `H3_FP8_FUSION_SPLIT` / `_BLOCK_H` / `_WARPS` | `1` / `8` / `4` | same |
| `H3_FP8_SAT_LOG` | optional | `<json>` writes clipped-element counts per layer and tensor |

The table JSON maps `str(layer_idx)` to `[descale_q, descale_k, descale_v]`; a descale is `max(amax / 448, 1 / (448 · 512))` rounded to BF16, the same value the dynamic quantizer produces. The loader also accepts the raw calibration log (`{"descale_max": {...}}`).

## 2. Tables

| File | Calibration set | Margin | e2e S over dynamic (R200, 544×960×124, U1) | Ship check (¶14884 rule) |
|---|---|---|---|---|
| `table-job1-in-calibration.json` | gate seeds 42 / 7 / 123 of the ship prompt + the 12 ship prompts (job 921774 dynamic log) | 1.0 | 1.038 (job 922083) | pass at the limit, 2 of 10 lost, in-calibration: not a valid check |
| `table-job4-heldout1-margin1.0.json` | `heldout-prompts-544x960x124.json` (12 held-out prompts, seed 42; job 922208) | 1.0 | 1.039 | **pass at the limit**, 2 of 10 lost (thunderstorm, blacksmith), 0 won; 1,537 clipped elements on the ship clips (`job4-saturation-ship.json`), `job4-heldout1-vs-ship.json` gives the ship-set amax over the table (63 of 150 pairs above 1.0, max 1.223) |
| `table-job5-heldout2-margin1.194.json` | `heldout2-prompts-544x960x124.json` (second split; job 922502) | 1.194 (`job5-margin.json`: smallest uniform factor covering the ship and first-split logs) | 1.046 | **fail**: persistent striping band on the surf clip; 12 clipped V elements (`job5-saturation-ship.json`) |

Reading (lead ¶15055): widening every scale costs more than clipping a dozen outliers; the static direction is range, not saturation.

## 3. How a table is made

1. Run the dynamic variant with `H3_FP8_AMAX_LOG=<json>` over a prompt set (the sweep used `gen_quality_batch.py` with the prompt JSONs here; seed 42, 28 steps, 544×960×124); the log holds the running maximum descale per layer and tensor.
2. Merge the logs of several runs by element-wise maximum (`descale_max`), keep the file as the table.
3. Choose `H3_FP8_STATIC_MARGIN`: 1.0 reproduces the job-4 row; a factor above 1.0 widens every scale uniformly (job 5 used the smallest factor that covered two other logs and failed the ship check).
4. Run with `H3_FP8_SCALE_FUSION=static H3_FP8_STATIC_SCALES=<table> H3_FP8_SAT_LOG=<json>` and read the saturation counts.

Job scripts and pulled results: `docs/minimax-h3/sweep-nvfp4/` in the workspace (`jobs/lever8-fp8-scale-fusion-r200.sbatch`, parts `calib` / `e2e` / `gate` / `ship`).

## 4. Composition with Ulysses

Under Ulysses the per-rank amax must be reduced (MAX) over the sequence-parallel group before the cast, so the dynamic variant adds all-reduces on the compute stream. How many depends on the entry point:

| Entry point | Used by | Dynamic mode: all-reduces per layer | Static mode |
|---|---|---|---|
| `h3_lever8::qk_norm_rope_fp8` (packed, also the split launch `H3_FP8_FUSION_SPLIT=1`) | synchronous Ulysses (the composed stack) | 1, on a 3-element tensor (q, k, v together; 50 per step) | 0 |
| `h3_lever8::norm_rope_fp8_tensor` via `norm_rope_tensor(..., fp8=True)` | asynchronous Ulysses pairing (`H3_L13_PAIR=fp8`), one call per tensor inside the async closures | 3 (150 per step, measured 39.5 ms at U4 19k, job 922930) | 0 |

**Dynamic mode and the async path do not compose.** The per-tensor API cannot merge its three reductions into one: the async closures quantize Q, K and V at different points, so each call reduces its own amax. Rule (lead ruling ¶15146): an asynchronous-Ulysses arm that carries lever 8 runs the static table (`table-job4-heldout1-margin1.0.json`, the gated alternative row) or does not carry lever 8; the synchronous arm runs dynamic mode and is unaffected.
