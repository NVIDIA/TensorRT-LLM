# MuseSpark-open on VR (sm_107), TEP8 decode: banked env layers

Append to the gen (decode) role env, in this order:

| file | applies to | what |
|---|---|---|
| `decode_base_levers.env` | every TEP8 decode server | the 6 bitwise base-env lever settings the banked layers were measured on (prep fold, packed prep append, overlap fold 2, hop input boundary, dist fabric + verify grid, drafter paged commit) |
| `banked_stack_bk3fa.env` | every TEP8 decode server | bs<=16 banked levers (step316 combo C, SWA 2-CTA, unpool direct wide, tile placement 33..128, o_proj L2 prefetch, B4 2-CTA producer, KF651 + v2; v3 is code) |
| `decode_large_batch.env` | mbs > 16 | the 5 large-batch base keys (inert at <= 128 tokens) |
| `w256_W6.env` | mbs > 16 (bs17..32) | W6: the 129..256-row wide paths + KF651 v2 |

The comment lines name the Hecate overlays the measurements ran on; this branch contains those overlays' code, so no overlay is needed.
The four layers reproduce the measured stacks (afcaf734 base gen env + the banked deltas): every other afcaf734 key equals its lever
auto default (dieexp lever audit, 2026-11-15).
