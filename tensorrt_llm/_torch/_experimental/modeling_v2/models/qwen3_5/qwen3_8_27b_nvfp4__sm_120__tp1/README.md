# `qwen3_8_27b_nvfp4__sm_120__tp1`

Qwen3.8-27B (family `qwen3_5`, architecture `Qwen3_5ForConditionalGeneration`), the dense
Qwen3.5 Gated-DeltaNet hybrid on one RTX PRO 6000 Blackwell, tuned for long shared-prefix
serving from concurrency 1 to 128. It claims the ModelOpt MIXED_PRECISION checkpoint (FP8
attention and Gated-DeltaNet projections, NVFP4 MLP and lm_head) with an fp8 KV cache
(`kv_cache_config.dtype='fp8'`) and the 8-bit recurrent-state pool
(`kv_cache_config.mamba_ssm_cache_dtype='int8'`, which only a target that claims the model may
ask for), without speculative decoding and with eager prefill steps. The vision half of the
composite checkpoint is not part of it: it serves text, and a request carrying images or video
fails with an error that names `modeling_v2='off'`.

Every projection is routed by the step's row count: the FP8 projections run load-time NVFP4
copies of their weights (a CUDA-core W4A16 GEMV at the fewest rows, W4A16 Marlin up to 32 or 64
rows, W4A4 above), and the producers before a W4A4 GEMM quantize in their own epilogue. Its own
CUDA kernels live in `csrc/` and build into `libth_modeling_v2_qwen3_5.so`: the W4A16 GEMV
(plain and with the fused conv1d update), a cuda-core W4A4 GEMM for decode rows, the Marlin
W4A16 GEMM with launch tables for this checkpoint's shapes (`marlin/`), the gate_up GEMM with
the SwiGLU + NVFP4 epilogue, and the attention op: the engine's KV-cache append, then the
SM120 e4m3 head-256 context FMHA (`fmha/`) and the XQA decode kernel compiled ahead of time
(`xqa/`, with the one-wave multi-block cap and the shared-prefix (cascade) pass).

Where it differs from the built-in `Qwen3_5VLModel` at the same settings:

| | built-in | this target |
|---|---|---|
| FP8 projections | FP8 x FP8 (static input scales) | NVFP4 copies: W4A16 at decode rows, W4A4 above |
| lm_head | NVFP4 entry dequantized to a bf16 weight | the checkpoint's NVFP4: W4A16 Marlin below 56 rows, W4A4 (fp32 logits) from 56 |
| recurrent state | bf16 | 8-bit rows with fp32 row scales |
| context softmax | `expf` | base-2 `ex2.approx` |
| shared-prefix decode | each request's KV walked alone | one prefix pass over all requests (FP8 Q/P), each suffix alone |
| final layer of a long prefill step | every row | gate, o_proj, MLP and final norm on the read rows only |

The target and the entries it owns are certified on sm_120.
