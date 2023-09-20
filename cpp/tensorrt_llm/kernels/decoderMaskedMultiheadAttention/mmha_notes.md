# Notes on the Masked Multihead Attention Kernel

## Types

* `T`: type of the `q`, `k`, `v`, `kv_cache` inputs and outputs: `float`, `uint16_t`, `__nv_bfloat16`, `__nv_fp8_e4m3`
* `Tk`: compute type internal to the kernel mapped from `T` as follows:
    * `float` -> `float`
    * `uint16_t` -> `uint16_t` (i.e. `half`)
    * `__nv_bfloat16` -> `__nv_bfloat16`
    * `__nv_fp8_e4m3` -> `float`

## Constraints

* `THREADS_PER_BLOCK`: in `{64, 128, 256}`
* `Dh`: `32 <= Dh <= 256`

## Constants

* `Dh_MAX`: round `Dh` up to the next power of 2
* `THREADS_PER_KEY`: `256 / THREADS_PER_BLOCK` in `{1, 2, 4}`
* `THREADS_PER_VALUE`: `Dh_MAX * sizeof(T) / 16`, except for `FP8` where `sizeof(T)` is assumed to be `4`.

Note that `THREADS_PER_KEY` is currently computed by the simple heuristic above which seems to work fine for the moment.

### Auxiliary vector types

* `Qk_vec_m`: vector for Q/K elements with memory precision depending on `T` and `Dh_MAX` in `(32, 64, 128, 256)`:
    * `float`: `(float, float, float2, float4)` with sizes `(4, 4, 8, 16)`
    * `uint16_t`: `(uint32_t, uint32_t, uint2, uint4)` with sizes `(4, 4, 8, 16)`
    * `__nv_bfloat16`: `(__nv_bfloat162, __nv_bfloat162, bf16_4_t, bf16_8_t)` with sizes `(4, 4, 8, 16)`
    * `__nv_fp8_e4m3`: `(fp8_4_t, fp8_4_t, fp8_4_t, fp8_4_t)` with sizes `(4, 4, 4, 4)`
* `Qk_vec_k`: vector for Q/K elements with kernel precision depending on `T` and `Dh_MAX` in `(32, 64, 128, 256)`:
    * `__nv_fp8_e4m3`: `(float4, float4, float4, float4)` with sizes `(16, 16, 16, 16)`
    * other types sames as `Qk_vec_m`

Associated constants are:

* `QK_VEC_SIZE`: `sizeof(Qk_vec_m) / sizeof(T)` in `{1, 2, 4}` depending on `T` and `Dh_MAX` in `(32, 64, 128, 256)`
    * `float`, `uint16_t`, `__nv_bfloat16` : `(1, 1, 2, 4)`
    * `__nv_fp8_e4m3`: `(4, 4, 4, 4)`
* `QK_VECS_PER_Dh_MAX`: `Dh_MAX / QK_VEC_SIZE` in `{8, 16, 32, 64}` depending on `T` and `Dh_MAX`
  in `(32, 64, 128, 256)`
    * `float`, `uint16_t`, `__nv_bfloat16`: `(32, 64, 64, 64)`
    * `__nv_fp8_e4m3`: `(8, 16, 32, 64)`
* `QK_ELTS_IN_16B`: `16 / sizeof(T)` in `{16, 8, 4}`
* `QK_VECS_IN_16B`: `16 / sizeof(Qk_vec_m)` in `{16, 8, 4}` and `<= QK_ELTS_IN_16B`

Note that `QK_ELTS_IN_16B / QK_VECS_IN_16B == QK_VEC_SIZE`.

Similarly, we have:

* `k_vec_m`: vector for K elements with memory precision depending on `T` and `THREADS_PER_KEY` in `(1, 2, 4)`:
  * `float`: `(float4, float2, float)` with sizes `(16, 8, 4)`
  * `uint16_t`: `(uint4, uint2, uint32_t)` with sizes `16, 8, 4)`
  * `__nv_bfloat16`: `(bf16_8_t, bf16_4_t, nv_bfloat162)` with sizes `(16, 8, 4)`
  * `__nv_fp8_e4m3`: `(fp8_4_t, fp8_4_t, fp8_4_t)` with sizes `(4, 4, 4)`
* `k_vec_k`: vector for K elements with kernel precision depending on `T` and `THREADS_PER_KEY` in `(1, 2, 4)`:
  * `__nv_fp8_e4m3`: `(float4, float4, float4)` with sizes `(16, 16, 16)`
  * other types sames as `k_vec_m`

Associated constants are:

* `K_VEC_SIZE`: `sizeof(k_vec_m) / sizeof(T)` in `{1, 2, 4}` depending on `T` and `THREADS_PER_KEY` in `(1, 2, 4)`
  * `float` : `(4, 2, 1)`
  * `uint16_t`: `(8, 4, 2)`
  * `__nv_bfloat16`: `(8, 4, 2)`
  * `__nv_fp8_e4m3`: `(4, 4, 4)`


## Memory Layout

Notation:

* `B`:  Batch size (number of sequences),
* `L`:  Sequence length,
* `D`:  Hidden dimension,
* `H`:  Number of heads,
* `Dh`: Hidden dimension per head - `Dh = D / H`.

### `k_cache`

The `k_cache` stores elements of the `T`.

Layout: `[B, H, Dh/x, L, x]` where `x == QK_ELTS_IN_16B`, i.e., `x == 16` (FP8), `x == 8` (FP16), `x == 4` (FP32)

Each thread writes `QK_VEC_SIZE` elements.

### `v_cache`

The `v_cache` stores elements of the `T`.

Layout: `[B, H, L, Dh]`

### `QKV` buffer

The `qkv` buffer stores elements of type `T`.

Layout: `[B, H, Dh]`

### Shared memory

Dynamic size of shared memory `smem_`: max over several expressions since the memory is reused in different contexts

## Notes on GEMMs in the context of the MMHA kernel

### GEMM in `DecoderSelfAttentionLayer.cc`

```c++
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  3 * local_hidden_units_,  // n
                                  batch_size,
                                  d_model_,  // k
                                  attention_weights->query_weight.kernel,
                                  3 * local_hidden_units_,  // n
                                  attention_input,
                                  d_model_,  // k
                                  qkv_buf_,
                                  3 * local_hidden_units_ /* n */);
```

* `A`: query, key, value weights with shape `d_model_ x 3 * local_hidden_units_`
* `B`: attention input with shape `batch_size x d_model`
* `C`: has shape `batch_size x 3 * local_hidden_units_`
