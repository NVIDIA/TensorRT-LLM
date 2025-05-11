#include <vector_types.h>

extern "C"
{
    /*
     * Rounding mode modifiers:
     *   _rn : round to nearest even (default)
     *   _rm : round towards negative infinity
     *   _rp : round towards positive infinity
     *   _rz : round towards zero
     *
     * _ftz : flush denormalized values to zero
     */

    /*
     * FFMA2 - fused multiply-add
     */
    __device__ float2 __nv_ptx_builtin_ocg_ffma2(float2 a, float2 b, float2 c);
    __device__ float2 __nv_ptx_builtin_ocg_ffma2_rn(float2 a, float2 b, float2 c);
    __device__ float2 __nv_ptx_builtin_ocg_ffma2_rm(float2 a, float2 b, float2 c);
    __device__ float2 __nv_ptx_builtin_ocg_ffma2_rp(float2 a, float2 b, float2 c);
    __device__ float2 __nv_ptx_builtin_ocg_ffma2_rz(float2 a, float2 b, float2 c);
    __device__ float2 __nv_ptx_builtin_ocg_ffma2_ftz(float2 a, float2 b, float2 c);
    __device__ float2 __nv_ptx_builtin_ocg_ffma2_ftz_rn(float2 a, float2 b, float2 c);
    __device__ float2 __nv_ptx_builtin_ocg_ffma2_ftz_rm(float2 a, float2 b, float2 c);
    __device__ float2 __nv_ptx_builtin_ocg_ffma2_ftz_rp(float2 a, float2 b, float2 c);
    __device__ float2 __nv_ptx_builtin_ocg_ffma2_ftz_rz(float2 a, float2 b, float2 c);

    /*
     * FADD2 - add
     */
    __device__ float2 __nv_ptx_builtin_ocg_fadd2(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fadd2_rn(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fadd2_rm(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fadd2_rp(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fadd2_rz(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fadd2_ftz(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fadd2_ftz_rn(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fadd2_ftz_rm(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fadd2_ftz_rp(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fadd2_ftz_rz(float2 a, float2 b);

    /*
     * FMUL2 - multiply
     */
    __device__ float2 __nv_ptx_builtin_ocg_fmul2(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fmul2_rn(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fmul2_rm(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fmul2_rp(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fmul2_rz(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fmul2_ftz(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fmul2_ftz_rn(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fmul2_ftz_rm(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fmul2_ftz_rp(float2 a, float2 b);
    __device__ float2 __nv_ptx_builtin_ocg_fmul2_ftz_rz(float2 a, float2 b);
}
