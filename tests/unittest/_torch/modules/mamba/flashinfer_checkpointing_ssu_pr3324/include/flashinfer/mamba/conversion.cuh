#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#ifdef FLASHINFER_ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace flashinfer::mamba::conversion {

inline __device__ float toFloat(float f) { return f; }

inline __device__ float toFloat(__half h) { return __half2float(h); }

#ifdef FLASHINFER_ENABLE_BF16
inline __device__ float toFloat(__nv_bfloat16 val) { return __bfloat162float(val); }
#endif

// No accuracy loss: int8_t / int16_t range fits exactly in float32 (24-bit
// mantissa represents all integers up to 2^24 = 16M exactly).
inline __device__ float toFloat(int8_t val) { return static_cast<float>(val); }
inline __device__ float toFloat(int16_t val) { return static_cast<float>(val); }

// fp8 e4m3 → fp32.  Goes via __half (cuda_fp8 library has the implicit
// conversion that compiles to `cvt.rn.f16.e4m3` PTX on sm_89+), then
// __half2float for the final step.  No direct fp8→fp32 PTX op exists.
inline __device__ float toFloat(__nv_fp8_e4m3 val) {
  return __half2float(static_cast<__half>(val));
}

// Packed 2-element conversion: convert a packed pair to float2.
// Uses native packed intrinsics for bf16/fp16 (fewer PRMT/SHF instructions).
inline __device__ float2 toFloat2(float2 packed) { return packed; }

inline __device__ float2 toFloat2(__half2 packed) { return __half22float2(packed); }

// Pointer-based overloads: read two consecutive elements and convert to float2.
// Dispatches to the packed intrinsic for bf16/fp16 via the overloads above.
inline __device__ float2 toFloat2(float const* ptr) { return {ptr[0], ptr[1]}; }

inline __device__ float2 toFloat2(__half const* ptr) {
  return toFloat2(*reinterpret_cast<__half2 const*>(ptr));
}

#ifdef FLASHINFER_ENABLE_BF16
// inline __device__ float2 toFloat2(__nv_bfloat162 packed) { return __bfloat1622float2(packed); }

inline __device__ float2 toFloat2(__nv_bfloat162 packed) {
  // bf16 is the upper 16 bits of f32 — shift/mask is cheaper than PRMT byte permutation.
  // NOTE: this ignores denormals
  uint32_t bits = reinterpret_cast<uint32_t const&>(packed);
  float2 out;
  out.x = __uint_as_float(bits << 16);          // low bf16 → upper 16 bits of f32
  out.y = __uint_as_float(bits & 0xFFFF0000u);  // high bf16 already in upper 16 bits
  return out;
}

inline __device__ float2 toFloat2(__nv_bfloat16 const* ptr) {
  return toFloat2(*reinterpret_cast<__nv_bfloat162 const*>(ptr));
}

// Paired f32 → bf16 conversion: pack two f32 values into __nv_bfloat162.
// Uses native cvt.rn.bf16x2.f32 — single instruction, round-to-nearest-even.
inline __device__ __nv_bfloat162 fromFloat2(float2 val) {
  uint32_t result;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(result) : "f"(val.y), "f"(val.x));
  return reinterpret_cast<__nv_bfloat162 const&>(result);
}

#endif

inline __device__ float2 toFloat2(int8_t const* ptr) { return {toFloat(ptr[0]), toFloat(ptr[1])}; }
inline __device__ float2 toFloat2(int16_t const* ptr) { return {toFloat(ptr[0]), toFloat(ptr[1])}; }

inline __device__ void convertAndStore(float* output, float input) { *output = input; }

inline __device__ void convertAndStore(__half* output, float input) {
  *output = __float2half(input);
}

#ifdef FLASHINFER_ENABLE_BF16
inline __device__ void convertAndStore(__nv_bfloat16* output, float input) {
  *output = __float2bfloat16(input);
}
#endif

inline __device__ void convertAndStore(int16_t* output, float input) {
  // Symmetric clip: [-max, max] (not [-max-1, max]) so that negation is safe.
  // Matches Triton reference which clips to [-32767, 32767] before storing.
  constexpr float int16_max = static_cast<float>(std::numeric_limits<int16_t>::max());
  input = fminf(fmaxf(input, -int16_max), int16_max);
  *output = static_cast<int16_t>(__float2int_rn(input));
}

// =============================================================================
// Philox-4x32 PRNG (matches Triton's tl.randint)
// =============================================================================

// Generates four pseudorandom uint32s from (seed, offset) using the Philox-4x32 algorithm.
// Produces bit-identical output to Triton's tl.randint4x(seed, offset, n_rounds).
// The offset is int64 and split across Philox c0 (low 32 bits) and c1 (high
// 32 bits) — matches the i64 path of `randint4x` in triton/language/random.py.
// Provides 2^64 unique counter values per seed, avoiding collisions in large
// caches where `cache_slot * stride` exceeds 2^32.
// All four outputs (c0..c3) are independent and uniformly distributed.
template <int n_rounds = 10>
__device__ __forceinline__ void philox_randint4x(int64_t seed, int64_t offset, uint32_t& r0,
                                                 uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t PHILOX_KEY_A = 0x9E3779B9u;
  constexpr uint32_t PHILOX_KEY_B = 0xBB67AE85u;
  constexpr uint32_t PHILOX_ROUND_A = 0xD2511F53u;
  constexpr uint32_t PHILOX_ROUND_B = 0xCD9E8D57u;

  uint32_t k0 = static_cast<uint32_t>(static_cast<uint64_t>(seed));
  uint32_t k1 = static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32);
  uint64_t uoffset = static_cast<uint64_t>(offset);
  uint32_t c0 = static_cast<uint32_t>(uoffset);
  uint32_t c1 = static_cast<uint32_t>(uoffset >> 32);
  uint32_t c2 = 0, c3 = 0;

#pragma unroll
  for (int i = 0; i < n_rounds; i++) {
    uint32_t _c0 = c0, _c2 = c2;
    c0 = __umulhi(PHILOX_ROUND_B, _c2) ^ c1 ^ k0;
    c2 = __umulhi(PHILOX_ROUND_A, _c0) ^ c3 ^ k1;
    c1 = PHILOX_ROUND_B * _c2;
    c3 = PHILOX_ROUND_A * _c0;
    k0 += PHILOX_KEY_A;
    k1 += PHILOX_KEY_B;
  }
  r0 = c0;
  r1 = c1;
  r2 = c2;
  r3 = c3;
}

// Generates a pseudorandom uint32 from (seed, offset) using the Philox-4x32 algorithm.
// Produces bit-identical output to Triton's tl.randint(seed, offset, n_rounds).
// The offset is int64 (low/high split across Philox c0/c1) — see
// philox_randint4x for the full rationale.
// NOTE: This discards 3 of the 4 Philox outputs. For better throughput, use
// philox_randint4x to get all 4 outputs from a single Philox invocation.
template <int n_rounds = 10>
__device__ __forceinline__ uint32_t philox_randint(int64_t seed, int64_t offset) {
  uint32_t r0, r1, r2, r3;
  philox_randint4x<n_rounds>(seed, offset, r0, r1, r2, r3);
  return r0;
}

// =============================================================================
// Stochastic rounding: fp32 → fp16
// =============================================================================

// Software stochastic rounding: convert one fp32 value to fp16 using 13 random bits.
// Adds random noise at the sub-fp16-mantissa position, then truncates.
// rand13: 13-bit random value in bits [12:0].
__device__ __forceinline__ uint16_t cvt_rs_f16_sw(float x, uint32_t rand13) {
  uint32_t bits = __float_as_uint(x);
  uint32_t sign = bits & 0x80000000u;
  uint32_t abs_bits = bits & 0x7FFFFFFFu;

  // fp32 has 23 mantissa bits, fp16 has 10. The 13 LSBs are the remainder.
  // Add 13-bit random noise at bits [12:0]. Carry into bit 13 → round up.
  abs_bits += (rand13 & 0x1FFFu);

  // Convert to fp16 by truncation.
  uint32_t f32_exp = (abs_bits >> 23) & 0xFFu;
  uint32_t f32_mantissa = abs_bits & 0x7FFFFFu;

  uint16_t f16_bits;
  if (f32_exp == 0xFF) {
    f16_bits = (f32_mantissa != 0) ? 0x7E00u : 0x7C00u;  // NaN or Inf
  } else if (f32_exp > 142) {                            // 127 + 15 = 142 → overflow to Inf
    f16_bits = 0x7C00u;
  } else if (f32_exp < 113) {  // 127 - 14 = 113 → underflow to zero
    f16_bits = 0;
  } else {
    uint16_t f16_exp = static_cast<uint16_t>(f32_exp - 112);  // rebias: 127→15
    uint16_t f16_mantissa = static_cast<uint16_t>(f32_mantissa >> 13);
    f16_bits = (f16_exp << 10) | f16_mantissa;
  }

  return static_cast<uint16_t>(sign >> 16) | f16_bits;
}

// Forward declaration (defined below, after cvt_rs_f16x2_f32).
__device__ __forceinline__ uint32_t cvt_rs_f16x2_f32(float a, float b, uint32_t rbits);

// Stochastic rounding: convert one fp32 value to fp16 using 13 random bits.
// On sm_100a+: uses PTX cvt.rs.f16x2.f32 with a dummy zero second input.
// On other archs: software emulation.
__device__ __forceinline__ __half cvt_rs_f16_f32(float x, uint32_t rand13) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)
  // Pack rand13 into rbits[12:0] (for PTX operand b → low half → our x).
  // High half gets zero noise for the dummy input.
  uint32_t rbits = rand13 & 0x1FFFu;
  uint32_t packed = cvt_rs_f16x2_f32(x, 0.0f, rbits);
  return __ushort_as_half(static_cast<uint16_t>(packed & 0xFFFFu));
#else
  return __ushort_as_half(cvt_rs_f16_sw(x, rand13));
#endif
}

// Stochastic rounding: convert two fp32 values to packed fp16x2 using random bits.
// On sm_100a+: uses PTX cvt.rs.f16x2.f32 instruction.
// On other archs: software emulation matching the hardware behavior.
//
// rbits layout (from PTX docs):
//   bits [28:16] = 13 random bits for PTX operand "a" (→ d[31:16], high half)
//   bits [12:0]  = 13 random bits for PTX operand "b" (→ d[15:0], low half)
//   bits [31:29] and [15:13] = unused (zero)
// from: https://docs.nvidia.com/cuda/parallel-thread-execution/#cvt-rs-rbits-layout-f16
//
// Our asm maps: %1→C++ a→PTX b→d[15:0], %2→C++ b→PTX a→d[31:16]
// So: C++ a uses rbits[12:0], C++ b uses rbits[28:16].
__device__ __forceinline__ uint32_t cvt_rs_f16x2_f32(float a, float b, uint32_t rbits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)
  uint32_t packed;
  asm("cvt.rs.f16x2.f32 %0, %2, %1, %3;"
      : "=r"(packed)
      : "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(rbits));
  return packed;
#else
  uint32_t rand_a = rbits & 0x1FFFu;          // bits [12:0] → C++ a (PTX b → low half)
  uint32_t rand_b = (rbits >> 16) & 0x1FFFu;  // bits [28:16] → C++ b (PTX a → high half)
  uint16_t a_fp16 = __half_as_ushort(cvt_rs_f16_f32(a, rand_a));
  uint16_t b_fp16 = __half_as_ushort(cvt_rs_f16_f32(b, rand_b));
  return static_cast<uint32_t>(a_fp16) | (static_cast<uint32_t>(b_fp16) << 16);
#endif
}

// Stochastic rounding store: generates Philox random bits and converts fp32 → fp16 in one call.
// PHILOX_ROUNDS: number of Philox rounds (compile-time), must be > 0.
// seed: Philox seed (from params.rand_seed).
// offset: unique per-element offset (e.g. d * DSTATE + i) for deterministic randomness.
template <int PHILOX_ROUNDS>
inline __device__ void convertSRAndStore(__half* output, float input, int64_t seed,
                                         uint32_t offset) {
  uint32_t rand = philox_randint<PHILOX_ROUNDS>(seed, offset);
  *output = cvt_rs_f16_f32(input, rand & 0x1FFFu);
}

// =============================================================================
// Stochastic rounding: fp32 → fp8 (e4m3)
// =============================================================================

// Bit-reverse 16 bits.  HW gives both elements of a pair 16 bits of independent
// SR randomness while consuming a single 16-bit chunk per pair: one element uses
// the chunk straight, the other uses bitrev16 of it.  Two bitrev16'd halves of a
// uniformly random 16-bit value remain uniformly distributed and are statistically
// independent, so each element gets full 16-bit unbiased SR while sharing one
// 16-bit register with its pair-mate.
//
// Implementation: PTX `brev.b32` (CUDA intrinsic `__brev`) is a single-ALU-op
// 32-bit reverse; we right-shift by 16 to land the 16 LSBs of input in the 16
// LSBs of output.  Total: 2 SASS instructions (brev + shf/shr), vs the prior
// software 4-step mask/shift/OR chain (~12-16 inst).
__device__ __forceinline__ uint32_t bitrev16(uint32_t b) { return __brev(b) >> 16; }

// Software stochastic rounding: convert one fp32 value to e4m3 (FN, satfinite) using 16 random
// bits.
//
// Algorithm: place `rand16` at the top of the discarded mantissa range, then truncate.
//   shift_truncate = 20  for normal binade (unbiased >= -6)
//                  = 14 - unbiased  for subnormal/underflow
//   contribution   = rand16 << (shift_truncate - 16)
//   total          = mant24 + contribution        (in uint64 to avoid overflow)
//   int_part       = total >> shift_truncate
// Then re-encode int_part as e4m3, handling subnormal→normal transitions and saturation.
//
// Saturation (satfinite):
//   |x| > 448  → ±448  (max finite e4m3 = 0x7E)
//   ±Inf       → ±448
//   NaN        → canonical NaN with sign preserved (S|1111|111 = 0x7F or 0xFF)
//
// Verified bitwise against HW (cvt.rs.satfinite.e4m3x4.f32 on sm_100a) across 22528
// inputs spanning subnormal, normal, and saturation regions during the SR
// reverse-engineering effort — see .plans/e4m3_stochastic_rounding.md.
__device__ __forceinline__ uint8_t cvt_rs_e4m3_sw(float x, uint32_t rand16) {
  uint32_t bits = __float_as_uint(x);
  uint32_t sign = (bits >> 31) & 1u;
  uint32_t abs_bits = bits & 0x7FFFFFFFu;
  uint32_t f32_exp = (abs_bits >> 23) & 0xFFu;
  uint32_t f32_mant = abs_bits & 0x7FFFFFu;

  // NaN / Inf
  if (f32_exp == 0xFFu) {
    if (f32_mant != 0) {
      return static_cast<uint8_t>(0x7Fu | (sign << 7));  // canonical e4m3 NaN
    } else {
      return static_cast<uint8_t>(0x7Eu | (sign << 7));  // Inf → ±max finite
    }
  }

  // fp32 zero / denormal → e4m3 zero (with sign).
  if (f32_exp == 0u) {
    return static_cast<uint8_t>(sign << 7);
  }

  int unbiased = static_cast<int>(f32_exp) - 127;
  uint64_t mant24 = 0x800000u | f32_mant;  // implicit-1 + mantissa, 24-bit
  int shift_truncate = (unbiased >= -6) ? 20 : (14 - unbiased);
  int rand_shift = shift_truncate - 16;
  uint64_t rand_contrib;
  if (rand_shift < 0) {
    // Defensive: shift_truncate < 16 shouldn't happen for valid normal/subnormal e4m3.
    rand_contrib = static_cast<uint64_t>(rand16 & 0xFFFFu) >> (-rand_shift);
  } else if (rand_shift < 56) {
    rand_contrib = static_cast<uint64_t>(rand16 & 0xFFFFu) << rand_shift;
  } else {
    rand_contrib = 0;
  }
  uint64_t total = mant24 + rand_contrib;
  // Guard the shift: shift_truncate reaches 140 for tiny-normal fp32
  // (unbiased < -49), which is UB for a uint64_t shift.  Mathematically the
  // result is 0 there (the value is far below e4m3's smallest subnormal),
  // so flush to int_part = 0.  The downstream subnormal branch rounds it to ±0.
  uint32_t int_part = (shift_truncate >= 64) ? 0u : static_cast<uint32_t>(total >> shift_truncate);

  if (unbiased >= -6) {
    // Started in normal binade.  int_part ∈ [8, 15] normally; can overflow to 16+ if rand
    // bumped the exponent.
    int e4m3_exp = unbiased + 7;
    while (int_part >= 16u) {
      int_part >>= 1;
      e4m3_exp += 1;
    }
    if (e4m3_exp > 15 || (e4m3_exp == 15 && (int_part & 0x7u) == 7u)) {
      return static_cast<uint8_t>(0x7Eu | (sign << 7));
    }
    return static_cast<uint8_t>((sign << 7) | (e4m3_exp << 3) | (int_part & 0x7u));
  } else {
    // Started in subnormal/underflow.  int_part:
    //   0       → zero
    //   1..7    → subnormal e4m3
    //   8..15   → smallest normal binade (e4m3_exp = 1)
    //   16+     → higher normal binades (rare; only if rand pushed up multiple binades)
    if (int_part == 0u) {
      return static_cast<uint8_t>(sign << 7);
    }
    if (int_part <= 7u) {
      return static_cast<uint8_t>((sign << 7) | int_part);
    }
    int e4m3_exp = 1;
    while (int_part >= 16u) {
      int_part >>= 1;
      e4m3_exp += 1;
    }
    if (e4m3_exp > 15 || (e4m3_exp == 15 && (int_part & 0x7u) == 7u)) {
      return static_cast<uint8_t>(0x7Eu | (sign << 7));
    }
    return static_cast<uint8_t>((sign << 7) | (e4m3_exp << 3) | (int_part & 0x7u));
  }
}

// Stochastic rounding: convert four fp32 values to packed fp8x4 e4m3 using random bits.
// On sm_100a+: uses PTX cvt.rs.satfinite.e4m3x4.f32 (combined stochastic-round + saturate).
// On other archs: software fallback via cvt_rs_e4m3_sw.
//
// Output layout (low byte first):
//   packed[ 7: 0] = e4m3(a)
//   packed[15: 8] = e4m3(b)
//   packed[23:16] = e4m3(c)
//   packed[31:24] = e4m3(d)
//
// rbits layout (per PTX docs + empirical HW oracle, see .plans/e4m3_stochastic_rounding.md):
//   bits [31:16] = pair rbits for PTX operands a/b (high two outputs)
//   bits [15: 0] = pair rbits for PTX operands e/f (low two outputs)
// Each PAIR shares its 16-bit chunk: HW uses the chunk straight for the "even"
// PTX operand (b, f → low byte of pair output) and bitrev16 of the chunk for
// the "odd" operand (a, e → high byte of pair output).  Both elements get the
// full 16 bits of independent SR randomness this way.
//   our `a` (PTX f, → byte 0) uses        rbits[15: 0]
//   our `b` (PTX e, → byte 1) uses bitrev16(rbits[15: 0])
//   our `c` (PTX b, → byte 2) uses        rbits[31:16]
//   our `d` (PTX a, → byte 3) uses bitrev16(rbits[31:16])
//
// PTX syntax `cvt.rs.satfinite.e4m3x4.f32 d, {a3, a2, a1, a0}, rbits` writes
// e4m3(a_i) into byte i of d.  We want byte 0 = e4m3(a), so the source-vector
// ordering is {d, c, b, a} = {%4, %3, %2, %1}.  See PTX ISA:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
__device__ __forceinline__ uint32_t cvt_rs_e4m3x4_f32(float a, float b, float c, float d,
                                                      uint32_t rbits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)
  uint32_t packed;
  asm("cvt.rs.satfinite.e4m3x4.f32 %0, {%4, %3, %2, %1}, %5;"
      : "=r"(packed)
      : "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(__float_as_uint(c)),
        "r"(__float_as_uint(d)), "r"(rbits));
  return packed;
#else
  uint32_t low_chunk = rbits & 0xFFFFu;
  uint32_t high_chunk = (rbits >> 16) & 0xFFFFu;
  uint8_t pa = cvt_rs_e4m3_sw(a, low_chunk);             // PTX f → byte 0
  uint8_t pb = cvt_rs_e4m3_sw(b, bitrev16(low_chunk));   // PTX e → byte 1
  uint8_t pc = cvt_rs_e4m3_sw(c, high_chunk);            // PTX b → byte 2
  uint8_t pd = cvt_rs_e4m3_sw(d, bitrev16(high_chunk));  // PTX a → byte 3
  return static_cast<uint32_t>(pa) | (static_cast<uint32_t>(pb) << 8) |
         (static_cast<uint32_t>(pc) << 16) | (static_cast<uint32_t>(pd) << 24);
#endif
}

// =============================================================================
// Round-to-nearest-even + saturate: fp32 → int8
// =============================================================================

// cvt.rni.sat.s8.f32: single PTX instruction on sm_80+, replaces
// the F2I.S32 + VIMNMX(min 127) + VIMNMX(max -127) chain.
// Saturates to [-128, 127].  Callers using encode_scale = 127/amax
// guarantee |input| ≤ 127.0, so -128 is never produced.
__device__ __forceinline__ int8_t cvt_rni_sat_s8(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  int32_t result;
  asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(result) : "f"(x));
  return static_cast<int8_t>(result);
#else
  return static_cast<int8_t>(max(-128, min(127, __float2int_rn(x))));
#endif
}

// =============================================================================
// Stochastic rounding + saturate: fp32 → int8
// =============================================================================

// Software SR for int8: add uniform noise in [0, 1) then floor.
// Matches Triton's `floor(scaled_value + rand01)` where
// `rand01 = (rand & 0x00FFFFFF) * (1.0 / (1 << 24))`.
// Saturates to [-127, 127] (symmetric, matching encode_scale = 127/amax).
__device__ __forceinline__ int8_t cvt_rs_sat_s8(float x, uint32_t rand_bits) {
  float const rand01 =
      static_cast<float>(rand_bits & 0x00FFFFFFu) * (1.0f / static_cast<float>(1 << 24));
  // `__float2int_rd` (round toward -infinity) fuses `floorf` + `__float2int_rz`
  // into a single `cvt.rmi.s32.f32` SASS instruction, saving one FRND per call.
  int32_t const clamped = max(-127, min(127, __float2int_rd(x + rand01)));
  return static_cast<int8_t>(clamped);
}

// Stochastic rounding: convert four fp32 values to packed s8x4 using a single
// 32-bit random integer.  Analogous to cvt_rs_e4m3x4_f32: 16-bit chunks are
// reused via bitrev16 so each output gets 16 bits of independent randomness
// while consuming only one shared 16-bit chunk per pair (two bitrev16'd halves
// of a uniform 16-bit value remain uniformly distributed and statistically
// independent).
//
// 16-bit entropy per element is far more than int8 SR requires: the rounding
// decision compares against a fractional residual with at most ~7 bits of
// meaningful precision for int8, so no quality loss vs the 24-bit scalar path.
//
// Amortization: 1 random u32 → 4 SR int8s.  A single Philox call (4 u32s)
// covers 16 int8 conversions, a 4× reduction in PRNG cost vs the scalar
// cvt_rs_sat_s8 path.
__device__ __forceinline__ uint32_t cvt_rs_sat_s8x4_f32(float a, float b, float c, float d,
                                                        uint32_t rbits) {
  uint32_t const low_chunk = rbits & 0xFFFFu;
  uint32_t const high_chunk = (rbits >> 16) & 0xFFFFu;
  constexpr float kInv16 = 1.0f / static_cast<float>(1u << 16);

  float const r_a = static_cast<float>(low_chunk) * kInv16;
  float const r_b = static_cast<float>(bitrev16(low_chunk)) * kInv16;
  float const r_c = static_cast<float>(high_chunk) * kInv16;
  float const r_d = static_cast<float>(bitrev16(high_chunk)) * kInv16;

  // `__float2int_rd` (round toward -infinity) emits a single `cvt.rmi.s32.f32`
  // SASS op, fusing the `floorf` + `__float2int_rz` chain into one instruction.
  int32_t const pa = max(-127, min(127, __float2int_rd(a + r_a)));
  int32_t const pb = max(-127, min(127, __float2int_rd(b + r_b)));
  int32_t const pc = max(-127, min(127, __float2int_rd(c + r_c)));
  int32_t const pd = max(-127, min(127, __float2int_rd(d + r_d)));

  return (static_cast<uint32_t>(pa) & 0xFFu) | ((static_cast<uint32_t>(pb) & 0xFFu) << 8) |
         ((static_cast<uint32_t>(pc) & 0xFFu) << 16) | ((static_cast<uint32_t>(pd) & 0xFFu) << 24);
}

}  // namespace flashinfer::mamba::conversion
