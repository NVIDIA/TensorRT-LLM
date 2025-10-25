import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.auto_deploy.custom_ops.agent_ops.cutedsl_gemm import utils

F32_or_F32x2 = Float32 | Tuple[Float32, Float32]


@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def sigmoid(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        # return 0.5 + 0.5 * cute.math.tanh(0.5 * x, fastmath=True)
        return 0.5 + 0.5 * tanh(0.5 * x)
    else:
        x_half = utils.mul_packed_f32x2((0.5, 0.5), x)
        tanh_x_half = (tanh(x_half[0]), tanh(x_half[1]))
        return utils.fma_packed_f32x2(tanh_x_half, (0.5, 0.5), (0.5, 0.5))


@dsl_user_op
def relu(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0))
    else:
        return cute.arch.fmax(x[0], Float32(0.0)), cute.arch.fmax(x[1], Float32(0.0))


@cute.jit
@dsl_user_op
def drelu(
    x: F32_or_F32x2, d_out: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        x_pos = cutlass.Boolean(x > 0)
        return d_out if x_pos else Float32(0.0), cute.arch.fmax(x, Float32(0.0))
    else:
        x0_pos = cutlass.Boolean(x[0] > 0)
        x1_pos = cutlass.Boolean(x[1] > 0)
        dx = (d_out[0] if x0_pos else Float32(0.0), d_out[1] if x1_pos else Float32(0.0))
        return dx, relu(x)


@dsl_user_op
def relu_sq(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0)) * x
    else:
        relu_x = (
            cute.arch.fmax(x[0], Float32(0.0)),
            cute.arch.fmax(x[1], Float32(0.0)),
        )
        return utils.mul_packed_f32x2(relu_x, x)


@cute.jit
@dsl_user_op
def drelu_sq(
    x: F32_or_F32x2, d_out: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    """
    ReLU squared backward pass: computes gradient w.r.t. x and recomputes forward
    Given: relu_sq_out = max(x, 0) * x, and d_out = grad w.r.t. relu_sq_out
    Returns: (dx, relu_sq_out) where:
    - dx = d_out * 2 * x if x > 0, else 0
    - relu_sq_out = max(x, 0) * x
    """
    if const_expr(not isinstance(x, tuple)):
        relu_x = relu(x)
        relu_sq_out = relu_x * x
        # Derivative: d/dx[max(x,0) * x] = 2*x if x > 0, else 0
        dx = 2.0 * (d_out * relu_x)
        return dx, relu_sq_out
    else:
        relu_x = relu(x)
        relu_sq_out = utils.mul_packed_f32x2(relu_x, x)
        dx = utils.mul_packed_f32x2((2.0, 2.0), utils.mul_packed_f32x2(d_out, relu_x))
        return dx, relu_sq_out


@dsl_user_op
def gelu_tanh_approx(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            = 0.5 * x * (1 + tanh(x * (0.797885 + 0.0356774 * x * x)))
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)  # ~0.797885
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi  # ~0.0356774
    if const_expr(not isinstance(x, tuple)):
        return 0.5 * (
            x
            # Currently cute.math.tanh(x, fastmath=True) generates very slow code
            # * (1 + cute.math.tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * (x * x)), fastmath=True))
            * (1.0 + tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * (x * x))))
        )
    else:
        x_sq = utils.mul_packed_f32x2(x, x)
        x_sq_scaled = utils.fma_packed_f32x2(
            x_sq,
            (sqrt_2_over_pi_coeff, sqrt_2_over_pi_coeff),
            (sqrt_2_over_pi, sqrt_2_over_pi),
        )
        z = utils.mul_packed_f32x2(x, x_sq_scaled)
        tanh_z = (tanh(z[0]), tanh(z[1]))
        x_tanh_z = utils.fma_packed_f32x2(tanh_z, x, x)
        return utils.mul_packed_f32x2((0.5, 0.5), x_tanh_z)


@dsl_user_op
def dgelu_tanh_approx(
    x: F32_or_F32x2, d_out: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    """
    GELU tanh approximation backward pass: computes gradient w.r.t. x and recomputes forward
    Given: gelu_out = 0.5 * x * (1 + tanh(x * (c1 + c2 * x^2))), and d_out = grad w.r.t. gelu_out
    Returns: (dx, gelu_out)

    Derivative uses the chain rule:
    d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
    where z = x * (c1 + c2 * x^2), dz/dx = c1 + 3 * c2 * x^2
    and sech^2(z) = 1 - tanh^2(z)
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)  # c1 ~0.797885
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi  # c2 ~0.0356774
    sqrt_2_over_pi_coeff_3 = 3.0 * sqrt_2_over_pi_coeff  # c3 ~0.01070322

    if const_expr(not isinstance(x, tuple)):
        # Compute z = x * (c1 + c2 * x^2)
        x_sq = x * x
        # tanh_z = cute.math.tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq), fastmath=True)
        tanh_z = tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq))
        half_tanh_z_plus_one = 0.5 + 0.5 * tanh_z
        gelu_out = x * half_tanh_z_plus_one

        # Compute gradient
        # sech^2(z) = 1 - tanh^2(z)
        sech2_z = 1 - tanh_z * tanh_z
        # dz/dx = c1 + 3 * c2 * x^2
        dz_dx = sqrt_2_over_pi + sqrt_2_over_pi_coeff_3 * x_sq
        # d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
        dgelu = half_tanh_z_plus_one + x * (0.5 * (sech2_z * dz_dx))

        dx = d_out * dgelu
        return dx, gelu_out
    else:
        # Compute z = x * (c1 + c2 * x^2)
        x_sq = utils.mul_packed_f32x2(x, x)
        x_sq_scaled = utils.fma_packed_f32x2(
            x_sq,
            (sqrt_2_over_pi_coeff, sqrt_2_over_pi_coeff),
            (sqrt_2_over_pi, sqrt_2_over_pi),
        )
        z = utils.mul_packed_f32x2(x, x_sq_scaled)
        tanh_z = (tanh(z[0]), tanh(z[1]))
        half_tanh_z_plus_one = utils.fma_packed_f32x2(tanh_z, (0.5, 0.5), (0.5, 0.5))
        gelu_out = utils.mul_packed_f32x2(x, half_tanh_z_plus_one)

        # Compute gradient
        # sech^2(z) = 1 - tanh^2(z)
        sech2_z = utils.fma_packed_f32x2(tanh_z, (-tanh_z[0], -tanh_z[1]), (1.0, 1.0))
        # dz/dx = c1 + 3 * c2 * x^2
        dz_dx = utils.fma_packed_f32x2(
            x_sq,
            (sqrt_2_over_pi_coeff_3, sqrt_2_over_pi_coeff_3),
            (sqrt_2_over_pi, sqrt_2_over_pi),
        )
        # d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
        sech2_dz_dx = utils.mul_packed_f32x2(sech2_z, dz_dx)
        x_sech2_dz_dx = utils.mul_packed_f32x2(x, sech2_dz_dx)
        dgelu = utils.fma_packed_f32x2(x_sech2_dz_dx, (0.5, 0.5), half_tanh_z_plus_one)

        dx = utils.mul_packed_f32x2(d_out, dgelu)
        return dx, gelu_out


@dsl_user_op
def silu(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """
    silu(x) = x * sigmoid(x) = x * (1 + tanh(x / 2)) / 2 = (0.5 * x) * tanh(0.5 * x) + (0.5 * x)
    This compiles down to 3 SASS instructions: FMUL to get 0.5 * x, MUFU.TANH, and FFMA.
    """
    if const_expr(not isinstance(x, tuple)):
        x_half = 0.5 * x
        # return x_half * cute.math.tanh(x_half, fastmath=True) + x_half
        return x_half * tanh(x_half) + x_half
    else:
        x_half = utils.mul_packed_f32x2((0.5, 0.5), x)
        tanh_x_half = (tanh(x_half[0]), tanh(x_half[1]))
        return utils.fma_packed_f32x2(x_half, tanh_x_half, x_half)


@dsl_user_op
def swiglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return silu(x) * y
    else:
        return utils.mul_packed_f32x2(silu(x), y)


@dsl_user_op
def dswiglu(
    x: F32_or_F32x2, y: F32_or_F32x2, d_out: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    """
    SwiGLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: swiglu_out = silu(x) * y, and d_out = grad w.r.t. swiglu_out
    Returns: (dx, dy, swiglu_out) where dx = d_out * y * d_silu(x), dy = d_out * silu(x)

    d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    This has been optimized to use fewer instructions (i.e. we expand things out
    to use FFMA instead of FADD and FMUL).
    """
    if const_expr(not isinstance(x, tuple)):
        # Compute sigmoid(x) using tanh: sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))
        # FMUL, MUFU.TANH, then FFMA
        sigmoid_x = sigmoid(x)
        silu_x = x * sigmoid_x  # FMUL
        silu_x_d_out = silu_x * d_out  # FMUL
        #   d_silu(x) * d_out
        # = sigmoid_x * (1 + x * (1 - sigmoid_x)) * d_out
        # = (sigmoid_x + sigmoid_x * x * (1 - sigmoid_x)) * d_out
        # = (sigmoid_x + silu_x * (1 - sigmoid_x)) * d_out
        # = (sigmoid_x + silu_x - silu_x * sigmoid_x) * d_out
        # = (sigmoid_x - silu_x * sigmoid_x) * d_out + silu_x * d_out
        d_silu_x_d_out = (sigmoid_x - silu_x * sigmoid_x) * d_out + silu_x_d_out  # FFMA, FFMA
        dx = d_silu_x_d_out * y  # FMUL
        dy = silu_x_d_out
        swiglu_out = silu_x * y  # FMUL
        # Overall it's 1 MUFU.TANH, 5 FMUL, 3 FFMA
        return dx, dy, swiglu_out
    else:
        # Compute sigmoid(x) and silu(x)
        sigmoid_x = sigmoid(x)
        silu_x = utils.mul_packed_f32x2(x, sigmoid_x)
        silu_x_d_out = utils.mul_packed_f32x2(silu_x, d_out)
        # d_silu(x) * d_out = (sigmoid_x - silu_x * sigmoid_x) * d_out + silu_x * d_out
        sigmoid_x_minus_silu_x_sigmoid_x = utils.fma_packed_f32x2(
            sigmoid_x, (-silu_x[0], -silu_x[1]), sigmoid_x
        )
        d_silu_x_d_out = utils.fma_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x, d_out, silu_x_d_out
        )
        dx = utils.mul_packed_f32x2(d_silu_x_d_out, y)
        dy = silu_x_d_out
        swiglu_out = utils.mul_packed_f32x2(silu_x, y)
        return dx, dy, swiglu_out


@dsl_user_op
def swiglu_oai(
    x: F32_or_F32x2, y: F32_or_F32x2, alpha: float = 1.702, *, loc=None, ip=None
) -> F32_or_F32x2:
    """The swiglu variant used in gpt-oss, which has a scaling factor on x and bias of 1 to y.
    https://github.com/openai/gpt-oss/blob/7be9334950053a888e24887a57dac797a17d6e00/gpt_oss/torch/model.py#L249
    x * sigmoid(alpha * x) * (y + 1)
    Compile down to FMUL, FMUL, TANH, FFMA, FFMA
    """
    # Compute sigmoid(alpha * x) using tanh: sigmoid(z) = 0.5 * (1 + tanh(z/2))
    if const_expr(not isinstance(x, tuple)):
        x_half = 0.5 * x
        # silu_x = x_half * cute.math.tanh(alpha * x_half, fastmath=True) + x_half
        silu_x = x_half * tanh(alpha * x_half) + x_half
        return silu_x * y + silu_x
    else:
        x_half = utils.mul_packed_f32x2((0.5, 0.5), x)
        alpha_x_half = utils.mul_packed_f32x2((alpha, alpha), x_half)
        tanh_alpha_x_half = (tanh(alpha_x_half[0]), tanh(alpha_x_half[1]))
        silu_x = utils.fma_packed_f32x2(x_half, tanh_alpha_x_half, x_half)
        return utils.fma_packed_f32x2(silu_x, y, silu_x)


@dsl_user_op
def dswiglu_oai(
    x: F32_or_F32x2,
    y: F32_or_F32x2,
    d_out: F32_or_F32x2,
    alpha: float = 1.702,
    *,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    """
    Swiglu OAI backward pass: computes gradients w.r.t. x and y
    Given: swiglu_oai_out = x * sigmoid(alpha * x) * (y + 1), and d_out = grad w.r.t. swiglu_oai_out
    Returns: (dx, dy, swiglu_oai_out)

    Derivative of x * sigmoid(alpha * x) w.r.t. x:
    d/dx[x * sigmoid(alpha * x)] = sigmoid(alpha * x) + alpha * x * sigmoid(alpha * x) * (1 - sigmoid(alpha * x))
    """
    if const_expr(not isinstance(x, tuple)):
        # Compute sigmoid(alpha * x) using tanh: sigmoid(z) = 0.5 * (1 + tanh(z/2))
        alpha_x_half = (0.5 * alpha) * x  # FMUL
        # MUFU.TANH, then FFMA
        # sigmoid_alpha_x = 0.5 + 0.5 * cute.math.tanh(alpha_x_half, fastmath=True)
        sigmoid_alpha_x = 0.5 + 0.5 * tanh(alpha_x_half)
        silu_x = x * sigmoid_alpha_x  # FMUL
        silu_x_d_out = silu_x * d_out  # FMUL
        # FFMA, FFMA, FMUL
        d_silu_x_d_out = (sigmoid_alpha_x + alpha * (silu_x - silu_x * sigmoid_alpha_x)) * d_out
        dx = d_silu_x_d_out * y + d_silu_x_d_out  # FFMA, instead of multiply by y + 1
        dy = silu_x_d_out
        swiglu_out = silu_x * y + silu_x  # FFMA, instead of multiply by y + 1
        # Overall it's 1 MUFU.TANH, 4 FMUL, 5 FFMA
        return dx, dy, swiglu_out
    else:
        # Compute sigmoid(alpha * x)
        alpha_x_half = utils.mul_packed_f32x2(((0.5 * alpha), (0.5 * alpha)), x)
        tanh_alpha_x_half = (tanh(alpha_x_half[0]), tanh(alpha_x_half[1]))
        sigmoid_alpha_x = utils.fma_packed_f32x2(tanh_alpha_x_half, (0.5, 0.5), (0.5, 0.5))
        silu_x = utils.mul_packed_f32x2(x, sigmoid_alpha_x)
        silu_x_d_out = utils.mul_packed_f32x2(silu_x, d_out)
        # d_silu_x_d_out = (sigmoid_alpha_x + alpha * (silu_x - silu_x * sigmoid_alpha_x)) * d_out
        silu_x_minus_product = utils.fma_packed_f32x2(
            silu_x, (-sigmoid_alpha_x[0], -sigmoid_alpha_x[1]), silu_x
        )
        sigmoid_plus_alpha_diff = utils.fma_packed_f32x2(
            (alpha, alpha), silu_x_minus_product, sigmoid_alpha_x
        )
        d_silu_x_d_out = utils.mul_packed_f32x2(sigmoid_plus_alpha_diff, d_out)
        dx = utils.fma_packed_f32x2(d_silu_x_d_out, y, d_silu_x_d_out)
        dy = silu_x_d_out
        swiglu_out = utils.fma_packed_f32x2(silu_x, y, silu_x)
        return dx, dy, swiglu_out


@dsl_user_op
def glu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """GLU: Gated Linear Unit
    glu(x, y) = sigmoid(x) * y
    Using tanh to compute sigmoid: sigmoid(x) = 0.5 * (1 + tanh(x/2))
    """
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)  # FMUL, MUFU.TANH, then FFMA
        return sigmoid_x * y  # FMUL
    else:
        sigmoid_x = sigmoid(x)
        return utils.mul_packed_f32x2(sigmoid_x, y)


@dsl_user_op
def dglu(
    x: F32_or_F32x2, y: F32_or_F32x2, d_out: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    """
    GLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: glu_out = sigmoid(x) * y, and d_out = grad w.r.t. glu_out
    Returns: (dx, dy, glu_out) where:
    - dx = d_out * y * sigmoid(x) * (1 - sigmoid(x))
    - dy = d_out * sigmoid(x)
    - glu_out = sigmoid(x) * y
    """
    if const_expr(not isinstance(x, tuple)):
        # Compute sigmoid(x) using tanh: sigmoid(x) = 0.5 * (1 + tanh(x/2))
        sigmoid_x = sigmoid(x)  # FMUL, MUFU.TANH, then FFMA
        sigmoid_x_d_out = sigmoid_x * d_out  # FMUL
        glu_out = sigmoid_x * y  # FMUL
        # dx = y * sigmoid(x) * (1 - sigmoid(x)) * d_out
        #    = y * (1 - sigmoid(x)) * sigmoid_x_d_out
        #    = (y - y * sigmoid(x)) * sigmoid_x_d_out
        #    = (y - glu_out) * sigmoid_x_d_out
        dx = (y - glu_out) * sigmoid_x_d_out  # FADD, FMUL
        dy = sigmoid_x_d_out
        # Total: 1 MUFU.TANH, 4 FMUL, 1 FADD, 1 FFMA
        return dx, dy, glu_out
    else:
        sigmoid_x = sigmoid(x)
        sigmoid_x_d_out = utils.mul_packed_f32x2(sigmoid_x, d_out)
        glu_out = utils.mul_packed_f32x2(sigmoid_x, y)
        # dx = (y - glu_out) * sigmoid_x_d_out
        y_minus_glu_out = utils.sub_packed_f32x2(y, glu_out)
        dx = utils.mul_packed_f32x2(y_minus_glu_out, sigmoid_x_d_out)
        dy = sigmoid_x_d_out
        return dx, dy, glu_out


@dsl_user_op
def reglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """ReGLU: ReLU Gated Linear Unit
    reglu(x, y) = relu(x) * y = max(x, 0) * y
    """
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0)) * y
    else:
        relu_x = relu(x)
        return utils.mul_packed_f32x2(relu_x, y)


@cute.jit
@dsl_user_op
def dreglu(
    x: F32_or_F32x2, y: F32_or_F32x2, d_out: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    """
    ReGLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: reglu_out = relu(x) * y, and d_out = grad w.r.t. reglu_out
    Returns: (dx, dy, reglu_out) where:
    - dx = d_out * y if x > 0, else 0
    - dy = d_out * relu(x)
    - reglu_out = relu(x) * y
    """
    if const_expr(not isinstance(x, tuple)):
        x_pos = cutlass.Boolean(x > 0)
        relu_x = cute.arch.fmax(x, Float32(0.0))
        dx = (d_out * y) if x_pos else Float32(0.0)
        dy = d_out * relu_x
        reglu_out = relu_x * y
        return dx, dy, reglu_out
    else:
        x0_pos = cutlass.Boolean(x[0] > 0)
        x1_pos = cutlass.Boolean(x[1] > 0)
        relu_x = relu(x)
        d_out_y = utils.mul_packed_f32x2(d_out, y)
        dx = (
            (d_out_y[0] if x0_pos else Float32(0.0)),
            (d_out_y[1] if x1_pos else Float32(0.0)),
        )
        dy = utils.mul_packed_f32x2(d_out, relu_x)
        reglu_out = utils.mul_packed_f32x2(relu_x, y)
        return dx, dy, reglu_out


@dsl_user_op
def geglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """GeGLU: GELU Gated Linear Unit
    geglu(x, y) = gelu(x) * y
    Uses the tanh approximation of GELU
    """
    if const_expr(not isinstance(x, tuple)):
        return gelu_tanh_approx(x) * y
    else:
        return utils.mul_packed_f32x2(gelu_tanh_approx(x), y)


@dsl_user_op
def dgeglu(
    x: F32_or_F32x2, y: F32_or_F32x2, d_out: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    """
    GeGLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: geglu_out = gelu(x) * y, and d_out = grad w.r.t. geglu_out
    Returns: (dx, dy, geglu_out) where:
    - dx = d_out * y * d_gelu(x)
    - dy = d_out * gelu(x)
    - geglu_out = gelu(x) * y
    """
    if const_expr(not isinstance(x, tuple)):
        # Reuse dgelu_tanh_approx to compute d_gelu(x) * d_out and gelu(x)
        dgelu_x_d_out, gelu_x = dgelu_tanh_approx(x, d_out)
        # Compute gradients for geglu
        dx = dgelu_x_d_out * y
        dy = gelu_x * d_out
        geglu_out = gelu_x * y
        return dx, dy, geglu_out
    else:
        # Reuse dgelu_tanh_approx to compute d_gelu(x) * d_out and gelu(x)
        dgelu_x_d_out, gelu_x = dgelu_tanh_approx(x, d_out)
        # Compute gradients for geglu
        dx = utils.mul_packed_f32x2(dgelu_x_d_out, y)
        dy = utils.mul_packed_f32x2(gelu_x, d_out)
        geglu_out = utils.mul_packed_f32x2(gelu_x, y)
        return dx, dy, geglu_out
