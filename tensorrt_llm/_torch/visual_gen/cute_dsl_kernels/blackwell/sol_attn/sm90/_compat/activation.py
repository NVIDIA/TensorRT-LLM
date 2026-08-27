import cutlass.cute as cute


def sub_packed_f32x2(a, b):
    return cute.arch.add_packed_f32x2(a, (-b[0], -b[1]))
