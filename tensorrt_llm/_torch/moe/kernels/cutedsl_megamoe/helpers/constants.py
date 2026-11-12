"""Specification-defined numeric constants shared across kernel components."""


Log2E = 1.4426950408889634
Fp32Max = 3.40282346638528859812e38

Nvfp4E2M1Max = 6.0
Fp8E4M3FNMax = 448.0
Fp8E5M2Max = 57344.0

Nvfp4E2M1RcpLimit = 1.0 / Nvfp4E2M1Max
Fp8E4M3RcpLimit = 1.0 / Fp8E4M3FNMax
Fp8E5M2RcpLimit = 1.0 / Fp8E5M2Max


__all__ = [
    "Fp32Max",
    "Fp8E4M3FNMax",
    "Fp8E4M3RcpLimit",
    "Fp8E5M2Max",
    "Fp8E5M2RcpLimit",
    "Log2E",
    "Nvfp4E2M1Max",
    "Nvfp4E2M1RcpLimit",
]
