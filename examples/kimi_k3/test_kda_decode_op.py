"""Parity test: in-tree KDA decode thop op (optimized path) vs FLA reference."""
import copy
import os

import torch

assert os.environ.get("KIMI_KDA_OPTIMIZED_KERNEL_DIR"), \
    "set KIMI_KDA_OPTIMIZED_KERNEL_DIR (needed for kernel_path=OPTIMIZED)"

from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import (
    KimiKDACachedState, KimiKDALinearAttention)

torch.manual_seed(0)
dev = "cuda"
B, H, D, W = 32, 96, 128, 4

common = dict(hidden_size=7168, num_heads=H, head_dim=D, conv_kernel_size=W,
              use_full_rank_gate=True, gate_lower_bound=-5.0,
              rms_norm_eps=1e-5, dtype=torch.bfloat16)

opt = KimiKDALinearAttention(**common).to(dev)
assert opt.kernel_path == "optimized", opt.kernel_path
ref = KimiKDALinearAttention(**common, force_use_fallback_kernel=True).to(dev)
ref.load_state_dict(opt.state_dict())
assert ref.kernel_path == "fla", ref.kernel_path

x = torch.randn(B, 1, 7168, dtype=torch.bfloat16, device=dev) * 0.05


def mk_cache():
    return KimiKDACachedState(
        conv_state_q=torch.randn(B, H * D, W, dtype=torch.bfloat16, device=dev) * 0.05,
        conv_state_k=torch.randn(B, H * D, W, dtype=torch.bfloat16, device=dev) * 0.05,
        conv_state_v=torch.randn(B, H * D, W, dtype=torch.bfloat16, device=dev) * 0.05,
        recurrent_state=torch.randn(B, H, D, D, dtype=torch.float32, device=dev) * 0.05,
    )


c0 = mk_cache()
out_opt, cache_opt = opt.forward_decode(x, copy.deepcopy(c0))
out_ref, cache_ref = ref.forward_decode(x, copy.deepcopy(c0))


def rep(name, a, b):
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).norm() / (b.norm() + 1e-12)).item()
    print(f"{name}: cos={cos:.6f} rel_l2={rel:.3e} max_abs={(a - b).abs().max().item():.3e}")
    return cos, rel


ok = True
for name, a, b in [("output", out_opt, out_ref),
                   ("recurrent_state", cache_opt.recurrent_state, cache_ref.recurrent_state),
                   ("conv_state_q", cache_opt.conv_state_q, cache_ref.conv_state_q),
                   ("conv_state_k", cache_opt.conv_state_k, cache_ref.conv_state_k),
                   ("conv_state_v", cache_opt.conv_state_v, cache_ref.conv_state_v)]:
    c, r = rep(name, a, b)
    ok &= c > 0.999 and r < 3e-2
print("decode source (optimized):", opt.decode_kernel_source)
print("KDA-DECODE-OP-PARITY:", "PASS" if ok else "FAIL")
