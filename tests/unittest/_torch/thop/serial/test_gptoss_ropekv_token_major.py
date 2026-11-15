# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bitwise tests for TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR, the token-major context-phase RoPE + FP8 Q + paged KV write.

Each case runs the serving TRTLLM attention backend on a context batch in one process with the V2 preprocessing
kernel (applyBiasRopeUpdateKVCacheV2), with each variant (1-3) of the token-major kernel, switched through the
library's tllm_gptoss_ropekv_token_major_override(), and once more with the V2 kernel (a determinism control).
It compares bit for bit:
  - the FP8 Q that the preprocessing hands to the context FMHA (captured at thop.trtllm_gen_context_preprocess),
    and the FMHA scales it writes;
  - every byte of the layer's paged K/V pool (filled with random FP8 values first, so cached prefixes are real);
  - the attention output.
Layouts: GPT-OSS-120B per rank (64 Q / 8 KV heads, head size 64, YaRN RoPE fused into attention, attention sinks,
FP8 KV cache with 32-token pages) with single-token, odd, chunked (cached prefix), sliding-window and non-unit KV
scale batches, and the EAGLE3 drafter's context call (Llama-3 RoPE on its own layer of the same KV pool, no sinks).
A generation-phase forward must keep the V2 kernel. Kernel launches are counted with torch.profiler.

Run as a script, for example:
    python3 test_gptoss_ropekv_token_major.py cases [large]    JSON line per case (hashes, equality, launches)
    python3 test_gptoss_ropekv_token_major.py env-check        the TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR=1 path, child process
"""

import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile

import torch

H, HKV, D, P = 64, 8, 64, 32
FLAG = "TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR"
V2_KERNEL = "applyBiasRopeUpdateKVCacheV2"
VARIANTS = (1, 2, 3)  # the flag's values: occupancy/ILP variants of the same kernel
TM_KERNEL = "ropeKvTokenMajorKernel"
MAX_SEQ = 65536 + 256

# name: (layer, new tokens per request, cached tokens per request, attention window, KV scale)
CASES = {
    "target-1tok": ("target", [1], [0], None, None),
    "target-odd": ("target", [7, 33, 129], [0, 0, 0], None, None),
    "target-chunked": ("target", [150, 41, 96], [200, 0, 37], None, None),
    "target-swa128": ("target", [300, 520], [0, 64], 128, None),
    "target-kvscale": ("target", [64, 65], [0, 0], None, 0.875),
    "drafter": ("drafter", [128, 77], [0, 51], None, None),
}
# A large prefill step: 98,304 tokens of ~5.2K-token prompts, one of them a chunk continuation.
LARGE_CASES = {
    "target-98304": ("target", [5170] * 18 + [5244], [0] * 18 + [2000], 128, None),
    "drafter-98304": ("drafter", [5170] * 18 + [5244], [0] * 18 + [2000], None, None),
}


def log(*a):
    print(*a, flush=True)


def lib_override():
    """The library's tllm_gptoss_ropekv_token_major_override, or None if this libtensorrt_llm.so has none."""
    import tensorrt_llm

    path = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs", "libtensorrt_llm.so")
    lib = ctypes.CDLL(path)
    try:
        fn = lib.tllm_gptoss_ropekv_token_major_override
    except AttributeError:
        return None
    fn.argtypes = [ctypes.c_int]
    fn.restype = None
    return fn


class PreprocessCapture:
    """Copies the FP8 Q and the FMHA scales out of every thop.trtllm_gen_context_preprocess call."""

    def __init__(self):
        self.calls = []
        self.installed = False

    def install(self):
        try:
            from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fig
        except ImportError:
            return False
        if getattr(fig.thop, "_ropekv_capture", None) is self:
            return True
        real, cap = fig.thop, self

        class _Thop:
            _ropekv_capture = cap

            def __getattr__(self, name):
                return getattr(real, name)

            def trtllm_gen_context_preprocess(self, *args, **kwargs):
                r = real.trtllm_gen_context_preprocess(*args, **kwargs)
                q = r[0]
                n = q.shape[0] * q.shape[1] * q.shape[2]
                cap.calls.append(
                    dict(
                        q=q.view(torch.uint8).flatten()[:n].clone(),
                        bmm1=None if r[4] is None else r[4].clone(),
                        bmm2=None if r[5] is None else r[5].clone(),
                    )
                )
                return r

        fig.thop = _Thop()
        self.installed = True
        return True


def make_attention(layer):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.utils import create_attention
    from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    if layer == "target":  # GPT-OSS-120B: YaRN, theta 150000, factor 32 over 4096 positions
        pos = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=RopeParams(
                dim=D,
                theta=150000.0,
                scale_type=RotaryScalingType.yarn,
                scale=32.0,
                max_positions=131072,
                original_max_positions=4096,
                beta_fast=32,
                beta_slow=1,
                duplicate_data=False,
            ),
            is_neox=True,
        )
        layer_idx = 0
    else:  # its EAGLE3 drafter (LlamaForCausalLMEagle3, 64/8 heads, head size 64): Llama-3 RoPE, theta 500000
        pos = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams(
                dim=D,
                theta=500000.0,
                scale_type=RotaryScalingType.llama3,
                scale=8.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                max_positions=131072,
                original_max_positions=8192,
            ),
            is_neox=True,
        )
        layer_idx = 1
    return create_attention(
        "TRTLLM",
        layer_idx=layer_idx,
        num_heads=H,
        head_dim=D,
        num_kv_heads=HKV,
        pos_embd_params=pos,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        q_scaling=1.0,
    ), layer_idx


def make_manager(token_nums):
    import tensorrt_llm.bindings as tb
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    b = len(token_nums)
    mgr = KVCacheManager(
        KvCacheConfig(
            max_tokens=sum(math.ceil(L / P) * P for L in token_nums) + 2 * b * P,
            enable_block_reuse=False,
        ),
        tb.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=HKV,
        head_dim=D,
        tokens_per_block=P,
        max_seq_len=MAX_SEQ,
        max_batch_size=b,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tb.DataType.FP8,
        max_num_tokens=max(8192, sum(token_nums)),
    )
    rids = list(range(b))
    mgr.add_dummy_requests(rids, token_nums)
    return mgr, rids


def kernel_counts(fn):
    """{kernel name: launches} of one call of fn, from torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    prof.export_chrome_trace(path)
    events = json.load(open(path)).get("traceEvents", [])
    os.unlink(path)
    counts, us = {}, {}
    for e in events:
        if e.get("cat") in ("kernel", "Kernel"):
            name = e.get("name", "")
            counts[name] = counts.get(name, 0) + 1
            us[name] = us.get(name, 0.0) + float(e.get("dur", 0))
    return counts, us


def launches(counts, needle):
    return sum(n for name, n in counts.items() if needle in name)


def kernel_us(us, counts, needle):
    n = launches(counts, needle)
    return sum(t for name, t in us.items() if needle in name) / n if n else None


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class CaseRunner:
    """One case: KV manager, inputs, metadata and the forward; run(mode) returns the bytes to compare."""

    def __init__(self, name, spec, capture):
        from tensorrt_llm._torch.attention.backends.interface import (
            AttentionForwardArgs,
            PredefinedAttentionMask,
        )
        from tensorrt_llm._torch.metadata import KVCacheParams

        layer, seq_lens, cached, window, kv_scale = spec
        self.name, self.capture = name, capture
        self.attn, self.layer_idx = make_attention(layer)
        token_nums = [c + s for c, s in zip(cached, seq_lens)]
        self.mgr, rids = make_manager(token_nums)
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.kv = self.mgr.get_buffers(self.layer_idx, kv_layout="HND").view(torch.uint8)
        self.kv.copy_(
            (torch.randn(self.kv.shape, device="cuda", generator=g) * 0.5)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        self.kv_init = self.kv.clone()
        self.nnz = sum(seq_lens)
        self.qkv = torch.randn(
            self.nnz, (H + 2 * HKV) * D, device="cuda", dtype=torch.bfloat16, generator=g
        )
        sinks = (
            torch.randn(H, device="cuda", dtype=torch.float32, generator=g)
            if layer == "target"
            else None
        )
        scales = {}
        if kv_scale is not None:
            scales = dict(
                kv_scale_orig_quant=torch.tensor([kv_scale], device="cuda", dtype=torch.float32),
                kv_scale_quant_orig=torch.tensor(
                    [1.0 / kv_scale], device="cuda", dtype=torch.float32
                ),
            )
        self.fa = AttentionForwardArgs(
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_window_size=window,
            attention_sinks=sinks,
            **scales,
        )
        self.md = self.attn.Metadata(
            num_contexts=len(seq_lens),
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            max_num_requests=len(seq_lens),
            max_num_tokens=max(8192, self.nnz),
            kv_cache_manager=self.mgr,
            request_ids=rids,
            prompt_lens=token_nums,
            kv_layout="HND",
        )
        self.md.prepare()
        self.gen = (rids, token_nums)

    def forward(self):
        o = self.attn.forward(self.qkv, None, None, self.md, forward_args=self.fa)
        return o[0] if isinstance(o, tuple) else o

    def run(self, mode, override):
        if override is not None:
            override(mode)
        self.kv.copy_(self.kv_init)
        self.capture.calls.clear()
        out = self.forward()
        torch.cuda.synchronize()
        call = self.capture.calls[-1] if self.capture.calls else None
        res = dict(
            out=out[: self.nnz].contiguous().view(torch.uint8).clone(),
            kv=self.kv.clone(),
            q=None if call is None else call["q"],
            bmm1=None if call is None else call["bmm1"],
            bmm2=None if call is None else call["bmm2"],
        )
        self.kv.copy_(self.kv_init)
        counts, _ = kernel_counts(self.forward)
        res["v2"], res["tm"] = launches(counts, V2_KERNEL), launches(counts, TM_KERNEL)
        return res

    def generation_check(self, override):
        """A decode forward (1 token per request, the last allocated position) on the same requests: the V2 kernel
        in every mode, same bytes."""
        from tensorrt_llm._torch.metadata import KVCacheParams

        rids, token_nums = self.gen
        b = len(rids)
        md = self.attn.Metadata(
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[L - 1 for L in token_nums]
            ),
            seq_lens=torch.tensor([1] * b, dtype=torch.int),
            max_num_requests=b,
            max_num_tokens=8192,
            kv_cache_manager=self.mgr,
            request_ids=rids,
            prompt_lens=token_nums,
            kv_layout="HND",
        )
        md.prepare()
        g = torch.Generator(device="cuda").manual_seed(7)
        qkv = torch.randn(b, (H + 2 * HKV) * D, device="cuda", dtype=torch.bfloat16, generator=g)
        rows = {}
        for mode in (0,) + VARIANTS:
            override(mode)
            self.kv.copy_(self.kv_init)

            def fwd():
                o = self.attn.forward(qkv, None, None, md, forward_args=self.fa)
                return o[0] if isinstance(o, tuple) else o

            out = fwd()
            torch.cuda.synchronize()
            kv = self.kv.clone()
            self.kv.copy_(self.kv_init)
            counts, _ = kernel_counts(fwd)
            rows[mode] = dict(
                out=out[:b].contiguous().view(torch.uint8).clone(),
                kv=kv,
                v2=launches(counts, V2_KERNEL),
                tm=launches(counts, TM_KERNEL),
            )
        override(-1)
        return dict(
            equal=all(
                torch.equal(rows[0]["out"], r["out"]) and torch.equal(rows[0]["kv"], r["kv"])
                for r in rows.values()
            ),
            v2=[r["v2"] for r in rows.values()],
            tm=[r["tm"] for r in rows.values()],
        )

    def close(self):
        self.mgr.shutdown()


def compare(a, b):
    """Field -> equal (None if the field was not captured in either run)."""
    out = {}
    for k in ("q", "bmm1", "bmm2", "kv", "out"):
        if a[k] is None or b[k] is None:
            out[k] = None if a[k] is None and b[k] is None else False
        else:
            out[k] = a[k].shape == b[k].shape and torch.equal(a[k], b[k])
    return out


def run_cases(names, override, capture, with_generation=True, variants=VARIANTS):
    """One JSON-able row per case: hashes of the V2 run, equality V2 vs each token-major variant and V2 vs V2,
    launches (V2, token-major) per mode."""
    rows = []
    for name in names:
        spec = {**CASES, **LARGE_CASES}[name]
        c = CaseRunner(name, spec, capture)
        try:
            a = c.run(0, override)
            tm = {v: c.run(v, override) for v in variants} if override is not None else {}
            a2 = c.run(0, override)
            row = dict(
                case=name,
                tokens=c.nnz,
                layer=spec[0],
                q_captured=a["q"] is not None,
                hashes={k: (sha(a[k]) if a[k] is not None else None) for k in ("q", "kv", "out")},
                determinism=compare(a, a2),
                launches_v2_mode=[a["v2"], a["tm"]],
            )
            if tm:
                row["variants"] = {
                    str(v): dict(equal=compare(a, b), launches=[b["v2"], b["tm"]])
                    for v, b in tm.items()
                }
                if with_generation and spec[0] == "target" and c.nnz < 4096:
                    row["generation"] = c.generation_check(override)
            rows.append(row)
        finally:
            if override is not None:
                override(-1)
            c.close()
            torch.cuda.empty_cache()
    return rows


def row_passes(row, variants=VARIANTS):
    """Every token-major variant equals V2 on every captured field and ran once (no V2 launch); V2 is
    deterministic; a decode forward keeps the V2 kernel."""
    det = row["determinism"]
    det_ok = det.get("kv") and det.get("out") and det.get("q") is not False
    vs = row.get("variants", {})
    var_ok = len(vs) == len(variants)
    for v in variants:
        r = vs.get(str(v))
        eq = r["equal"] if r else {}
        var_ok = (
            var_ok and bool(r) and all(x is not False for x in eq.values()) and bool(eq.get("kv"))
        )
        var_ok = var_ok and bool(eq.get("out")) and r["launches"] == [0, 1]
    gen = row.get("generation")
    gen_ok = gen is None or (gen["equal"] and not any(gen["tm"]) and min(gen["v2"]) >= 1)
    return bool(
        det_ok
        and var_ok
        and row["launches_v2_mode"][1] == 0
        and row["launches_v2_mode"][0] >= 1
        and gen_ok
    )


def env_check():
    """Child process with TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR=1 and no override: the stderr line and the kernel."""
    code = (
        "import json, sys; sys.path.insert(0, %r); import test_gptoss_ropekv_token_major as t; "
        "cap = t.PreprocessCapture(); cap.install(); "
        "c = t.CaseRunner('target-odd', t.CASES['target-odd'], cap); r = c.run(1, None); c.close(); "
        "print('ENVCHECK ' + json.dumps(dict(v2=r['v2'], tm=r['tm'])))"
        % os.path.dirname(os.path.abspath(__file__))
    )
    res = {}
    for value in ("1", None):
        env = {k: v for k, v in os.environ.items() if k != FLAG}
        if value is not None:
            env[FLAG] = value
        p = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=900
        )
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("ENVCHECK ")]
        res["on" if value else "off"] = dict(
            exit=p.returncode,
            launches=json.loads(line[-1][9:]) if line else None,
            active_lines=sum(f"{FLAG}=1 active" in ln for ln in p.stderr.splitlines()),
            kept_lines=sum(
                f"{FLAG}=1: a context-phase launch keeps" in ln for ln in p.stderr.splitlines()
            ),
            stderr_tail=p.stderr.strip().splitlines()[-3:] if p.returncode else [],
        )
    on, off = res["on"], res["off"]
    res["pass"] = bool(
        on["exit"] == 0
        and off["exit"] == 0
        and on["launches"] == dict(v2=0, tm=1)
        and off["launches"] == dict(v2=1, tm=0)
        and on["active_lines"] == 1
        and on["kept_lines"] == 0
        and off["active_lines"] == 0
    )
    return res


def _require_gpu():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() < 100:
        pytest.skip("FP8 context FMHA through trtllm-gen requires SM100 or newer")
    override = lib_override()
    if override is None:
        pytest.skip("libtensorrt_llm.so has no tllm_gptoss_ropekv_token_major_override")
    return override


def test_token_major_matches_v2_bitwise():
    override = _require_gpu()
    capture = PreprocessCapture()
    capture.install()
    names = list(CASES) + (list(LARGE_CASES) if os.environ.get("ROPEKV_TEST_LARGE") == "1" else [])
    rows = run_cases(names, override, capture)
    bad = [r for r in rows if not row_passes(r)]
    assert not bad, json.dumps(bad, indent=1)


def test_environment_variable_path():
    _require_gpu()
    res = env_check()
    assert res["pass"], json.dumps(res, indent=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "cases"
    if what == "env-check":
        r = env_check()
        log("ENVCHECK_RESULT " + json.dumps(r))
        sys.exit(0 if r["pass"] else 3)
    ov = lib_override()
    cap = PreprocessCapture()
    log(f"capture installed: {cap.install()}; override symbol: {ov is not None}")
    names = list(CASES) + (list(LARGE_CASES) if "large" in sys.argv[2:] else [])
    for row in run_cases(names, ov, cap):
        if ov is not None:
            row["pass"] = row_passes(row)
        log("CASE " + json.dumps(row))
