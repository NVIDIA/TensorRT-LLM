# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warm the MiniMax-M3 Triton sparse decode for every local batch before serving (sparse-decode prewarm).

Why. Triton builds one specialization of _gqa_sparse_decode_kernel (or the pair kernel) and
_merge_topk_attn_out_kernel per launch policy (split-K, query pairing, warps, stages, merge warps), per total_q % 16
and per 16-byte alignment of every pointer argument. CUDA-graph capture builds the keys of the captured batch sizes
with every slice starting at row 0. Eager mixed (context + generation) steps reach two more kinds of key:
  * local batches that are not CUDA-graph sizes, whose policies (and total_q % 16) no captured graph used;
  * pointers that start past the context rows: msa_decode.MsaDecodeFmha.run_generation passes
    msa_seq_lens_cuda[row_first:row_last] and msa_block_table[row_first:row_last] (row_first = context requests in
    the step), so the int32 seq_lens pointer is not 16-byte aligned whenever row_first % 4 != 0 (likewise q / output /
    top-k when a token row is not a multiple of 16 bytes, and the block table when its row is not).
Each such key is a Triton compile (0.2-0.3 s) that stalls an eager mixed step the first time serving reaches it.

What. hook.py calls begin() when the decoder runner's MSA prewarm runs (start of warmup) and finish() when
DecoderWarmup.run returns:
  begin    for an MSA runner only: wraps triton_sparse_decode.minimax_m3_sparse_attn_decode (the module attribute that
           msa_decode imports at every call) with a recorder that notes each call's shape class (dtypes, head geometry,
           strides, decode query length) and keeps references to that class's persistent arguments only (paged K/V
           views, block-scale views, the per-layer global-scale buffers). Every call goes through unchanged.
  finish   restores the module attribute (serving calls the function itself, no wrapper), then for every recorded
           class and every decode query length the runner can produce (the recorded one, 1 + max draft length, and
           1 + each dynamic draft length; DSpark K=3 gives 4, K=7 gives 8) calls minimax_m3_sparse_attn_decode for
           every local batch b = 1..max_batch_size and every distinct pointer-alignment signature an eager step with
           b generation rows can produce (plan_variants: the pure generation step, and mixed steps with r = 1..4
           context rows (r + b <= max_batch_size) and t = r..r+15 context tokens before them), on fresh q / top-k /
           output buffers laid out like the recorded ones and sliced at token t, and on seq_lens / block_table slices
           starting at row r of buffers with the recorded row stride. The function itself picks the launch policy, so
           the keys are serving's. seq_lens are 0: the kernels' CUDA-graph padding path (kv_len clamps to 0), so no
           KV page, top-k entry or block-table entry is read; K/V/scales are passed only for their dtype, strides and
           alignment. Outputs are discarded. Then one line per rank:
           "[M3AX msa-prewarm] sparse-decode prewarm done rank=R ... new_specializations=N ...".
           It also sets Triton's jit_post_compile_hook (chained to any previous hook; Triton calls it only when a
           kernel is compiled or loaded into the process, never on a cache hit) to log, after warmup, one line per
           new Triton specialization of any kernel: "[M3AX msa-prewarm] triton serve compile kernel=K
           sparse_decode=0|1 n=N ..." (warning level when sparse_decode=1: a key this prewarm missed).
Serving afterwards: the same functions and routing; the in-process Triton caches and the on-disk cache already hold
the keys. Numerics are unchanged (the same kernels, compiled earlier).

Best effort: any failure is logged and the stock lazy Triton JIT applies unchanged.
No torch / tensorrt_llm / triton import at module import time.
"""

import functools
import importlib
import time

TAG = "[M3AX msa-prewarm]"
TSD = "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode"
FUNC = "minimax_m3_sparse_attn_decode"
KERNELS = (
    "_gqa_sparse_decode_kernel",
    "_gqa_sparse_decode_pair_kernel",
    "_merge_topk_attn_out_kernel",
)
PERSIST = ("kv_scale", "k_block_scale", "v_block_scale", "k_global_scale", "v_global_scale")
POSITIONAL = ("q", "k_paged", "v_paged", "topk_idx", "block_table", "seq_lens")
MAX_CTX_ROWS = (
    4  # context-row offsets r = 1..4 cover every residue of r mod 4 (int32 rows: 4 per 16 bytes)
)
TOKEN_RESIDUES = (
    16  # context-token offsets t = r..r+15 cover every residue of t mod 16 (1-byte rows at worst)
)
MAX_BATCH_CAP = 1024
RECORD_CAP = 100000  # recorder stops looking at calls after this many (it is meant for warmup only)

_S = {}  # one warmup's recorder state (reset by begin)
_H = {
    "hook": False,
    "serve_compiles": 0,
    "serve_sd_compiles": 0,
}  # process-wide serve-time compile observer


def _reset():
    _S.clear()
    _S.update(
        mod=None,
        orig=None,
        recording=False,
        classes={},
        seen=0,
        recorded=0,
        skipped=0,
        skip_reasons={},
        last=None,
    )


_reset()


def enabled():
    return True


def log(msg, warn=False):
    try:
        from tensorrt_llm.logger import logger

        (logger.warning if warn else logger.info)("%s %s" % (TAG, msg))
    except Exception:  # noqa: BLE001
        print("%s %s" % (TAG, msg), flush=True)


# ----------------------------------------------------------------------------------------------- pure planning (CPU)
def plan_variants(
    b, max_batch, q_row_bytes, out_row_bytes, topk_row_bytes, seq_row_bytes, bt_row_bytes
):
    """[(signature, r, t)]: one representative (context-row offset r, context-token offset t) per alignment signature.

    A signature is whether the q, output, top-k, seq_lens and block_table pointers are 16-byte aligned in an eager
    step with b generation rows. (0, 0) is the pure-generation step (every slice at row 0, as in a captured graph); a
    mixed step has r >= 1 context requests (r + b <= max_batch) and t >= r context tokens before the generation rows.
    Row 0 of every buffer is assumed 16-byte aligned (torch allocations are; the recorder logs the recorded pointers'
    alignment).
    """

    def al(nbytes):
        return nbytes % 16 == 0

    reps = {}
    cands = [(0, 0)] + [
        (r, t)
        for r in range(1, MAX_CTX_ROWS + 1)
        if r + b <= max_batch
        for t in range(r, r + TOKEN_RESIDUES)
    ]
    for r, t in cands:
        sig = (
            al(t * q_row_bytes),
            al(t * out_row_bytes),
            al(t * topk_row_bytes),
            al(r * seq_row_bytes),
            al(r * bt_row_bytes),
        )
        reps.setdefault(sig, (r, t))
    return [(sig, r, t) for sig, (r, t) in reps.items()]


def plan(max_batch, row_bytes):
    """Every (b, signature, r, t) the warm runs for one class.

    row_bytes = (q, out, topk, seq, bt) row strides in bytes.
    """
    out = []
    for b in range(1, max_batch + 1):
        out += [(b, sig, r, t) for sig, r, t in plan_variants(b, max_batch, *row_bytes)]
    return out


def row_outermost(sizes, strides, rd):
    """True when the row dimension rd has a stride larger than the extent of every other dimension (rows never overlap
    and the stride does not depend on the row count), so a buffer with more rows can reuse the recorded strides."""
    ext = sum((s - 1) * st for d, (s, st) in enumerate(zip(sizes, strides)) if d != rd and s > 0)
    return all(st >= 0 for st in strides) and strides[rd] > ext


# ----------------------------------------------------------------------------------------------- recorder (warmup)
def _layout(t, rd):
    sizes, strides = tuple(int(x) for x in t.shape), tuple(int(x) for x in t.stride())
    return dict(
        dtype=t.dtype,
        sizes=sizes,
        strides=strides,
        rd=rd,
        esize=int(t.element_size()),
        outer=row_outermost(sizes, strides, rd),
        align=int(t.data_ptr()) % 16,
    )


def _layout_key(lay):
    rest = tuple(s for d, s in enumerate(lay["sizes"]) if d != lay["rd"])
    st = (
        lay["strides"]
        if lay["outer"]
        else tuple(sorted(range(len(lay["strides"])), key=lambda d: -lay["strides"][d]))
    )
    return (str(lay["dtype"]), rest, lay["outer"], st)


def _skip(reason):
    _S["skipped"] += 1
    _S["skip_reasons"][reason] = _S["skip_reasons"].get(reason, 0) + 1


def _record(args, kwargs):
    import torch

    if len(args) != len(POSITIONAL):
        return _skip("positional")
    if kwargs.get("num_topk_chunks") is not None or kwargs.get("policy_override") is not None:
        return _skip("override")
    q, k, v, topk, bt, seq = args
    out = kwargs.get("output")
    persist = {n: kwargs.get(n) for n in PERSIST}
    tensors = [q, k, v, topk, bt, seq, out] + [p for p in persist.values() if p is not None]
    if not all(isinstance(x, torch.Tensor) and x.is_cuda for x in tensors):
        return _skip("not-cuda-tensor")
    if (
        q.dim() != 3
        or out is None
        or out.dim() != 3
        or topk.dim() != 3
        or bt.dim() != 2
        or seq.dim() != 1
    ):
        return _skip("rank")
    dql = int(kwargs.get("decode_query_len") or 0)
    total_q = int(q.shape[0])
    if dql <= 0 or total_q <= 0 or total_q % dql or int(seq.shape[0]) * dql != total_q:
        return _skip("span")
    lq, lo, lt = _layout(q, 0), _layout(out, 0), _layout(topk, 1)
    palign = tuple(
        int(x.data_ptr()) % 16 == 0 for x in [k, v] + [p for p in persist.values() if p is not None]
    )
    key = (
        _layout_key(lq),
        _layout_key(lo),
        _layout_key(lt),
        str(k.dtype),
        tuple(int(x) for x in k.shape[1:]),
        tuple(int(x) for x in k.stride()),
        str(v.dtype),
        tuple(int(x) for x in v.stride()),
        str(bt.dtype),
        int(bt.stride(0)),
        int(bt.shape[1]),
        str(seq.dtype),
        int(seq.stride(0)),
        dql,
        tuple(n for n, p in persist.items() if p is not None),
        palign,
        str(q.device),
    )
    c = _S["classes"].get(key)
    if c is None:
        c = _S["classes"][key] = dict(
            shape=key[:13] + key[14:],
            q=lq,
            out=lo,
            topk=lt,
            k=k,
            v=v,
            persist=persist,
            dql=dql,
            sm_scale=float(kwargs.get("sm_scale")),
            bt_dtype=bt.dtype,
            bt_stride0=int(bt.stride(0)),
            bt_width=int(bt.shape[1]),
            bt_esize=int(bt.element_size()),
            bt_align=int(bt.data_ptr()) % 16,
            seq_dtype=seq.dtype,
            seq_stride=int(seq.stride(0)),
            seq_esize=int(seq.element_size()),
            seq_align=int(seq.data_ptr()) % 16,
            device=q.device,
            capturing=bool(torch.cuda.is_current_stream_capturing()),
            calls=0,
            max_total_q=0,
        )
    c["calls"] += 1
    c["max_total_q"] = max(c["max_total_q"], total_q)
    _S["recorded"] += 1


def _make_recorder(orig):
    def recorder(*args, **kwargs):
        if _S["recording"] and _S["seen"] < RECORD_CAP:
            _S["seen"] += 1
            try:
                _record(args, kwargs)
            except Exception as exc:  # noqa: BLE001  (recording must never affect the call)
                _skip("error:%s" % type(exc).__name__)
        return orig(*args, **kwargs)

    functools.update_wrapper(recorder, orig)
    recorder._m3ax_sdwarm_recorder = True
    return recorder


def begin(runner, kvm):
    """Start recording sparse-decode call shapes (an MSA runner's warmup). Never raises."""
    try:
        if not enabled():
            return
        from . import core

        if core.geometry(runner, kvm) is None:
            return
        mod = importlib.import_module(TSD)
        cur = getattr(mod, FUNC)
        if getattr(cur, "_m3ax_sdwarm_recorder", False):
            _S["recording"] = True
            return
        _reset()
        _S.update(mod=mod, orig=cur, recording=True)
        setattr(mod, FUNC, _make_recorder(cur))
    except Exception as exc:  # noqa: BLE001
        log(
            "sparse-decode prewarm off (recorder not installed): %s: %s"
            % (type(exc).__name__, exc),
            warn=True,
        )


def _restore():
    mod, orig = _S.get("mod"), _S.get("orig")
    _S["recording"] = False
    if mod is not None and getattr(getattr(mod, FUNC, None), "_m3ax_sdwarm_recorder", False):
        setattr(mod, FUNC, orig)


# ----------------------------------------------------------------------------------------------- warm (end of warmup)
def _rows(lay, nrows, dev):
    """Zeroed tensor shaped like the recorded one with nrows rows: the recorded strides when the row dim is outermost
    (non-overlapping, row-count independent), else a dense row-major tensor."""
    import torch

    sizes = list(lay["sizes"])
    sizes[lay["rd"]] = nrows
    if lay["outer"]:
        strides = list(lay["strides"])
        n = 1 + sum((s - 1) * st for s, st in zip(sizes, strides) if s > 0)
        raw = torch.zeros(n * lay["esize"], dtype=torch.uint8, device=dev)
        return raw.view(lay["dtype"]).as_strided(sizes, strides)
    n = 1
    for s in sizes:
        n *= s
    raw = torch.zeros(n * lay["esize"], dtype=torch.uint8, device=dev)
    return raw.view(lay["dtype"]).view(sizes)


def _call(fn, c, b, r, t, dql):
    import torch

    dev = c["device"]
    tq = b * dql
    q = _rows(c["q"], t + tq, dev).narrow(0, t, tq)
    out = _rows(c["out"], t + tq, dev).narrow(0, t, tq)
    topk = _rows(c["topk"], t + tq, dev).narrow(1, t, tq)
    seq = torch.zeros((r + b) * c["seq_stride"], dtype=c["seq_dtype"], device=dev)[
        r * c["seq_stride"] :: c["seq_stride"]
    ]
    bt = torch.zeros((r + b, c["bt_stride0"]), dtype=c["bt_dtype"], device=dev)[r:, : c["bt_width"]]
    persist = {n: p for n, p in c["persist"].items() if p is not None}
    fn(
        q,
        c["k"],
        c["v"],
        topk,
        bt,
        seq,
        sm_scale=c["sm_scale"],
        output=out,
        decode_query_len=dql,
        **persist,
    )


def _jit_fn(k):
    for _ in range(4):
        if k is None or hasattr(k, "device_caches"):
            break
        k = getattr(k, "fn", None)
    return k


def cache_entries(mod):
    """{kernel: in-process Triton specializations} of the sparse-decode kernels (None where unreadable)."""
    out = {}
    for name in KERNELS:
        fn = _jit_fn(getattr(mod, name, None))
        try:
            n = 0
            for v in list(getattr(fn, "device_caches").values()):
                if isinstance(v, tuple) and v and isinstance(v[0], dict):
                    n += len(v[0])
            out[name] = n
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def _max_batch(runner):
    try:
        mb = int(getattr(getattr(runner, "_config", None), "max_batch_size", 0) or 0)
    except Exception:  # noqa: BLE001
        mb = 0
    src = "config"
    if mb <= 0:
        mb = max([c["max_total_q"] // c["dql"] for c in _S["classes"].values()] or [1])
        src = "recorded"
    return max(1, min(mb, MAX_BATCH_CAP)), src


def decode_query_lens(runner):
    """Decode query lengths the runner's generation steps can use besides the recorded ones: 1 + the max draft
    length and 1 + each dynamic draft length under speculative decoding (DSpark K=3: 4; K=7: 8), else none."""
    out = set()
    try:
        cfg = runner._config
        if cfg.is_spec_decode:
            out.add(1 + int(cfg.original_max_draft_len))
            from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import (
                compute_dynamic_draft_len_mapping,
            )

            mapping = compute_dynamic_draft_len_mapping(cfg) or {}
            out.update(1 + int(d) for d in mapping.values())
    except Exception:  # noqa: BLE001  (the recorded lengths still apply)
        pass
    return out


def _warm(runner):
    import torch

    t0 = time.time()
    rank = int(getattr(getattr(runner, "mapping", None), "rank", 0) or 0)
    mod, fn = _S["mod"], _S["orig"]
    mb, src = _max_batch(runner)
    before = cache_entries(mod)
    calls = failed = 0
    errs = []
    extra_dqls = decode_query_lens(runner)
    done = set()
    for c in list(_S["classes"].values()):
        dqls = [d for d in sorted({c["dql"]} | extra_dqls) if (c["shape"], d) not in done]
        done.update((c["shape"], d) for d in dqls)
        rb = (
            c["q"]["strides"][0] * c["q"]["esize"],
            c["out"]["strides"][0] * c["out"]["esize"],
            c["topk"]["strides"][1] * c["topk"]["esize"],
            c["seq_stride"] * c["seq_esize"],
            c["bt_stride0"] * c["bt_esize"],
        )
        if not (c["q"]["outer"] and c["out"]["outer"] and c["topk"]["outer"]):
            errs.append("class layout approximated (row dim not outermost)")
        cf = 0
        with torch.cuda.device(c["device"]):
            for dql, (b, _sig, r, t) in ((d, p) for d in dqls for p in plan(mb, rb)):
                try:
                    _call(fn, c, b, r, t, dql)
                    calls += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    cf += 1
                    errs.append(
                        "dql=%d b=%d r=%d t=%d %s: %s"
                        % (
                            dql,
                            b,
                            r,
                            t,
                            type(exc).__name__,
                            str(exc).splitlines()[0][:160] if str(exc) else "",
                        )
                    )
                    if cf >= 3:
                        errs.append("class abandoned after 3 failures")
                        break
            torch.cuda.synchronize()
    after = cache_entries(mod)
    new = {
        k: (after[k] - before[k]) if after[k] is not None and before[k] is not None else None
        for k in KERNELS
    }
    tot = (
        sum(v for v in new.values() if v is not None)
        if any(v is not None for v in new.values())
        else None
    )
    rec = [
        (
            c["dql"],
            c["calls"],
            c["max_total_q"],
            int(c["capturing"]),
            c["q"]["align"],
            c["seq_align"],
            c["bt_align"],
        )
        for c in _S["classes"].values()
    ]
    _S["last"] = dict(rank=rank, max_batch=mb, calls=calls, failed=failed, new=new)
    log(
        "sparse-decode prewarm done rank=%d classes=%d local_batches=1..%d(%s) dqls=%s calls=%d failed=%d "
        "new_specializations=%s "
        "(producer=%s pair=%s merge=%s) recorded_calls=%d skipped=%d seconds=%.1f classes(dql,calls,max_total_q,"
        "capturing,q_align,seq_align,bt_align)=%s"
        % (
            rank,
            len(_S["classes"]),
            mb,
            src,
            sorted({c["dql"] for c in _S["classes"].values()} | extra_dqls),
            calls,
            failed,
            tot,
            new[KERNELS[0]],
            new[KERNELS[1]],
            new[KERNELS[2]],
            _S["recorded"],
            _S["skipped"],
            time.time() - t0,
            rec,
        ),
        warn=bool(failed),
    )
    for e in errs[:6]:
        log(
            "sparse-decode prewarm: %s (those keys compile lazily on first use, as stock)" % e,
            warn=True,
        )


def _install_compile_observer():
    if _H["hook"]:
        return
    from triton import knobs

    prev = knobs.runtime.jit_post_compile_hook
    from . import core

    def m3ax_jit_post_compile_hook(*args, **kwargs):
        try:
            if core._PHASE[0] == "serve":
                fn = kwargs.get("fn")
                name = str(getattr(fn, "name", None) or "?")
                sd = int(name.rsplit(".", 1)[-1] in KERNELS)
                _H["serve_compiles"] += 1
                _H["serve_sd_compiles"] += sd
                log(
                    "triton serve compile kernel=%s sparse_decode=%d n=%d sd_n=%d %s"
                    % (
                        name,
                        sd,
                        _H["serve_compiles"],
                        _H["serve_sd_compiles"],
                        str(kwargs.get("repr") or "")[:900],
                    ),
                    warn=bool(sd),
                )
        except Exception:  # noqa: BLE001
            pass
        if prev is not None:
            return prev(*args, **kwargs)
        return None

    knobs.runtime.jit_post_compile_hook = m3ax_jit_post_compile_hook
    _H["hook"] = True


def finish(runner, run=True):
    """End of DecoderWarmup.run: restore the decode function; after a successful warmup warm every local batch and
    set the serve-time compile observer. Never raises."""
    if _S.get("mod") is None:
        return
    try:
        _restore()
    except Exception as exc:  # noqa: BLE001
        log("sparse-decode prewarm: restore failed: %s: %s" % (type(exc).__name__, exc), warn=True)
    if not run:
        return
    try:
        if _S["classes"]:
            _warm(runner)
        else:
            log(
                "sparse-decode prewarm skipped: no sparse-decode call recorded during warmup (seen=%d skipped=%d %s)"
                % (_S["seen"], _S["skipped"], _S["skip_reasons"]),
                warn=True,
            )
    except Exception as exc:  # noqa: BLE001
        log(
            "sparse-decode prewarm failed after the recorder (stock lazy JIT applies): %s: %s"
            % (type(exc).__name__, exc),
            warn=True,
        )
    finally:
        _S["classes"] = {}  # drop the references to the persistent K/V/scale views
    try:
        _install_compile_observer()
    except Exception as exc:  # noqa: BLE001
        log(
            "triton serve-compile observer not installed: %s: %s" % (type(exc).__name__, exc),
            warn=True,
        )
