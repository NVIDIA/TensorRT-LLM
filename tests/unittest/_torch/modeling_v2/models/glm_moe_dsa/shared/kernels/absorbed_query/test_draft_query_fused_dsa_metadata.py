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
"""draft_absorbed_query in the MTP draft loop on the DSA decode-metadata path of the served env.

The served GEN environment exports ``TRTLLM_FUSED_DSA_METADATA=1``. TensorRT-LLM
does not read that name. The fused DSA decode-metadata kernel
(``dsa/kernels.py::fused_dsa_decode_metadata``) is on by default and is
turned off only by ``TRTLLM_DISABLE_FUSED_DSA_METADATA``
(``dsa/metadata.py::_fused_dsa_meta_enabled``, cached per process). On a
pure-decode step, ``DSAtrtllmAttentionMetadata.on_update_kv_lens`` then fills
the request index per token, the indexer K-cache slot mappings and the two
generation indptrs with one launch instead of the eager chain.

One decode iteration (the harness of ``test_draft_query_draft_loop_integration.py``:
target layer 6, then the native MTP Eagle draft loop over layer 78, real
weights, FP8 KV cache, two requests with a 2,600-token context) updates the
DSA decode metadata six times:

* ``u0``: ``on_update_kv_lens()`` at the start of the step (B x 6 rows);
* ``u1``..``u5``: after each of the five draft steps (B rows), through
  ``update_for_spec_dec()`` -> ``on_update_kv_lens()`` (native step update),
  or through draft_loop's one-launch ``draft_step_update`` when the step update of
  the served draft_loop package is attached to the worker.

``u0`` feeds layer 6 and draft step 0, and ``u<i>`` feeds draft step ``i``.
Every update is recorded: which path ran (native or draft_loop), how many fused
metadata kernels it launched, and the values it left in the request index,
the slot mappings, the indptrs, the KV lengths (1-D and 2-D views) and the
paged-MQA schedules.

Arms per case (custom-op or ``forward_impl`` attention path; native or draft_loop
draft-step update):

* native, eager metadata chain (``TRTLLM_DISABLE_FUSED_DSA_METADATA=1``);
* native, served chain (fused kernel), repeated for the output-noise bound;
* draft_absorbed_query (layer 78 at every draft step), served chain, then repeated;
* draft_absorbed_query, eager chain.

Asserted:

* served env: no TensorRT-LLM source or library reads
  ``TRTLLM_FUSED_DSA_METADATA``; the fused path is on with the served env
  and off only with the opt-out;
* the update sequence: served chain -> one fused launch at each native update
  (six per iteration, or one at ``u0`` with the draft_loop step update, whose five
  steps then run draft_loop's kernel); eager chain -> no fused launch;
* every recorded metadata value of every update is identical between the
  eager and the fused chain, with and without draft_absorbed_query, and with the draft_loop step
  update;
* draft_absorbed_query under the served chain against native under the served chain: caches,
  MLA inputs, indexer inputs and appended indexer keys bit-identical; top-k
  sets equal; draft-loop top-k reuse at steps 1-4; outputs within the
  native run-to-run bound; draft_absorbed_query repeat bit-identical; draft_absorbed_query under the
  eager chain bit-identical to draft_absorbed_query under the served chain;
* CUDA graph: the whole iteration captured with the served chain (draft_absorbed_query graph
  and native graph in one pool), replayed 1, 1 and 20 times with fresh
  inputs and accepted counts: the captured fused launches, the replayed
  metadata, caches and records equal the eager runs as above;
* a negative control: a slot mapping corrupted after the fused launch of one
  draft step is caught in that update's metadata and in the layer-78 indexer
  K cache.
"""

import contextlib
import mmap
import os
import re
from types import SimpleNamespace

import pytest
import test_draft_query_draft_loop_integration as dq_loop_integration
import torch

TARGET, MTP, LAYERS = (
    dq_loop_integration.TARGET,
    dq_loop_integration.MTP,
    dq_loop_integration.LAYERS,
)
DRAFT_LEN, VERIFY_ROWS = dq_loop_integration.DRAFT_LEN, dq_loop_integration.VERIFY_ROWS
NATIVE_RUNS = dq_loop_integration.NATIVE_RUNS
SERVED_ENV = "TRTLLM_FUSED_DSA_METADATA"
DISABLE_ENV = "TRTLLM_DISABLE_FUSED_DSA_METADATA"
# Metadata outputs of one update: per token, per sequence.
TOKEN_FIELDS = ("req_idx_per_token", "slot_mapping_fp8", "slot_mapping_scale")
SEQ_FIELDS = ("gen_kv_indptr", "gen_cached_token_indptr")
SCHEDULE_FIELDS = ("scheduler_metadata_buffer", "scheduler_metadata_buffer_full_next_n")
# One indexer key in the FP8 slot mapping (bytes per token).
INDEX_KEY_BYTES = 128

pytestmark = dq_loop_integration.pytestmark
harness = dq_loop_integration.harness


def _select_chain(chain: str):
    """Served env, with the fused metadata chain ("fused") or the opt-out
    ("eager"). The gate is cached per process; it is reset here and never
    flipped while a graph is being captured."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import _fused_dsa_meta_enabled

    os.environ[SERVED_ENV] = "1"
    if chain == "fused":
        os.environ.pop(DISABLE_ENV, None)
    else:
        os.environ[DISABLE_ENV] = "1"
    _fused_dsa_meta_enabled.cache_clear()
    assert _fused_dsa_meta_enabled() == (chain == "fused")


def _features(step: str):
    """Feature selections of the native and the draft_absorbed_query arm. The draft_loop arm attaches
    draft_loop's draft-loop operations to the worker (``_draft_loop_step_update``) instead
    of selecting ``draft_loop``: the selection would also wrap the draft embedding,
    which the harness's stub draft model does not have and which plays no
    part in the decode metadata."""
    del step
    return {"draft_absorbed_query": "draft_absorbed_query", "native": ""}


def _inputs(h, nreq, seed, accepted=None):
    it = h.iteration_inputs(nreq, seed=seed, accepted=accepted)
    # Step-0 rows of the last accepted tokens (the draft_loop step-0 gather).
    it.gather_ids = (
        torch.arange(nreq, device="cuda", dtype=torch.int64) * VERIFY_ROWS
        + it.num_accepted.to(torch.int64)
        - 1
    )
    return it


class _MetaLog:
    """Records every DSA decode-metadata update of the metadata object
    ``meta``: the native ``on_update_kv_lens`` and draft_loop's step update. Inside a
    capture the clones are graph-pool tensors that every replay refreshes;
    the update kinds and launch counts are host values seen at capture.

    ``corrupt_update``: index of the update after whose fused launch the FP8
    slot mapping is moved by one indexer key (negative control)."""

    def __init__(self, meta, corrupt_update=None):
        self.meta = meta
        self.corrupt_update = corrupt_update
        self.updates = []
        self.fused_rows = []

    def __enter__(self):
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_loop import (  # noqa: E501
            draft_loop as draft_loop_mod,
        )
        from tensorrt_llm._torch.attention.backends.sparse.dsa import metadata as dsa_metadata

        log = self
        self._mod = dsa_metadata
        self._fused = dsa_metadata.fused_dsa_decode_metadata
        self._ops_cls = draft_loop_mod.DraftLoopOps
        self._step_update = draft_loop_mod.DraftLoopOps.step_update
        native_update = self.meta.on_update_kv_lens

        def fused(*args, **kwargs):
            out = log._fused(*args, **kwargs)
            log.fused_rows.append(kwargs["num_tokens"])
            if log.corrupt_update == len(log.updates):
                args[4].add_(INDEX_KEY_BYTES)  # slot_mapping_fp8[:num_tokens]
            return out

        def on_update_kv_lens():
            before = len(log.fused_rows)
            native_update()
            log._record("native", len(log.fused_rows) - before)

        def step_update(ops, attn_metadata, *args, **kwargs):
            before = len(log.fused_rows)
            result = log._step_update(ops, attn_metadata, *args, **kwargs)
            if result is not None and attn_metadata is log.meta:
                log._record("draft_loop", len(log.fused_rows) - before)
            return result

        dsa_metadata.fused_dsa_decode_metadata = fused
        draft_loop_mod.DraftLoopOps.step_update = step_update
        self.meta.on_update_kv_lens = on_update_kv_lens
        return self

    def __exit__(self, *exc):
        self._mod.fused_dsa_decode_metadata = self._fused
        self._ops_cls.step_update = self._step_update
        del self.meta.on_update_kv_lens
        return False

    def _record(self, kind, fused):
        m = self.meta
        n, s, g = m.num_tokens, m.num_seqs, m.num_generations
        fields = {k: getattr(m, k)[:n].clone() for k in TOKEN_FIELDS}
        fields.update({k: getattr(m, k)[: s + 1].clone() for k in SEQ_FIELDS})
        fields["kv_lens_cuda"] = m.kv_lens_cuda[:s].clone()
        width = min(m.gen_token_stride, m.kv_lens_cuda_2d.shape[1])
        fields["kv_lens_cuda_2d"] = m.kv_lens_cuda_2d[:g, :width].clone()
        for k in SCHEDULE_FIELDS:
            t = getattr(m, k, None)
            if torch.is_tensor(t):
                fields[k] = t.clone()
        self.updates.append(SimpleNamespace(kind=kind, rows=n, seqs=s, fused=fused, fields=fields))

    def snapshot(self):
        return [
            SimpleNamespace(
                kind=u.kind,
                rows=u.rows,
                seqs=u.seqs,
                fused=u.fused,
                fields={k: v.clone() for k, v in u.fields.items()},
            )
            for u in self.updates
        ]


@contextlib.contextmanager
def _draft_loop_step_update(h, meta, it, enabled):
    """Attach draft_loop's draft-loop operations to the worker, as the served
    ``_forward_impl`` does for an eligible forward with ``draft_loop`` selected. The
    harness has no target sampling and no draft embedding, so only draft_loop's
    step-0 row gather and its per-step metadata update run; acceptance, the
    draft-token stack and the embedding stay native."""
    if not enabled:
        yield None
        return
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_loop import (  # noqa: E501
        draft_loop as draft_loop_mod,
    )

    ops = draft_loop_mod.DraftLoopOps(h.worker, meta, SimpleNamespace(runtime_draft_len=DRAFT_LEN))
    ops.gather_ids = it.gather_ids
    h.worker._draft_loop_ops = ops
    try:
        yield ops
    finally:
        h.worker._draft_loop_ops = None


def _run(h, nreq, it, features, step, chain, path="custom_op", corrupt_update=None):
    """One eager iteration from the current caches."""
    _select_chain(chain)
    dq_loop_integration._set_features(features)
    h.reset_runners()
    meta = h.gen_metadata(nreq)
    with (
        _MetaLog(meta, corrupt_update) as ml,
        _draft_loop_step_update(h, meta, it, step == "draft_loop") as ops,
        dq_loop_integration._LaunchLog() as launches,
    ):
        out, rec, calls = h.run(nreq, it, path, meta=meta)
    return SimpleNamespace(
        out=out,
        rec=rec,
        calls=calls,
        meta=ml.updates,
        launches=launches.calls,
        cache=h.snapshot(),
        draft_loop_engaged=ops is not None and ops.step_engaged,
    )


def _expected_updates(nreq, step, fused):
    """``(kind, rows, fused launches)`` of the six metadata updates."""
    draft_loop_pkg = step == "draft_loop" and fused
    first = ("native", nreq * VERIFY_ROWS, int(fused))
    later = ("draft_loop", nreq, 0) if draft_loop_pkg else ("native", nreq, int(fused))
    return [first] + [later] * DRAFT_LEN


def _check_updates(updates, nreq, step, fused, tag):
    got = [(u.kind, u.rows, u.fused) for u in updates]
    want = _expected_updates(nreq, step, fused)
    if got != want or any(u.seqs != nreq for u in updates):
        return [f"{tag}: metadata updates {got}, expected {want}"]
    return []


def _compare_meta(ref, cand, report, tag):
    """Every recorded value of every update bit-identical."""
    if [(u.rows, u.seqs) for u in ref] != [(u.rows, u.seqs) for u in cand]:
        return [
            f"{tag}: metadata updates {[(u.rows, u.seqs) for u in cand]}"
            f" vs {[(u.rows, u.seqs) for u in ref]}"
        ]
    fails = []
    for i, (r, c) in enumerate(zip(ref, cand)):
        if set(r.fields) != set(c.fields):
            fails.append(f"{tag}: u{i} fields {sorted(c.fields)} vs {sorted(r.fields)}")
            continue
        for key, a in r.fields.items():
            b = c.fields[key]
            same = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            report[f"{tag}:u{i}.{key}_same"] = same
            if not same:
                fails.append(f"{tag}: u{i} {key} differs")
    return fails


def _summary(report):
    """Counters of a report: byte-difference counters (all must be 0),
    key-set and metadata comparisons (all must hold)."""
    nbytes = [v for k, v in report.items() if k.endswith("_bytes_differing")]
    sets = [v for k, v in report.items() if k.endswith("_same_sets")]
    meta = [v for k, v in report.items() if k.endswith("_same")]
    return (
        f"byte counters {len(nbytes)} (nonzero "
        f"{sum(1 for v in nbytes if v)}), key-set comparisons "
        f"{len(sets)} (unequal {sum(1 for v in sets if not v)}), "
        f"metadata comparisons {len(meta)} (unequal "
        f"{sum(1 for v in meta if not v)})"
    )


def _repeat_native(h, nreq, it, features, step, snap, path="custom_op"):
    """Native iterations under the served chain from one snapshot; they must
    repeat bit for bit in caches, recorded inputs and metadata."""
    runs, fails = [], []
    accepted = it.num_accepted.tolist()
    expected = dq_loop_integration._expected_slots(nreq, accepted)
    for i in range(NATIVE_RUNS):
        h.restore(snap)
        run = _run(h, nreq, it, features, step, "fused", path)
        assert run.calls == [], f"native run called absorbed_query entry points: {run.calls}"
        report = {}
        fails += _check_updates(run.meta, nreq, step, True, f"native{i}")
        fails += dq_loop_integration._check_reuse(run.rec, accepted, report, f"native{i}")
        if i:
            fails += dq_loop_integration._compare_caches(
                snap, runs[0].cache, run.cache, expected, report, f"native_repeat{i}"
            )
            fails += dq_loop_integration._compare_records(
                runs[0].rec, run.rec, report, f"native_repeat{i}"
            )
            fails += _compare_meta(runs[0].meta, run.meta, report, f"native_repeat{i}")
        runs.append(run)
    h.restore(snap)
    assert not fails, f"native iteration is not repeatable: {fails}"
    return runs


def _output_bounds(native_outs):
    """Per (layer, call) output bounds from native iterations of one input:
    twice the largest difference over all pairs of native runs (at least
    one BF16 ulp, relative RMS at least ``REL_RMS_FLOOR``).

    The attention inputs of these runs are bit-identical (checked
    separately), so their outputs differ only by the native layer-78
    attention's run-to-run nondeterminism, which leaves some runs
    identical and others about one BF16 ulp apart. Every pair samples it;
    differences from one reference run alone underestimate it."""
    aq_integration = dq_loop_integration.aq_integration
    bounds = {}
    for layer in LAYERS:
        for i in range(len(native_outs[0][layer])):
            outs = [run[layer][i] for run in native_outs]
            noise, rel = 0.0, 0.0
            for a in range(len(outs)):
                for b in range(a + 1, len(outs)):
                    noise = max(noise, (outs[a].float() - outs[b].float()).abs().max().item())
                    rel = max(rel, aq_integration._rel_rms(outs[a], outs[b]))
            ulp = aq_integration._bf16_ulp(outs[0].float().abs().max().item())
            bounds[(layer, i)] = (
                2.0 * max(noise, ulp),
                max(2.0 * rel, aq_integration.REL_RMS_FLOOR),
                noise,
            )
    return bounds


def _restore_defaults(h, snap):
    h.worker._draft_loop_ops = None
    h.restore(snap)
    dq_loop_integration._set_features("")
    h.reset_runners()
    _select_chain("fused")


# -- tests ------------------------------------------------------------------


def _names_served_env(data) -> bool:
    """Whether ``data`` contains ``TRTLLM_FUSED_DSA_METADATA`` as a whole
    name (not as the tail of ``TRTLLM_DISABLE_FUSED_DSA_METADATA``)."""
    name = SERVED_ENV.encode()
    start = 0
    while True:
        i = data.find(name, start)
        if i < 0:
            return False
        if i == 0 or not re.match(rb"[A-Za-z0-9_]", data[i - 1 : i]):
            return True
        start = i + 1


def test_served_env_selects_fused_metadata():
    """The served ``TRTLLM_FUSED_DSA_METADATA=1`` is read nowhere in the
    installed TensorRT-LLM (Python sources and shared libraries); the fused
    chain is on with or without it and off only with the opt-out."""
    import tensorrt_llm
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import _fused_dsa_meta_enabled

    root = os.path.dirname(tensorrt_llm.__file__)
    scanned, readers = {"py": 0, "so": 0}, []
    for dirpath, _, files in os.walk(root):
        for name in files:
            kind = "py" if name.endswith(".py") else "so" if ".so" in name else None
            if kind is None:
                continue
            path = os.path.join(dirpath, name)
            with open(path, "rb") as fh:
                if os.fstat(fh.fileno()).st_size == 0:
                    continue
                with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    scanned[kind] += 1
                    if _names_served_env(data):
                        readers.append(os.path.relpath(path, root))
    print(f"scanned {scanned} under {root}; readers of {SERVED_ENV}: {readers}")
    assert scanned["py"] > 1000 and scanned["so"] > 0, scanned
    assert readers == [], readers

    saved = {k: os.environ.get(k) for k in (SERVED_ENV, DISABLE_ENV)}
    try:
        os.environ.pop(DISABLE_ENV, None)
        for served in ("1", "0", None):
            if served is None:
                os.environ.pop(SERVED_ENV, None)
            else:
                os.environ[SERVED_ENV] = served
            _fused_dsa_meta_enabled.cache_clear()
            assert _fused_dsa_meta_enabled(), served
        os.environ[SERVED_ENV] = "1"
        os.environ[DISABLE_ENV] = "1"
        _fused_dsa_meta_enabled.cache_clear()
        assert not _fused_dsa_meta_enabled()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _fused_dsa_meta_enabled.cache_clear()


@pytest.mark.parametrize("path", ["custom_op", "forward_impl"])
@pytest.mark.parametrize("step", ["native", "draft_loop"])
@pytest.mark.parametrize("nreq", [1, 2])
@torch.inference_mode()
def test_draft_query_draft_loop_on_fused_metadata(harness, nreq, step, path):
    """draft_absorbed_query on layer 78 (all five draft steps) with the served metadata chain,
    against native on the served chain and against the eager chain."""
    h = harness
    feats = _features(step)
    it = _inputs(h, nreq, seed=40 + nreq)
    accepted = it.num_accepted.tolist()
    expected = dq_loop_integration._expected_slots(nreq, accepted)
    snap = h.snapshot()
    try:
        natives = _repeat_native(h, nreq, it, feats["native"], step, snap, path)
        native = natives[0]
        bounds = _output_bounds([r.out for r in natives])
        h.restore(snap)
        chain = _run(h, nreq, it, feats["native"], step, "eager", path)
        h.restore(snap)
        fused_run = _run(h, nreq, it, feats["draft_absorbed_query"], step, "fused", path)
        h.restore(snap)
        draft_query_again = _run(h, nreq, it, feats["draft_absorbed_query"], step, "fused", path)
        h.restore(snap)
        draft_query_chain = _run(h, nreq, it, feats["draft_absorbed_query"], step, "eager", path)

        assert fused_run.calls == dq_loop_integration._expected_calls(
            "draft_absorbed_query", nreq
        ), fused_run.calls
        assert fused_run.launches == dq_loop_integration._expected_launches(nreq), (
            fused_run.launches
        )
        assert draft_query_chain.calls == fused_run.calls and chain.calls == []
        if step == "draft_loop":
            assert native.draft_loop_engaged and fused_run.draft_loop_engaged

        report, fails = {}, []
        for tag, run, fused in (
            ("native_fused", native, True),
            ("native_eager", chain, False),
            ("draft_query_fused", fused_run, True),
            ("draft_query_again", draft_query_again, True),
            ("draft_query_eager", draft_query_chain, False),
        ):
            fails += _check_updates(run.meta, nreq, step, fused, tag)
        # Eager vs fused chain, native: metadata, caches and inputs.
        fails += _compare_meta(chain.meta, native.meta, report, "native_fused_vs_eager")
        fails += dq_loop_integration._compare_caches(
            snap, chain.cache, native.cache, expected, report, "native_fused_vs_eager"
        )
        fails += dq_loop_integration._compare_records(
            chain.rec, native.rec, report, "native_fused_vs_eager"
        )
        fails += dq_loop_integration._compare_outputs(
            native.out, chain.out, bounds, report, "native_eager_vs_fused"
        )
        # draft_absorbed_query vs native, both on the served chain.
        fails += _compare_meta(native.meta, fused_run.meta, report, "draft_query_vs_native")
        fails += dq_loop_integration._compare_to_native(
            snap,
            (native.out, native.rec, native.cache),
            (fused_run.out, fused_run.rec, fused_run.cache),
            accepted,
            nreq,
            bounds,
            report,
            "draft_query_vs_native",
        )
        # draft_absorbed_query determinism and draft_absorbed_query eager chain vs served chain.
        for tag, other in (
            ("draft_query_again", draft_query_again),
            ("draft_query_eager", draft_query_chain),
        ):
            fails += _compare_meta(fused_run.meta, other.meta, report, tag)
            fails += dq_loop_integration._compare_caches(
                snap, fused_run.cache, other.cache, expected, report, tag
            )
            fails += dq_loop_integration._compare_records(fused_run.rec, other.rec, report, tag)
            fails += dq_loop_integration._check_reuse(other.rec, accepted, report, tag)
        print(
            f"nreq={nreq} step={step} path={path} accepted={accepted} "
            f"updates={[(u.kind, u.rows, u.fused) for u in fused_run.meta]} "
            f"summary: {_summary(report)}"
        )
        print(f"report {report}")
        assert not fails, fails
    finally:
        _restore_defaults(h, snap)


@pytest.mark.parametrize("step", ["native", "draft_loop"])
@pytest.mark.parametrize("nreq", [1, 2])
@torch.inference_mode()
def test_draft_query_fused_metadata_cuda_graph(harness, nreq, step):
    """The whole iteration with the served chain captured in one CUDA graph
    per selection (draft_absorbed_query and native, one pool), replayed 1, 1 and 20 times with
    fresh inputs and accepted counts (``prepare()`` before every replay)."""
    h = harness
    snap = h.snapshot()
    graphs = {}
    try:
        _graph_checks(h, nreq, step, snap, graphs)
    except BaseException as exc:
        print(f"graph test failure: {exc!r}", flush=True)
        raise
    finally:
        graphs.clear()
        _restore_defaults(h, snap)


def _graph_checks(h, nreq, step, snap, graphs):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    path = "custom_op"
    feats = _features(step)
    _select_chain("fused")
    warm = {}
    for name, sel in feats.items():
        dq_loop_integration._set_features(sel)
        h.reset_runners()
        meta = h.gen_metadata(nreq, cuda_graph=True)
        static = _inputs(h, nreq, seed=200)
        with with_multi_stream(True):
            for _ in range(2):
                meta.prepare()
                with _draft_loop_step_update(h, meta, static, step == "draft_loop"):
                    h.iteration(meta, static, path)
        torch.cuda.synchronize()
        runners = {
            layer: (h.attn[layer]._absorbed_query, h.attn[layer]._draft_absorbed_query)
            for layer in LAYERS
        }
        if name == "draft_absorbed_query":
            assert runners[MTP][1] not in (None, False), "draft_absorbed_query did not engage"
        warm[name] = (meta, static, runners)
    h.restore(snap)
    pool = None
    for name, sel in feats.items():
        meta, static, runners = warm[name]
        dq_loop_integration._set_features(sel)
        for layer, (r_absorbed_query, r_draft_query) in runners.items():
            h.attn[layer]._absorbed_query, h.attn[layer]._draft_absorbed_query = (
                r_absorbed_query,
                r_draft_query,
            )
        meta.prepare()
        graph = torch.cuda.CUDAGraph()
        with (
            with_multi_stream(True),
            dq_loop_integration._Recorder(h.attn) as rec,
            dq_loop_integration._Engagement() as eng,
            dq_loop_integration._LaunchLog() as launches,
            _MetaLog(meta) as ml,
            _draft_loop_step_update(h, meta, static, step == "draft_loop") as ops,
        ):
            with torch.cuda.graph(graph, pool=pool):
                outs = h.iteration(meta, static, path)
        pool = graph.pool()
        assert eng.calls == dq_loop_integration._expected_calls(sel, nreq), eng.calls
        if name == "draft_absorbed_query":
            assert launches.calls == dq_loop_integration._expected_launches(nreq), launches.calls
        if step == "draft_loop":
            assert ops.step_engaged, "draft_loop step update declined at capture"
        captured = _check_updates(ml.updates, nreq, step, True, f"capture:{name}")
        assert not captured, captured
        print(
            f"[fused-meta graph] captured {name}: updates "
            f"{[(u.kind, u.rows, u.fused) for u in ml.updates]}",
            flush=True,
        )
        graphs[name] = (graph, meta, static, outs, rec, ml, runners)
        torch.cuda.synchronize()
    h.reset_runners()

    report, fails = {}, []
    trials = [(311, 1, None), (312, 1, [1] * nreq), (313, 20, [VERIFY_ROWS] * nreq)]
    for trial, (seed, replays, accepted) in enumerate(trials):
        it = _inputs(h, nreq, seed=seed, accepted=accepted)
        accepted = it.num_accepted.tolist()
        expected = dq_loop_integration._expected_slots(nreq, accepted)
        natives = _repeat_native(h, nreq, it, feats["native"], step, snap, path)
        native = natives[0]
        bounds = _output_bounds([r.out for r in natives])
        h.restore(snap)
        chain = _run(h, nreq, it, feats["native"], step, "eager", path)
        h.restore(snap)
        draft_query_eager = _run(h, nreq, it, feats["draft_absorbed_query"], step, "fused", path)
        _select_chain("fused")
        dq_loop_integration._set_features("")
        h.reset_runners()
        tag = f"t{trial}x{replays}"
        for name, (graph, meta, static, outs, rec, ml, runners) in graphs.items():
            h.restore(snap)
            meta.prepare()
            for key in ("hidden_t", "pos_t", "hidden_d", "num_accepted", "input_ids", "gather_ids"):
                getattr(static, key).copy_(getattr(it, key))
            for i in range(replays):
                if i:
                    meta.prepare()
                graph.replay()
            torch.cuda.synchronize()
            g_out = {layer: [o.clone() for o in outs[layer]] for layer in LAYERS}
            g_rec, g_meta, g_cache = rec.snapshot(), ml.snapshot(), h.snapshot()
            fails += dq_loop_integration._check_reuse(g_rec, accepted, report, f"{tag}:{name}")
            fails += _compare_meta(
                native.meta, g_meta, report, f"{tag}:{name}_graph_vs_native_fused"
            )
            fails += _compare_meta(
                chain.meta, g_meta, report, f"{tag}:{name}_graph_vs_native_eager"
            )
            if name == "draft_absorbed_query":
                fails += dq_loop_integration._compare_caches(
                    snap,
                    draft_query_eager.cache,
                    g_cache,
                    expected,
                    report,
                    f"{tag}:draft_query_graph_vs_draft_query_eager",
                )
                fails += dq_loop_integration._compare_records(
                    draft_query_eager.rec,
                    g_rec,
                    report,
                    f"{tag}:draft_query_graph_vs_draft_query_eager",
                )
                fails += dq_loop_integration._compare_to_native(
                    snap,
                    (native.out, native.rec, native.cache),
                    (g_out, g_rec, g_cache),
                    accepted,
                    nreq,
                    bounds,
                    report,
                    f"{tag}:draft_query_graph_vs_native",
                )
            else:
                fails += dq_loop_integration._compare_caches(
                    snap, native.cache, g_cache, expected, report, f"{tag}:native_graph_vs_eager"
                )
                fails += dq_loop_integration._compare_records(
                    native.rec, g_rec, report, f"{tag}:native_graph_vs_eager"
                )
                fails += dq_loop_integration._compare_outputs(
                    native.out, g_out, bounds, report, f"{tag}:native_graph_vs_eager"
                )
            for r_absorbed_query, r_draft_query in runners.values():
                for runner in (r_absorbed_query, r_draft_query):
                    if runner not in (None, False):
                        assert int(runner.sync.abs().sum()) == 0
            print(
                f"[fused-meta graph] trial {trial} {name}: replayed "
                f"x{replays}, accepted={accepted}",
                flush=True,
            )
    print(f"graph nreq={nreq} step={step} summary: {_summary(report)}")
    print(f"report {report}")
    assert not fails, fails


@torch.inference_mode()
def test_checks_detect_corrupted_draft_step_metadata(harness):
    """Negative control: the FP8 slot mapping of the update that feeds draft
    step 2 (``u2``), moved by one indexer key after the fused launch, must be
    flagged in that update only and in the layer-78 indexer K cache only."""
    h = harness
    nreq = 1
    it = _inputs(h, nreq, seed=12)
    accepted = it.num_accepted.tolist()
    snap = h.snapshot()
    try:
        h.restore(snap)
        native = _run(h, nreq, it, "", "native", "fused")
        h.restore(snap)
        bad = _run(h, nreq, it, "draft_absorbed_query", "native", "fused", corrupt_update=2)
        assert bad.calls == dq_loop_integration._expected_calls("draft_absorbed_query", nreq)
        report = {}
        meta_fails = _compare_meta(native.meta, bad.meta, report, "bad")
        cache_fails = dq_loop_integration._compare_caches(
            snap,
            native.cache,
            bad.cache,
            dq_loop_integration._expected_slots(nreq, accepted),
            report,
            "bad",
        )
        print(f"negative control: metadata {meta_fails}; caches {cache_fails}")
        assert meta_fails == ["bad: u2 slot_mapping_fp8 differs"], meta_fails
        assert any(f"index_k{MTP}" in f for f in cache_fails), cache_fails
        assert not any(f"index_k{TARGET}" in f or "latent" in f for f in cache_fails), cache_fails
    finally:
        _restore_defaults(h, snap)
