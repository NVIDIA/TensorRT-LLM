# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MEMCPY4B: the attention metadata's seq_lens copy of a forked DSpark step taken on the fork's stream.

DFlash's ``_prepare_attn_metadata_for_dflash`` runs ``prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")`` between the
forced acceptance and the KV rewind: a 4-byte graph memcpy node on the sampling chain at batch 1. On a forked step the
worker takes the same copy first on the fork's stream (``dspark_tailfold.take_seq_lens_snapshot``) and swaps it in at
the stock point after an event wait (``install_seq_lens_snapshot``). The state must be the stock one: the saved dict
(the original host and device tensors, in the stock order), the host copy, the device copy's bytes, and, after the
draft forward's fill and the restore, the originals untouched. Checked eagerly, in a CUDA graph built like the step
(the fork after a head kernel, the copy read on the main stream right after the swap and again after the fill) over
replays whose original seq_lens change through the setter's upload and whose accepted counts walk 1..8, and on the
worker (the copy is used only for the step that took it, never in the warmup's three-field branch).

MEMCPY4B is off by default (``TRTLLM_M31_ENABLE_MEMCPY4B=1`` opts in, ``TRTLLM_M31_DISABLE_MEMCPY4B=1`` still wins):
the switch only gates the worker's set-up (``_tf4b_on``), and these tests drive the opt-in path's functions and the
worker directly, whatever the switch.
"""

import ast
import importlib
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp4"
_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3." + TARGET
_DIR = Path(_family.__file__).resolve().parent / TARGET

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

KP1 = 8


class _Md:
    """The attention metadata's spec-dec save / restore (the base class's own functions) over its seq_lens fields."""

    prepare_for_spec_dec = AttentionMetadata.prepare_for_spec_dec
    restore_from_spec_dec = AttentionMetadata.restore_from_spec_dec

    def __init__(self, n, dev, value=KP1):
        self._saved_tensors = {}
        self._seq_lens = torch.full((n,), value, dtype=torch.int32)
        self._seq_lens_cuda = self._seq_lens.to(dev)
        self.kv_lens_cuda = torch.arange(n, dtype=torch.int32, device=dev) + 100


def _tfold():
    return importlib.import_module(_PKG + ".dspark_tailfold")


def _spec(is_cuda_graph=False):
    return SimpleNamespace(is_cuda_graph=is_cuda_graph)


@needs_gpu
@pytest.mark.parametrize("n", [1, 3])
def test_fork_copy_equals_the_stock_prepare(n):
    tf = _tfold()
    dev = torch.device("cuda")
    stock, fork = _Md(n, dev), _Md(n, dev)
    s_orig, f_orig = stock._seq_lens_cuda, fork._seq_lens_cuda
    s_host, f_host = stock._seq_lens, fork._seq_lens
    stock.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
    aux = torch.cuda.Stream()
    aux.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(aux):
        snap = tf.take_seq_lens_snapshot(fork, _spec())
    assert snap is not None and snap.src is f_orig
    assert tf.install_seq_lens_snapshot(snap, fork, _spec())
    assert (
        list(fork._saved_tensors) == list(stock._saved_tensors) == ["_seq_lens", "_seq_lens_cuda"]
    )
    assert (
        fork._saved_tensors["_seq_lens"] is f_host
        and fork._saved_tensors["_seq_lens_cuda"] is f_orig
    )
    assert (
        stock._saved_tensors["_seq_lens"] is s_host
        and stock._saved_tensors["_seq_lens_cuda"] is s_orig
    )
    assert fork._seq_lens is not f_host and torch.equal(fork._seq_lens, stock._seq_lens)
    assert fork._seq_lens_cuda is snap.copy and fork._seq_lens_cuda is not f_orig
    assert torch.equal(fork._seq_lens_cuda, stock._seq_lens_cuda)
    # The draft forward's fill writes the copy, the restore brings back the untouched originals.
    for md in (stock, fork):
        md._seq_lens_cuda[0:n] = KP1 + 1
        md._seq_lens[0:n] = KP1 + 1
        md.restore_from_spec_dec()
    assert fork._seq_lens_cuda is f_orig and fork._seq_lens is f_host and not fork._saved_tensors
    assert torch.equal(f_orig, s_orig) and int(f_orig.min()) == KP1 and int(f_host.min()) == KP1


@needs_gpu
def test_declines_leave_the_stock_prepare():
    tf = _tfold()
    dev = torch.device("cuda")
    md = _Md(1, dev)
    assert (
        tf.take_seq_lens_snapshot(md, _spec(is_cuda_graph=True)) is None
    )  # warmup: three-field branch
    assert tf.take_seq_lens_snapshot(None, _spec()) is None
    snap = tf.take_seq_lens_snapshot(md, _spec())
    other = _Md(1, dev)
    assert not tf.install_seq_lens_snapshot(None, md, _spec())
    assert not tf.install_seq_lens_snapshot(snap, other, _spec())  # another metadata
    assert not tf.install_seq_lens_snapshot(snap, md, _spec(is_cuda_graph=True))  # warmup
    md._saved_tensors["x"] = md.kv_lens_cuda
    assert not tf.install_seq_lens_snapshot(snap, md, _spec())  # a save already pending
    md._saved_tensors.clear()
    md._seq_lens_cuda = md._seq_lens_cuda.clone()
    assert not tf.install_seq_lens_snapshot(
        snap, md, _spec()
    )  # the field was replaced since the copy
    assert not md._saved_tensors

    class _Own(_Md):
        def prepare_for_spec_dec(self, *fields):  # a metadata with its own save: the stock path
            return AttentionMetadata.prepare_for_spec_dec(self, *fields)

    own = _Own(1, dev)
    assert tf.take_seq_lens_snapshot(own, _spec()) is None


def _step(md, tf, aux, use_fork, nacc, kv_lens, obs, n):
    """One step's order: head -> fork (copy first) | sampling -> forced counts -> prepare -> rewind -> read ->
    join -> fill -> read -> restore -> read."""
    main = torch.cuda.current_stream()
    head = nacc.float().sum()  # the verify head (any kernel before the fork)
    aux.wait_stream(main)
    with torch.cuda.stream(aux):
        snap = tf.take_seq_lens_snapshot(md, _spec()) if use_fork else None
        side = torch.ones(1 << 16, device=nacc.device).cumsum(0)  # the fork's own work
    for _ in range(4):  # the sampling chain
        head = head + nacc.float().sum()
    nacc.bitwise_or_(0)  # the forced counts (in place; a kernel of its own name)
    if not (use_fork and tf.install_seq_lens_snapshot(snap, md, _spec())):
        md.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
    kv_lens.add_(1)  # the KV rewind's bump
    obs[0].copy_(md._seq_lens_cuda[:n])
    main.wait_stream(aux)
    md._seq_lens_cuda[0:n] = KP1  # prepare_1st_drafter_inputs' fill
    obs[1].copy_(md._seq_lens_cuda[:n])
    obs[2].copy_(side[-1:].to(torch.int32))
    md.restore_from_spec_dec()
    torch.add(
        md._seq_lens_cuda[:n], 0, out=obs[3]
    )  # a kernel read (no copy of the original in the graph)
    return head


@needs_gpu
@pytest.mark.parametrize("n", [1, 2])
def test_fork_copy_in_a_cuda_graph(n):
    tf = _tfold()
    dev = torch.device("cuda")
    sides = {}
    for name in ("stock", "fork"):
        md = _Md(n, dev)
        sides[name] = SimpleNamespace(
            md=md,
            orig=md._seq_lens_cuda,
            nacc=torch.ones(n, dtype=torch.int32, device=dev),
            kv=torch.zeros(n, dtype=torch.int32, device=dev),
            obs=[torch.full((n,), -1, dtype=torch.int32, device=dev) for _ in range(4)],
            aux=torch.cuda.Stream(),
            g=torch.cuda.CUDAGraph(),
        )
    for name, s in sides.items():
        w = torch.cuda.Stream()
        w.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(w):
            _step(s.md, tf, s.aux, name == "fork", s.nacc, s.kv, s.obs, n)
        torch.cuda.current_stream().wait_stream(w)
        with torch.cuda.graph(s.g):
            _step(s.md, tf, s.aux, name == "fork", s.nacc, s.kv, s.obs, n)
        assert s.md._seq_lens_cuda is s.orig and not s.md._saved_tensors
    for t in range(2 * KP1):
        host = torch.tensor([(t + r) % KP1 + 1 for r in range(n)], dtype=torch.int32)
        for s in sides.values():
            s.orig.copy_(host.to(dev))  # the seq_lens setter's upload before the replay
            s.nacc.copy_(host.to(dev))
            s.g.replay()
        torch.cuda.synchronize()
        a, b = sides["stock"], sides["fork"]
        for i in range(4):
            assert torch.equal(a.obs[i], b.obs[i]), (t, i)
        assert torch.equal(b.obs[0].cpu(), host), (
            "the copy read after the swap is not the step's seq_lens"
        )
        assert torch.equal(b.obs[3].cpu(), host) and torch.equal(a.kv, b.kv)


def _dot():
    spec = importlib.util.spec_from_file_location(
        "_m31_r6_graph_dot", Path(__file__).parent / "_m31_r6_graph_dot.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_gpu
def test_the_copy_leaves_the_sampling_chain():
    """In the captured step the stock copy of seq_lens waits for the forced counts (it sits on the sampling chain);
    the fork's copy does not (only for the head, as the fork does)."""
    tf = _tfold()
    dot = _dot()
    dev = torch.device("cuda")
    found = {}
    for name in ("stock", "fork"):
        md = _Md(1, dev)
        orig = md._seq_lens_cuda
        nacc = torch.ones(1, dtype=torch.int32, device=dev)
        kv = torch.zeros(1, dtype=torch.int32, device=dev)
        obs = [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(4)]
        aux = torch.cuda.Stream()
        _, nodes, edges = dot.capture(lambda: _step(md, tf, aux, name == "fork", nacc, kv, obs, 1))
        copies = dot.copies_from(nodes, orig)
        assert len(copies) == 1 and nodes[copies[0]]["width"] == 4, name
        forced = dot.kernels(nodes, "BitwiseOr")
        assert len(forced) == 1, [v["name"] for v in nodes.values()]
        found[name] = forced[0] in dot.ancestors(edges, copies[0])
        # The read right after the swap (obs[0]) waits for the copy: on the fork through the event.
        reads = [
            k for k, v in nodes.items() if v["type"] == "MEMCPY" and v["dst"] == obs[0].data_ptr()
        ]
        assert len(reads) == 1 and nodes[reads[0]]["src"] == nodes[copies[0]]["dst"], name
        assert len(reads) == 1 and copies[0] in dot.ancestors(edges, reads[0]), name
    assert found == {"stock": True, "fork": False}


# A two-node dump as cudaGraphDebugDotPrint writes it (driver 580: kernel name mangled, record fields by position).
_KERNEL_MANGLED = (
    "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_16BitwiseOrFunctorIiEEEEviT0_T1_"
)
_DOT_MANGLED = "\n".join(
    [
        "digraph dot {",
        "subgraph cluster_1 {",
        'label="graph_1" graph[style="dashed"];',
        '"graph_1_node_0"[style="bold" shape="record" label="{KERNEL',
        "| {ID | 0 (topoId: 2) | " + _KERNEL_MANGLED + r"\<\<\<1,128,0\>\>\>}",
        "| {{node handle | func handle} | {0x00005995A5A6D110 | 0x00005995A57FCF30}}",
        '}"];',
        "",
        '"graph_1_node_1"[style="solid" shape="record" label="{',
        "MEMCPY",
        "| {{ID | node handle} | {1 (topoId: 1) | 0x00005995A5A6D878}}",
        "| {kind | DtoD (DEVICE to DEVICE)}",
        "| {{srcPtr | dstPtr} | {pitch | ptr | xsize | ysize | pitch | ptr | xsize | ysize} | "
        "{0 | 0x000076EF4F600000 | 0 | 0 | 0 | 0x000076EF46600000 | 0 | 0}}",
        "| {{srcPos | {{x | 0} | {y | 0} | {z | 0}}} | {dstPos | {{x | 0} | {y | 0} | {z | 0}}} | "
        "{Extent | {{Width | 4} | {Height | 1} | {Depth | 1}}}}",
        '}"];',
        "",
        '"graph_1_node_0" -> "graph_1_node_1" [headlabel=0];',
        "}",
        "}",
        "",
    ]
)


def test_dot_reader_reads_both_name_forms_and_refuses_an_unknown_layout():
    """The DOT reader takes a kernel's name from its launch configuration, mangled or demangled, and a copy's addresses
    by field name; a layout it cannot read raises 'DOT format not recognised' (a harness artifact)."""
    dot = _dot()
    nodes, edges = dot.parse(_DOT_MANGLED)
    k, m = nodes["graph_1_node_0"], nodes["graph_1_node_1"]
    assert (
        k["type"] == "KERNEL" and "BitwiseOr" in k["name"] and k["name"].startswith("_ZN2at6native")
    )
    assert (m["type"], m["width"], m["src"], m["dst"]) == (
        "MEMCPY",
        4,
        0x76EF4F600000,
        0x76EF46600000,
    )
    assert edges == [("graph_1_node_0", "graph_1_node_1")]
    assert dot.ancestors(edges, "graph_1_node_1") == {"graph_1_node_0"}
    demangled = _DOT_MANGLED.replace(
        "| {ID | 0 (topoId: 2) | " + _KERNEL_MANGLED,
        r"| {ID | 0 | void at::native::vectorized_elementwise_kernel\<4, at::native::BitwiseOrFunctor\<int\>, "
        r"std::array\<char*, 3ul\> \>(int, at::native::BitwiseOrFunctor\<int\>, std::array\<char*, 3ul\>)",
    )
    k2 = dot.parse(demangled)[0]["graph_1_node_0"]
    assert (
        k2["name"].startswith("void at::native::vectorized_elementwise_kernel")
        and "BitwiseOr" in k2["name"]
    )
    unknown = _DOT_MANGLED.replace("{{srcPtr | dstPtr}", "{{source | destination}")
    with pytest.raises(dot.DotFormatError, match="DOT format not recognised"):
        dot.parse(unknown)
    with pytest.raises(dot.DotFormatError, match="DOT format not recognised"):
        dot.parse("digraph dot {\n}\n")


# ------------------------------------------------------------------------------------------------------- the worker


def _worker():
    dspark = importlib.import_module(_PKG + ".dspark")
    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    torch.nn.Module.__init__(w)
    w.mapping = SimpleNamespace(tp_rank=2)
    return dspark, w


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

    warning = info


@needs_gpu
def test_worker_takes_the_copy_for_its_own_step_only(monkeypatch):
    tf = _tfold()
    dspark, w = _worker()
    log = _Log()
    monkeypatch.setattr(dspark, "logger", log)
    dev = torch.device("cuda")
    md = _Md(1, dev)
    orig = md._seq_lens_cuda
    st = object()
    w._tf = st
    # The engagement line reports the first take under capture (not an eager step before it).
    for capt, lines in ((False, 0), (True, 1)):
        monkeypatch.setattr(tf, "capturing", lambda capt=capt: capt)
        snap = tf.take_seq_lens_snapshot(md, _spec())
        w._tf4b = (st, snap)
        w._prepare_attn_metadata_for_dflash(md, _spec())
        assert md._seq_lens_cuda is snap.copy and md._saved_tensors["_seq_lens_cuda"] is orig
        assert w._tf4b is None
        taken = [x for x in log.lines if x.startswith("[M3 dispatch] MEMCPY4B taken: ")]
        assert len(taken) == lines
        md.restore_from_spec_dec()
    assert taken[0].endswith("(1 requests) on rank 2")
    # Another step's copy: the stock prepare (a fresh clone).
    w._tf4b = (object(), tf.take_seq_lens_snapshot(md, _spec()))
    w._prepare_attn_metadata_for_dflash(md, _spec())
    assert md._seq_lens_cuda is not orig and md._saved_tensors["_seq_lens_cuda"] is orig
    md.restore_from_spec_dec()
    # The warmup (graph metadata, not capturing): the stock three-field prepare.
    w._tf4b = (st, snap)
    w._prepare_attn_metadata_for_dflash(md, _spec(is_cuda_graph=True))
    assert list(md._saved_tensors) == ["_seq_lens", "_seq_lens_cuda", "kv_lens_cuda"]
    md.restore_from_spec_dec()
    w._tf = None
    w._tf4b = (None, snap)
    w._prepare_attn_metadata_for_dflash(md, _spec())
    assert md._seq_lens_cuda is not snap.copy
    md.restore_from_spec_dec()
    assert len([x for x in log.lines if "MEMCPY4B taken" in x]) == 1


@pytest.mark.parametrize(
    "enable, disable, on, reason",
    [
        ("", "", False, " (default off: opt in with TRTLLM_M31_ENABLE_MEMCPY4B=1)"),
        ("1", "", True, ""),
        ("1", "1", False, " (TRTLLM_M31_DISABLE_MEMCPY4B set)"),
        ("", "1", False, " (TRTLLM_M31_DISABLE_MEMCPY4B set)"),
    ],
)
def test_off_by_default_opt_in_and_the_kill_switch_wins(monkeypatch, enable, disable, on, reason):
    sw = importlib.import_module(_PKG + ".m31_switches")
    tf = _tfold()
    w = SimpleNamespace(mapping=SimpleNamespace(tp_rank=1))
    head = "M3.1 DSpark MEMCPY4B: seq_lens copy on the fork's stream "
    try:
        for name in ("TRTLLM_M31_ENABLE_MEMCPY4B", "TRTLLM_M31_DISABLE_MEMCPY4B"):
            monkeypatch.delenv(name, raising=False)
        if enable:
            monkeypatch.setenv("TRTLLM_M31_ENABLE_MEMCPY4B", enable)
        if disable:
            monkeypatch.setenv("TRTLLM_M31_DISABLE_MEMCPY4B", disable)
        importlib.reload(sw)
        importlib.reload(tf)
        assert tf.MEMCPY4B is on
        # The worker's set-up gate (prebuild: _tf4b_on = MEMCPY4B and the fork's stream exists) and its line.
        line = tf.tailfold_setup_line(w, True, True, tf.MEMCPY4B)[2]
        assert line == head + ("on" if on else "off") + " on rank 1" + reason
        # Switched on but not set up (no fork stream): a plain off line.
        if on:
            assert tf.tailfold_setup_line(w, True, True, False)[2] == head + "off on rank 1"
    finally:
        monkeypatch.undo()
        importlib.reload(sw)
        importlib.reload(tf)


def test_the_copy_is_the_forks_first_op_and_the_step_drops_it():
    src = (_DIR / "dspark.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "M3DSparkWorker")
    fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    launch = ast.get_source_segment(src, fns["_tf_launch"])
    with_at = launch.index("with torch.cuda.stream(aux):")
    take_at = launch.index("_tfold.take_seq_lens_snapshot(st.attn, spec)")
    copy_at = launch.index("copy_captures(")
    assert with_at < take_at < copy_at
    end = ast.get_source_segment(src, fns["_tfold_end_step"])
    assert "self._tf4b = None" in end
    flat = re.sub(r"[\s()]", "", ast.get_source_segment(src, fns["prebuild"]))
    assert (
        re.sub(r"[\s()]", "", "self._tf4b_on = _tfold.MEMCPY4B and self._tf_aux is not None")
        in flat
    )
