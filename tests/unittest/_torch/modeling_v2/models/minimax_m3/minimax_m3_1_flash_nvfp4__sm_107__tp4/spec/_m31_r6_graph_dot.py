# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The node graph of a captured CUDA graph, read from its DOT dump (cudaGraphDebugDotPrint through
``torch.cuda.CUDAGraph.debug_dump``): kernel nodes by name, memcpy nodes by size and source address, and the edges.
Used by the TAILFOLD tests to show which node a fold removed or moved.

The dump's record labels are read by field name, not by position: a kernel's name is the text before its launch
configuration (``name<<<grid,block,smem>>>``, mangled or demangled, any escaped character inside), a copy's source and
destination are the first and second ``ptr`` fields of its ``{srcPtr | dstPtr}`` record. A dump whose kernel or copy
nodes cannot be read that way raises ``DotFormatError`` ("DOT format not recognised"): a harness artifact (a driver
that prints another layout), not a product result."""

import os
import re
import tempfile
from typing import Callable, Dict, List, Set, Tuple

import torch

_NODE = re.compile(r'^"(graph_\d+_node_\d+)"\[[^\]]*?label="\{\s*(\w+)', re.M)
_EDGE = re.compile(r'^"(graph_\d+_node_\d+)" -> "(graph_\d+_node_\d+)"', re.M)
# The kernel name: the field text right before the launch configuration "\<\<\<grid,block,smem\>\>\>".
_KNAME = re.compile(r"\|\s*((?:\\.|[^|{}\\])+?)\s*\\<\\<\\<\s*\d")
_KNAME_ID = re.compile(r"\{\s*ID\s*\|[^|}]*\|\s*((?:\\.|[^|{}\\])+)")
_WIDTH = re.compile(r"\{\s*Width\s*\|\s*(\d+)\s*\}")
_SRCDST = re.compile(r"\{\s*\{\s*srcPtr\s*\|\s*dstPtr\s*\}\s*\|\s*\{([^{}]*)\}\s*\|\s*\{([^{}]*)\}")


class DotFormatError(AssertionError):
    """The DOT dump's layout is not one this reader knows (a harness artifact, not a product failure)."""


def capture(fn: Callable[[], object]) -> Tuple[object, Dict[str, dict], List[Tuple[str, str]]]:
    """Capture ``fn`` once (after one eager warm-up on a side stream) and return (its result, nodes, edges). Each node
    is {"type": "KERNEL" | "MEMCPY" | ..., "text": its label, "name": the kernel name, "width": the copy's bytes,
    "src" / "dst": the copy's source / destination address}."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph(keep_graph=True)
    g.enable_debug_mode()
    with torch.cuda.graph(g):
        out = fn()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.dot")
        g.debug_dump(path)
        with open(path) as f:
            text = f.read()
    nodes, edges = parse(text)
    return out, nodes, edges


def _kernel_name(block: str) -> str:
    m = _KNAME.search(block) or _KNAME_ID.search(block)
    return m.group(1).strip() if m else ""


def _copy_ptrs(block: str) -> Tuple[int, int]:
    """(source, destination) address of a memcpy node: the first and second "ptr" fields of its {srcPtr | dstPtr}
    record, -1 where absent."""
    m = _SRCDST.search(block)
    if not m:
        return -1, -1
    names = [x.strip() for x in m.group(1).split("|")]
    values = [x.strip() for x in m.group(2).split("|")]
    ptrs = [v for k, v in zip(names, values) if k == "ptr"]
    out = []
    for v in ptrs[:2]:
        try:
            out.append(int(v, 16) if v.lower().startswith("0x") else int(v))
        except ValueError:
            out.append(-1)
    out += [-1] * (2 - len(out))
    return out[0], out[1]


def parse(text: str, strict: bool = True) -> Tuple[Dict[str, dict], List[Tuple[str, str]]]:
    """Nodes and edges of a DOT dump. ``strict``: raise DotFormatError when the dump holds no node, or a kernel node
    without a readable name, or a memcpy node without a readable size, source or destination."""
    blocks = re.split(r'(?m)^(?="graph_\d+_node_\d+"\[)', text)
    nodes = {}
    bad = []
    for b in blocks:
        m = _NODE.match(b)
        if not m:
            continue
        info = {"type": m.group(2), "text": b, "name": "", "width": -1, "src": -1, "dst": -1}
        if info["type"] == "KERNEL":
            info["name"] = _kernel_name(b)
            if not info["name"]:
                bad.append(b)
        elif info["type"] == "MEMCPY":
            w = _WIDTH.search(b)
            info["width"] = int(w.group(1)) if w else -1
            info["src"], info["dst"] = _copy_ptrs(b)
            if info["width"] < 0 or info["src"] < 0 or info["dst"] < 0:
                bad.append(b)
        nodes[m.group(1)] = info
    edges = [(a, b) for a, b in _EDGE.findall(text)]
    if strict and (not nodes or bad):
        sample = (bad[0] if bad else text)[:600]
        raise DotFormatError(
            f"DOT format not recognised ({len(nodes)} nodes, {len(bad)} unreadable): a harness artifact, not a "
            f"product result. First unreadable record:\n{sample}"
        )
    return nodes, edges


def ancestors(edges: List[Tuple[str, str]], node: str) -> Set[str]:
    preds: Dict[str, List[str]] = {}
    for a, b in edges:
        preds.setdefault(b, []).append(a)
    seen: Set[str] = set()
    stack = [node]
    while stack:
        for p in preds.get(stack.pop(), []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def kernels(nodes: Dict[str, dict], needle: str) -> List[str]:
    return [k for k, v in nodes.items() if v["type"] == "KERNEL" and needle in v["name"]]


def copies_from(nodes: Dict[str, dict], tensor: torch.Tensor) -> List[str]:
    """The memcpy nodes that read ``tensor``'s storage (source address equal to its data pointer)."""
    ptr = tensor.data_ptr()
    return [k for k, v in nodes.items() if v["type"] == "MEMCPY" and v["src"] == ptr]
