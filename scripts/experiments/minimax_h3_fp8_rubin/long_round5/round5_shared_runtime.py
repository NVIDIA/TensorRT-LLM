# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-pinned shared final FP32 input; original output-head GEMMs and timer."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
import types
from collections import Counter
from pathlib import Path

import torch


def version(t: torch.Tensor) -> int | None:
    try:
        return int(t._version)
    except RuntimeError:
        return None


def saved(obj: object, name: str) -> tuple:
    return name in vars(obj), vars(obj).get(name)


def restore(obj: object, name: str, previous: tuple) -> None:
    if previous[0]:
        setattr(obj, name, previous[1])
    elif name in vars(obj):
        delattr(obj, name)


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(
            torch.equal(
                a.contiguous().reshape(-1).view(torch.uint8),
                b.contiguous().reshape(-1).view(torch.uint8),
            )
        )
    )


class SharedRuntime:
    def __init__(self, pipeline, *, timer) -> None:
        self.pipeline = pipeline
        self.model = pipeline.transformer
        self.timer = timer
        assert not pipeline.pipeline_config.cuda_graph.enable and int(self.model.sharder.size) == 1
        self.video = self.model.proj_out
        self.audio = self.model.audio_proj_out
        assert self.video is not self.audio
        for module in (self.video, self.audio):
            assert (
                not module.training
                and not module.has_any_quant
                and module.weight.dtype == torch.float32
            )
            assert getattr(module.forward, "__func__", None) is type(module).forward
            assert not any(m._forward_hooks or m._forward_pre_hooks for m in module.modules())
        self.original = self.model.forward
        self.forward_previous = saved(self.model, "forward")
        self.original_helper = self.model._projection_input
        self.helper_previous = saved(self.model, "_projection_input")
        assert getattr(self.original_helper, "__func__", None) is type(self.model)._projection_input
        closure = inspect.getclosurevars(self.original).nonlocals
        assert (
            closure["self"] is timer
            and closure["name"] == "transformer_step"
            and closure["nvtx"] == "h3/transformer"
        )
        native = closure["original"]
        assert self.original.__wrapped__ is native
        assert getattr(native, "__func__", None) is type(self.model).forward
        path = Path(__file__).with_name("round5_shared_forward_proof.json")
        proof = json.loads(path.read_text())
        assert (
            hashlib.sha256(Path(inspect.getsourcefile(native)).read_bytes()).hexdigest()
            == proof["native_file_sha256"]
        )
        node = ast.parse(textwrap.dedent(inspect.getsource(native))).body[0]
        assert (
            isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        )
        node.body.pop(0)
        assert (
            hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            == proof["native_forward_compute_AST_sha256"]
        )
        tnode = ast.parse(textwrap.dedent(inspect.getsource(type(timer).wrap))).body[0]
        assert (
            hashlib.sha256(ast.dump(tnode, include_attributes=False).encode()).hexdigest()
            == proof["EventTimer_wrap_AST_sha256"]
        )
        assert (
            hashlib.sha256(Path(inspect.getsourcefile(type(timer).wrap)).read_bytes()).hexdigest()
            == proof["EventTimer_source_file_sha256"]
        )
        forward = path.with_name("round5_shared_forward.py")
        assert hashlib.sha256(forward.read_bytes()).hexdigest() == proof["clone_sha256"]
        namespace = dict(native.__func__.__globals__)
        exec(compile(forward.read_text(), str(forward), "exec"), namespace)
        inner = types.MethodType(namespace["_h3_shared_output_cast428_forward"], self.model)
        holder = types.SimpleNamespace(forward=inner)
        timer.wrap(holder, "forward", "transformer_step", nvtx="h3/transformer")
        self.bound = holder.forward
        assert self.bound.__code__.co_code == self.original.__code__.co_code
        self.proof = proof
        self.enabled = False
        self.validate = False
        self.current = None
        self.audio_handle = None
        self.counts = Counter()
        self.checks = []
        self.owner = self.model_stamp()
        runtime = self

        def helper(module, projection, hidden: torch.Tensor) -> torch.Tensor:
            value = runtime.original_helper(projection, hidden)
            if projection is runtime.video:
                assert runtime.enabled and runtime.validate and runtime.current is None
                assert (
                    hidden.dtype == torch.bfloat16
                    and hidden.is_contiguous()
                    and hidden.shape[-1] == 5376
                )
                assert (
                    value.dtype == torch.float32
                    and value.is_contiguous()
                    and value.shape == hidden.shape
                )
                runtime.current = (hidden, value)
                runtime.counts["video_source_checks"] += 1
            return value

        self.bound_helper = types.MethodType(helper, self.model)

        def audio_pre(module, args: tuple) -> None:
            assert module is runtime.audio and runtime.enabled and runtime.validate
            assert len(args) == 1 and runtime.current is not None
            hidden, shared = runtime.current
            try:
                assert args[0] is shared, "Bothheads must receive the SAME converted input object"
                expected = runtime.original_helper(module, hidden)
                equal = same(shared, expected)
                row = dict(
                    index=len(runtime.checks),
                    shared_input_identity=True,
                    byte_equal_original_audio_input_after_video_GEMM=equal,
                    hidden_shape=list(hidden.shape),
                    shared_shape=list(shared.shape),
                    shared_stride=list(shared.stride()),
                    shared_dtype=str(shared.dtype),
                )
                runtime.checks.append(row)
                runtime.counts["audio_input_checks"] += 1
                assert equal, row
            finally:
                runtime.current = None

        self.audio_pre = audio_pre

    def model_stamp(self) -> tuple:
        assert self.model.proj_out is self.video and self.model.audio_proj_out is self.audio
        return tuple(
            (
                id(m),
                type(m),
                bool(m.has_any_quant),
                bool(m.training),
                id(getattr(m.forward, "__func__", m.forward)),
                tuple(
                    (
                        n,
                        id(p),
                        p.data_ptr(),
                        tuple(p.shape),
                        str(p.dtype),
                        str(p.device),
                        version(p),
                    )
                    for n, p in m.named_parameters()
                ),
            )
            for m in (self.video, self.audio)
        )

    def configure(self, enabled: bool, *, validate: bool = False) -> None:
        assert self.model_stamp() == self.owner, "Frozen originalFP32 outputheads changed"
        assert (
            not self.pipeline.pipeline_config.cuda_graph.enable
            and int(self.model.sharder.size) == 1
        )
        if enabled:
            assert (
                not self.enabled
                and self.model.forward == self.original
                and self.model._projection_input == self.original_helper
            )
            assert not self.video._forward_pre_hooks and not self.audio._forward_pre_hooks
            self.validate = bool(validate)
            self.enabled = True
            self.model.forward = self.bound
            if validate:
                self.model._projection_input = self.bound_helper
                self.audio_handle = self.audio.register_forward_pre_hook(self.audio_pre)
        else:
            if self.enabled:
                assert self.model.forward == self.bound
                if self.audio_handle is not None:
                    self.audio_handle.remove()
                    self.audio_handle = None
                    assert self.model._projection_input == self.bound_helper
                    restore(self.model, "_projection_input", self.helper_previous)
                restore(self.model, "forward", self.forward_previous)
            self.enabled = False
            self.validate = False
            self.current = None

    def report(self) -> dict:
        return dict(
            enabled=self.enabled,
            counts=dict(self.counts),
            checks=list(self.checks),
            source_proof=self.proof,
            scope=(
                "Same frozen U1 pipeline, original FP32 media heads and GEMMs. "
                "One exact conversion supplies both heads; validation hooks are removed before timing."
            ),
        )
