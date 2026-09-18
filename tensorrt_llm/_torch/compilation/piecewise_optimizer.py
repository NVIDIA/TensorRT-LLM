import copy
import dataclasses
import os
from typing import Callable, List, Optional, Sequence, Union
from unittest.mock import patch

import torch
from torch._guards import detect_fake_mode
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._subclasses import FakeTensor
from torch.fx import GraphModule, Interpreter
from torch.fx.passes.split_module import split_module

from tensorrt_llm.llmapi.utils import enable_llm_debug
from tensorrt_llm.logger import logger

# When torch_compile_config.max_num_streams > 1, apply the multi-stream schedule
# only to piecewise graphs running at most this many tokens; larger token counts
# (prefill / mixed iterations) keep the single-stream piece. Unset or 0 = no
# limit. Rationale (Qwen3.5-397B TP8 gen-only): the second stream overlaps the
# MoE shared expert / dense GEMMs with the routed experts and cuts small-decode
# iterations by 0.3-0.7 ms, but at prefill sizes both sides are compute-bound
# and the extra cross-stream events only add gaps.
_MULTI_STREAM_MAX_TOKENS: Optional[int] = (int(
    os.environ.get("TLLM_MULTI_STREAM_MAX_TOKENS", "0")) or None)


def multi_stream_max_tokens() -> Optional[int]:
    return _MULTI_STREAM_MAX_TOKENS

from ..utils import (get_model_extra_attrs,
                     get_per_request_prefill_cuda_graph_flag,
                     get_piecewise_cuda_graph_flag, make_weak_ref,
                     set_piecewise_running)
from .multi_stream.auto_multi_stream import multi_stream_schedule
# TLLM_PWCG_NVTX_PROBE=1 brackets every piecewise runner call in an NVTX range
# (host-side attribution of the glue between graph launches in nsys traces).
_PWCG_NVTX_PROBE = os.environ.get("TLLM_PWCG_NVTX_PROBE", "0") == "1"

from .utils import (get_capture_piecewise_cuda_graph_flag,
                    get_optional_trtllm_op, is_call_function)


def _piecewise_boundary_ops():
    op_names = [
        "attn_custom_op_inplace",
        "mla_custom_op_inplace",
        "mla_dsa_attn_inplace",
        "gdn_custom_op_inplace",
        "mamba2_custom_op_inplace",
        "minimax_m3_attn_custom_op_inplace",
    ]
    return [
        op for op in (get_optional_trtllm_op(op_name) for op_name in op_names)
        if op is not None
    ]


class PiecewiseInterpreter(Interpreter):

    def __init__(
        self,
        module: GraphModule,
        enable_inductor: bool,
        compile_time_num_tokens: Union[int | torch.SymInt],
        capture_num_tokens: list[int],
        exclude_modules_id: list[int],
        piecewise_runner_num: int,
        graph_pool_handle: tuple[int, int],
        garbage_collect_values: bool = True,
        graph=None,
        max_num_streams: int = 1,
    ):
        super().__init__(module, garbage_collect_values, graph)

        self.fake_mode = detect_fake_mode()

        self.compile_time_num_tokens = compile_time_num_tokens
        self.capture_num_tokens = capture_num_tokens
        self.piecewise_runner_num = piecewise_runner_num
        self.piecewise_runner_idx = 0
        self.exclude_modules = [f"submod_{i}" for i in exclude_modules_id]
        self.graph_pool_handle = graph_pool_handle
        self.enable_inductor = enable_inductor
        self.num_events = 0
        self.max_num_streams = max_num_streams
        self.runners: List["PiecewiseRunner"] = []

    def run(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode:
            return super().run(*fake_args)

    def call_module(self, target, args, kwargs):
        output = super().call_module(target, args, kwargs)

        submod: GraphModule = self.fetch_attr(target)
        if target not in self.exclude_modules:

            runtime_num_tokens_idx = None

            if isinstance(self.compile_time_num_tokens, torch.SymInt):
                found_dynamic_shape = False
                for input_idx, node in enumerate(submod.graph.nodes):
                    if found_dynamic_shape:
                        break
                    if node.op == "placeholder":
                        val = node.meta["val"]
                        if isinstance(val, FakeTensor):
                            for dim_idx, i in enumerate(val.shape):
                                if (isinstance(i, torch.SymInt)
                                        and i == self.compile_time_num_tokens):
                                    runtime_num_tokens_idx = (input_idx,
                                                              dim_idx)
                                    found_dynamic_shape = True
                                    break
                if not found_dynamic_shape:
                    raise RuntimeError(
                        "Cannot identify dynamic shape, please disable enable_piecewise_cuda_graph in TorchCompileConfig"
                    )

            single_stream_submod = None
            if self.max_num_streams > 1 and not self.enable_inductor:
                if multi_stream_max_tokens() is not None:
                    # Keep a single-stream twin of the piece for token counts
                    # above the limit: same parameters/buffers (shared by
                    # reference), its own copy of the graph.
                    single_stream_submod = GraphModule(
                        submod, copy.deepcopy(submod.graph))
                num_events = multi_stream_schedule(submod, self.max_num_streams)
                self.num_events = max(self.num_events, num_events)
                submod.recompile()

            runner = PiecewiseRunner(
                submod,
                target,
                self.compile_time_num_tokens,
                runtime_num_tokens_idx,
                self.capture_num_tokens,
                self.graph_pool_handle,
                compile_fx_inner(submod, args)
                if self.enable_inductor else submod,
                self.enable_inductor,
                self.piecewise_runner_idx == 0,
                self.piecewise_runner_idx == self.piecewise_runner_num - 1,
                large_callable=single_stream_submod,
                multi_stream_max_tokens=multi_stream_max_tokens(),
            )
            self.module.__dict__[target] = runner
            self.runners.append(runner)
            self.piecewise_runner_idx += 1
        return output


@dataclasses.dataclass
class Entry:
    shape: int

    enable_inductor: bool = False
    compiled: bool = False
    warmup_count: int = 0

    # Store the addresses of the input and output tensors for debug purpose
    input_addresses: Optional[List[int]] = None
    output_addresses: Optional[List[int]] = None

    cuda_graph: Optional[torch.cuda.CUDAGraph] = None
    callable: Optional[Callable] = None
    output: Optional[torch.Tensor] = None


class PiecewiseRunner(object):

    def __init__(
        self,
        graph: GraphModule,
        name: str,
        compile_time_num_tokens: Union[int | torch.SymInt],
        runtime_num_tokens_idx: tuple[int],
        capture_num_tokens: List[int],
        graph_pool_handle,
        default_callable: Callable,
        enable_inductor: bool,
        is_first_runner: bool,
        is_last_runner: bool,
        large_callable: Optional[Callable] = None,
        multi_stream_max_tokens: Optional[int] = None,
    ):
        """``large_callable`` (optional) is the single-stream twin of the piece,
        used instead of ``default_callable`` (the multi-stream schedule) for
        token counts above ``multi_stream_max_tokens``."""
        if runtime_num_tokens_idx != None:
            assert isinstance(compile_time_num_tokens, torch.SymInt)

        self.graph = graph
        self.name = name
        self.default_callable = default_callable
        self.large_callable = large_callable
        self.multi_stream_max_tokens = multi_stream_max_tokens
        self.compile_time_num_tokens = compile_time_num_tokens
        self.runtime_num_tokens_idx = runtime_num_tokens_idx
        self.call_count = 0
        self.graph_pool_handle = graph_pool_handle
        self.enable_inductor = enable_inductor

        self.entries: dict[int, Entry] = {}
        self.is_first_runner = is_first_runner
        self.is_last_runner = is_last_runner

        for num_tokens in capture_num_tokens:
            self.entries[num_tokens] = Entry(
                num_tokens,
                enable_inductor=self.enable_inductor,
                callable=self.callable_for(num_tokens),
            )

    def callable_for(self, num_tokens: Optional[int]) -> Callable:
        """The piece to run for ``num_tokens``: the single-stream twin above
        the multi-stream token limit, the (possibly multi-stream) default
        otherwise. Unknown token counts take the default."""
        if (self.large_callable is not None
                and self.multi_stream_max_tokens is not None
                and num_tokens is not None
                and num_tokens > self.multi_stream_max_tokens):
            return self.large_callable
        return self.default_callable

    def clear_cuda_graphs(self):
        """Release captures while retaining buckets for a later warmup."""
        for entry in self.entries.values():
            if entry.cuda_graph is not None:
                entry.cuda_graph.reset()
            entry.cuda_graph = None
            entry.warmup_count = 0
            entry.input_addresses = None
            entry.output_addresses = None
            entry.output = None

    def __call__(self, *args):
        if _PWCG_NVTX_PROBE:
            torch.cuda.nvtx.range_push("pwcg_runner")
            try:
                return self._call(*args)
            finally:
                torch.cuda.nvtx.range_pop()
        return self._call(*args)

    def _call(self, *args):
        runtime_num_of_token = None
        if self.runtime_num_tokens_idx != None:
            runtime_num_of_token = int(
                args[self.runtime_num_tokens_idx[0]].shape[
                    self.runtime_num_tokens_idx[1]])
        elif isinstance(self.compile_time_num_tokens, int):
            runtime_num_of_token = self.compile_time_num_tokens

        if (runtime_num_of_token is None
                or runtime_num_of_token not in self.entries
                or not get_piecewise_cuda_graph_flag()
                or not get_per_request_prefill_cuda_graph_flag()):
            return self.callable_for(runtime_num_of_token)(*args)

        if self.is_first_runner or self.is_last_runner:
            if self.is_first_runner == self.is_last_runner:
                set_piecewise_running(False)
            else:
                set_piecewise_running(self.is_first_runner)

        entry = self.entries[runtime_num_of_token]

        if entry.enable_inductor and not entry.compiled:
            entry.callable = compile_fx(entry.callable, args)
            entry.compiled = True

        if entry.cuda_graph is None:

            if not get_capture_piecewise_cuda_graph_flag():
                return entry.callable(*args)

            if entry.warmup_count < 3:
                entry.warmup_count += 1
                return entry.callable(*args)

            entry.input_addresses = [
                i.data_ptr() for i in args if isinstance(i, torch.Tensor)
            ]

            graph = torch.cuda.CUDAGraph()

            # Torch's cuda graph will call gc.collect() internally. This will slow down the performance.
            # We patch it to do nothing.
            with patch("gc.collect", lambda: None):
                # TODO: consider to use `make_graphed_callables()` when
                # it's ready rather than capture it ourselves
                # Graph Capture would override the stream. We need to setup the stream correctly.
                extra_attrs = get_model_extra_attrs()
                with torch.cuda.graph(graph, pool=self.graph_pool_handle):
                    extra_attrs["global_stream"] = torch.cuda.current_stream()
                    output = entry.callable(*args)
                extra_attrs["global_stream"] = torch.cuda.current_stream()

            entry.cuda_graph = graph
            # Mark weak ref here. The intermediate activation tensor should be freed properly.
            # Here we don't use python native weakref since we still need the object to be alive when the graph is replayed.
            entry.output = make_weak_ref(output)
            entry.output_addresses = [
                i.data_ptr() for i in output if isinstance(i, torch.Tensor)
            ]

            entry.cuda_graph.replay()

            return output

        if enable_llm_debug():
            runtime_input_addresses = [
                i.data_ptr() for i in args if isinstance(i, torch.Tensor)
            ]

            assert (entry.input_addresses == runtime_input_addresses
                    ), f"{entry.input_addresses} vs\n {runtime_input_addresses}"

        entry.cuda_graph.replay()

        return entry.output


def piecewise_optimizer(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    enable_inductor: bool,
    input_num_tokens: Union[int | torch.SymInt],
    capture_num_tokens: Sequence[int],
    max_num_streams: int = 1,
) -> tuple[GraphModule, int, List[PiecewiseRunner]]:
    graph_pool_handle = torch.cuda.graph_pool_handle()
    graph = gm.graph

    stop_partition = False
    node_to_graph_id = {}
    idx = 0
    exclude_modules_id = []
    piecewise_boundary_ops = _piecewise_boundary_ops()

    for node in graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        is_boundary = is_call_function(node, piecewise_boundary_ops)
        stop_target = is_call_function(node, [
            torch.ops.aten.index.Tensor,
            torch.ops.aten.cumsum.default,
        ])
        if not stop_partition and (is_boundary or stop_target):
            idx += 1
            node_to_graph_id[node] = idx
            exclude_modules_id.append(idx)
            if not is_boundary:
                # We only know it is safe to continue splitting after attention
                stop_partition = True
            else:
                idx += 1
        else:
            node_to_graph_id[node] = idx

    gm = split_module(gm,
                      None,
                      lambda node: node_to_graph_id[node],
                      keep_original_order=True)

    interpreter = PiecewiseInterpreter(
        gm,
        enable_inductor,
        input_num_tokens,
        capture_num_tokens,
        exclude_modules_id,
        len(set(node_to_graph_id.values())) - len(exclude_modules_id),
        graph_pool_handle,
        max_num_streams=max_num_streams,
    )

    interpreter.run(*example_inputs)

    if max_num_streams > 1:
        logger.info(
            f"piecewise multi-stream schedule: {len(interpreter.runners)} pieces, "
            f"max_num_streams={max_num_streams}, events={interpreter.num_events}, "
            f"token gate={multi_stream_max_tokens()}, "
            f"dump={os.environ.get('TLLM_MULTI_STREAM_DUMP') or 'off'}")

    return gm, interpreter.num_events, interpreter.runners
