import dataclasses
from typing import Callable, List, Optional, Sequence, Union
from unittest.mock import patch

import torch
from torch._guards import detect_fake_mode
from torch._inductor.compile_fx import compile_fx
from torch._subclasses import FakeTensor
from torch.fx import GraphModule, Interpreter
from torch.fx.passes.split_module import split_module

from tensorrt_llm.llmapi.utils import enable_llm_debug
from tensorrt_llm.logger import logger

from ..utils import get_piecewise_cuda_graph_flag, make_weak_ref
from .multi_stream import multi_stream_pass
from .utils import (get_arg, get_enable_piecewise_cuda_graph_capture_flag,
                    is_call_function)


class PiecewiseInterpreter(Interpreter):

    def __init__(
        self,
        module: GraphModule,
        enable_inductor: bool,
        enable_multi_stream: bool,
        compile_time_num_tokens: Union[int | torch.SymInt],
        cuda_graph_batch_sizes: list[int],
        exclude_modules_id: list[int],
        graph_pool_handle: tuple[int, int],
        garbage_collect_values: bool = True,
        graph=None,
    ):
        super().__init__(module, garbage_collect_values, graph)

        self.fake_mode = detect_fake_mode()

        self.compile_time_num_tokens = compile_time_num_tokens
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes
        self.exclude_modules = [f"submod_{i}" for i in exclude_modules_id]
        self.graph_pool_handle = graph_pool_handle
        self.enable_inductor = enable_inductor if not enable_multi_stream else False
        self.enable_multi_stream = enable_multi_stream

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

            self.module.__dict__[target] = PiecewiseRunner(
                submod,
                target,
                self.compile_time_num_tokens,
                runtime_num_tokens_idx,
                self.cuda_graph_batch_sizes,
                self.graph_pool_handle,
                submod,
                self.enable_inductor,
                self.enable_multi_stream,
            )

        return output


@dataclasses.dataclass
class Entry:
    shape: int

    enable_inductor: bool = False
    enable_multi_stream: bool = False
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
        cuda_graph_batch_sizes: List[int],
        graph_pool_handle,
        default_callable: Callable,
        enable_inductor: bool,
        enable_multi_stream: bool,
    ):
        if runtime_num_tokens_idx != None:
            assert isinstance(compile_time_num_tokens, torch.SymInt)

        self.graph = graph
        self.name = name
        self.default_callable = default_callable
        self.compile_time_num_tokens = compile_time_num_tokens
        self.runtime_num_tokens_idx = runtime_num_tokens_idx
        self.call_count = 0
        self.graph_pool_handle = graph_pool_handle
        self.enable_inductor = enable_inductor
        self.enable_multi_stream = enable_multi_stream

        self.entries: dict[int, Entry] = {}

        for bs in cuda_graph_batch_sizes:
            self.entries[bs] = Entry(
                bs,
                enable_inductor=self.enable_inductor,
                callable=default_callable,
            )

    def __call__(self, *args):
        runtime_num_of_token = None
        if self.runtime_num_tokens_idx != None:
            runtime_num_of_token = int(
                args[self.runtime_num_tokens_idx[0]].shape[
                    self.runtime_num_tokens_idx[1]])
        elif isinstance(self.compile_time_num_tokens, int):
            runtime_num_of_token = self.compile_time_num_tokens

        if runtime_num_of_token is None or runtime_num_of_token not in self.entries or not get_piecewise_cuda_graph_flag(
        ):
            return self.default_callable(*args)

        entry = self.entries[runtime_num_of_token]

        if not entry.compiled:
            if self.enable_multi_stream:
                entry.callable = multi_stream_pass(entry.callable)
            elif self.enable_inductor:
                entry.callable = compile_fx(entry.callable, args)

            entry.compiled = True

        if entry.cuda_graph is None:

            if not get_enable_piecewise_cuda_graph_capture_flag():
                logger.warning(
                    f"Unexpectedly capture cuda graph for {self.name} with runtime_num_of_token {runtime_num_of_token}. Will fallback to non-CUDA graph execution."
                )
                return entry.callable(*args)

            if entry.warmup_count < 2:
                entry.warmup_count += 1
                return self.default_callable(*args)

            entry.input_addresses = [
                i.data_ptr() for i in args if isinstance(i, torch.Tensor)
            ]

            graph = torch.cuda.CUDAGraph()

            # Torch's cuda graph will call gc.collect() internally. This will slow down the performance.
            # We patch it to do nothing.
            with patch("gc.collect", lambda: None):
                # TODO: consider to use `make_graphed_callables()` when
                # it's ready rather than capture it ourselves
                with torch.cuda.graph(graph, pool=self.graph_pool_handle):
                    output = entry.callable(*args)

            entry.cuda_graph = graph
            # Mark weak ref here. The intermediate activation tensor should be freed properly.
            # Here we don't use python native weakref since we still need the object to be alive when the graph is replayed.
            entry.output = make_weak_ref(output)
            entry.output_addresses = [
                i.data_ptr() for i in output if isinstance(i, torch.Tensor)
            ]

            return output

        if enable_llm_debug():
            runtime_input_addresses = [
                i.data_ptr() for i in args if isinstance(i, torch.Tensor)
            ]
            runtime_output_addresses = [
                i.data_ptr() for i in output if isinstance(i, torch.Tensor)
            ]

            assert (entry.input_addresses == runtime_input_addresses
                    ), f"{entry.input_addresses} vs\n {runtime_input_addresses}"
            assert (
                entry.output_addresses == runtime_output_addresses
            ), f"{entry.output_addresses} vs\n {runtime_output_addresses}"

        entry.cuda_graph.replay()

        return entry.output


def piecewise_optimizer(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    enable_inductor: bool,
    enable_multi_stream: bool,
    input_num_tokens: Union[int | torch.SymInt],
    cuda_graph_batch_sizes: Sequence[int],
    graph_pool_handle: tuple[int, int],
) -> GraphModule:
    graph_pool_handle = torch.cuda.graph_pool_handle()
    graph = gm.graph
    nodes_to_remove = []

    symint_node = {}

    for node in graph.nodes:
        if node.op == "placeholder":
            if isinstance(node.meta["val"], torch.SymInt):
                symint_node[str(node.meta["val"])] = node

        elif is_call_function(node, torch.ops.trtllm.attention.default):
            q = get_arg(node, 0, "q")
            fake_mode = detect_fake_mode()
            with fake_mode:
                new_output_tensor = torch.ops.trtllm.attention.default(
                    *[
                        i.meta["val"] if hasattr(i, "meta") else i
                        for i in node.args
                    ],
                    **node.kwargs,
                )
            with graph.inserting_before(node):
                output = graph.call_function(
                    torch.ops.aten.new_empty.default,
                    (q, [
                        symint_node[str(i)]
                        if isinstance(i, torch.SymInt) else i
                        for i in new_output_tensor.shape
                    ]),
                    {"dtype": new_output_tensor.dtype},
                )
                if len(node.args) > 3:
                    args = node.args[:3] + tuple((output, )) + node.args[3:]
                else:
                    node.kwargs["output"] = output
                new_attention = graph.call_function(
                    torch.ops.trtllm.attention_inplace.default, args,
                    node.kwargs)
                node.replace_all_uses_with(output)
                output.meta["val"] = new_output_tensor

            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.erase_node(node)

    gm.recompile()

    stop_partition = False
    node_to_graph_id = {}
    idx = 0
    exclude_modules_id = []

    for node in graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if (not stop_partition and is_call_function(node, [
                torch.ops.trtllm.attention_inplace.default,
                torch.ops.aten.index.Tensor,
                torch.ops.aten.cumsum.default,
        ])):
            idx += 1
            node_to_graph_id[node] = idx
            exclude_modules_id.append(idx)
            if node.target != torch.ops.trtllm.attention_inplace.default:
                # We only know it is safe to continue splitting after attention
                # since attention_inplace will not produce any new tensor
                stop_partition = True
            else:
                idx += 1
        else:
            node_to_graph_id[node] = idx

    gm = split_module(gm,
                      None,
                      lambda node: node_to_graph_id[node],
                      keep_original_order=True)

    PiecewiseInterpreter(
        gm,
        enable_inductor,
        enable_multi_stream,
        input_num_tokens,
        cuda_graph_batch_sizes,
        exclude_modules_id,
        graph_pool_handle,
    ).run(*example_inputs)

    return gm
