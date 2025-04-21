import os
from typing import List, Optional

import torch
import torch._inductor.config as inductor_config
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.compile_fx import compile_fx, select_decomp_table
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._subclasses import FakeTensor
from torch.fx import GraphModule

import tensorrt_llm
from tensorrt_llm import logger

from .multi_stream import multi_stream_pass
from .patterns.ar_residual_norm import register_ar_residual_norm
from .patterns.residual_add_norm import register_add_norm
from .patterns.ub_allreduce import register_ub_patterns
from .piecewise_optimizer import piecewise_optimizer
from .recover_pass import recover_pass
from .remove_copy_pass import remove_copy_for_mutates_args


class Backend:

    _custom_pass_instances: List[PatternMatcherPass] = None
    _graph_pool_handle: tuple[int, int] = None

    def __init__(
        self,
        enable_inductor=True,
        enable_userbuffers=False,
        enable_multi_stream=True,
        enable_piecewise_cuda_graph: bool = False,
        cuda_graph_batch_sizes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.elapsed_time = 0
        self.module_inference_event = []
        self.module_inference_time = 0
        self.call_count = 0
        self.custom_passes = Backend.get_custom_pass(enable_userbuffers)
        self.rank = tensorrt_llm.mpi_rank()
        # If multi-stream is enabled, we disable inductor since they are not compatible
        self.enable_inductor = enable_inductor if not enable_multi_stream else False
        self.enable_multi_stream = enable_multi_stream
        self.aux_stream = None
        self.cuda_graph_batch_sizes = (cuda_graph_batch_sizes
                                       if cuda_graph_batch_sizes is not None
                                       else [])
        self.piecewise_cuda_graph = enable_piecewise_cuda_graph
        self.no_optimization = False
        inductor_config.enable_auto_functionalized_v2 = False

        if Backend._graph_pool_handle is None:
            Backend._graph_pool_handle = torch.cuda.graph_pool_handle()

        self.match_count = []

    @classmethod
    def get_custom_pass(cls, enable_userbuffers):
        # TODO: add pp + tp support
        world_size = tensorrt_llm.mpi_world_size()
        if not cls._custom_pass_instances:
            # Really naive pass manager here
            cls._custom_pass_instances = [PatternMatcherPass()]
            if world_size > 1:
                # Currently torch compile cannot work properly with lamport fusion kernel
                # TO-DO: Fix this issue
                os.environ["DISABLE_LAMPORT_REDUCE_NORM_FUSION"] = "1"
                register_ar_residual_norm(cls._custom_pass_instances[0])
                if enable_userbuffers and tensorrt_llm.bindings.internal.userbuffers.ub_supported(
                ):
                    register_ub_patterns(cls._custom_pass_instances)
            else:
                register_add_norm(cls._custom_pass_instances[0])
        return cls._custom_pass_instances

    def bypass_optimization(self):
        self.no_optimization = True

    def enable_optimization(self):
        self.no_optimization = False

    def optimize(
        self,
        gm: GraphModule,
        example_inputs: List[torch.Tensor],
    ):
        graph = gm.graph
        for custom_pass in self.custom_passes:
            self.match_count.append(custom_pass.apply(graph))
            while self.match_count[-1]:
                self.match_count.append(custom_pass.apply(graph))
        graph.eliminate_dead_code()
        # After this pass, cannot run any dce!!!
        remove_copy_for_mutates_args(graph)
        gm.recompile()

        if self.piecewise_cuda_graph:
            return piecewise_optimizer(
                gm,
                example_inputs,
                self.enable_inductor,
                self.input_num_tokens,
                self.cuda_graph_batch_sizes,
                self._graph_pool_handle,
            )
        elif self.enable_multi_stream:
            # After multi-stream pass, we'd better not to run any pass on the graph since multi-stream relay on the specific op order to work properly
            gm, aux_stream = multi_stream_pass(gm)
            self.aux_stream = aux_stream
            return gm.forward
        elif self.enable_inductor:
            return compile_fx(gm, example_inputs)
        else:
            return gm

    def __call__(self, gm: GraphModule,
                 example_inputs: List[torch.Tensor]) -> callable:

        if self.no_optimization:
            logger.warning(
                "Bypassing torch.compile optimization and fallback to eager execution!"
            )
            return gm

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                if node.name == "l_input_ids_":
                    example_value = node.meta["example_value"]
                    assert isinstance(example_value, FakeTensor)
                    self.input_num_tokens = example_value.shape[0]
                    break

        if self.piecewise_cuda_graph:
            assert (
                self.input_num_tokens is not None
            ), "Cannot detect input_num_tokens. Cannot use piecewise CUDA graph. What is the name of `input_ids`?"

        gm = recover_pass(gm)

        return aot_module_simplified(
            gm,
            example_inputs,
            fw_compiler=self.optimize,
            decompositions=select_decomp_table(),
        )
