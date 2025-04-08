import os
from typing import List, Optional, Union

import torch
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.compile_fx import compile_fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.fx import Graph, GraphModule

import tensorrt_llm

from .patterns.ar_residual_norm import register_ar_residual_norm
from .patterns.residual_add_norm import register_add_norm
from .patterns.ub_allreduce import register_ub_patterns
from .recover_pass import recover_pass


class Backend:

    _custom_pass_instances: List[PatternMatcherPass] = None

    def __init__(self, enable_inductor=True, enable_userbuffers=False) -> None:
        super().__init__()
        self.elapsed_time = 0
        self.module_inference_event = []
        self.module_inference_time = 0
        self.call_count = 0
        self.custom_passes = Backend.get_custom_pass(enable_userbuffers)
        self.rank = tensorrt_llm.mpi_rank()
        self.enable_inductor = enable_inductor

        self.match_count = []

        if enable_inductor:
            from torch._inductor import config

            self.inductor_config = config.get_config_copy()
            self.inductor_config["joint_custom_post_pass"] = self.optimize

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

    def optimize(
        self,
        gm: Union[GraphModule | Graph],
        example_inputs: Optional[List[torch.Tensor]] = None,
    ):
        graph = gm.graph if isinstance(gm, GraphModule) else gm
        for custom_pass in self.custom_passes:
            self.match_count.append(custom_pass.apply(graph))
            while self.match_count[-1]:
                self.match_count.append(custom_pass.apply(graph))
        graph.eliminate_dead_code()
        if isinstance(gm, GraphModule):
            gm.recompile()

        return gm

    def __call__(self, gm: GraphModule,
                 example_inputs: List[torch.Tensor]) -> callable:

        gm = recover_pass(gm)

        if self.enable_inductor:
            return compile_fx(gm,
                              example_inputs,
                              config_patches=self.inductor_config)
        else:
            return aot_module_simplified(gm,
                                         example_inputs,
                                         fw_compiler=self.optimize)
