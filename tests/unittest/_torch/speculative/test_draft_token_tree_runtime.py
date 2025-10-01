import os
import sys
import unittest
from dataclasses import dataclass

import torch
from utils.llm_data import llm_models_root

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

# isort: off
from tensorrt_llm._torch.pyexecutor.resource_manager import (KVCacheManager,
                                                             ResourceManager,
                                                             ResourceManagerType
                                                             )
# isort: on
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTensors
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.eagle3 import (Eagle3ResourceManager,
                                                    Eagle3SpecMetadata)
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi import EagleDecodingConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
from tensorrt_llm.mapping import Mapping

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# Test the update_spec_dec_param function for static tree
def test_draft_token_static_tree_prepare_spec_params():
    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"  # It will not actually be used.

    max_num_tokens = 1024
    kv_cache_manager = None
    scheduled_requests = [
    ]  # for the static tree, we will not use the scheduled requests in update_spec_dec_param
    use_dynamic_tree = False

    def run_test(max_batch_size, max_draft_len, max_total_draft_tokens,
                 cur_draft_layer_idx, eagle_choices, is_draft_model,
                 is_spec_decoding_enabled, ref_spec_decoding_position_offsets,
                 ref_spec_decoding_packed_mask,
                 ref_spec_decoding_generation_lengths):

        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=max_batch_size,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=kv_cache_manager)

        spec_config = EagleDecodingConfig(
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            speculative_model_dir=eagle_model_dir,
            eagle3_one_model=False,
            eagle_choices=eagle_choices,
            use_dynamic_tree=use_dynamic_tree,
        )

        spec_tree_manager = SpecTreeManager(
            max_num_requests=max_batch_size,
            use_dynamic_tree=spec_config.use_dynamic_tree,
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            eagle_choices=spec_config.eagle_choices,
            dynamic_tree_max_topK=spec_config.dynamic_tree_max_topK,
        )
        spec_tree_manager.cur_draft_layer_idx = cur_draft_layer_idx

        spec_metadata = Eagle3SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_batch_size,
            num_layers=32,
            hidden_size=1024,
            max_num_tokens=max_num_tokens,
            dtype=torch.bfloat16,
            is_draft_model=is_draft_model,
            eagle3_resource_manager=None,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            eagle_choices=spec_config.eagle_choices,
            is_spec_dec_tree=spec_config.eagle_choices is not None
            or spec_config.use_dynamic_tree,
            is_spec_dec_dynamic_tree=spec_config.use_dynamic_tree,
        )

        attn_metadata.update_spec_dec_param(
            scheduled_requests=scheduled_requests,
            is_spec_decoding_enabled=is_spec_decoding_enabled,
            spec_metadata=spec_metadata,
            spec_tree_manager=spec_tree_manager,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
        )

        print(
            f"attn_metadata.spec_decoding_position_offsets: {attn_metadata.spec_decoding_position_offsets}"
        )
        print(
            f"ref_spec_decoding_position_offsets: {ref_spec_decoding_position_offsets}"
        )

        print(
            f"attn_metadata.spec_decoding_packed_mask: {attn_metadata.spec_decoding_packed_mask}"
        )
        print(f"ref_spec_decoding_packed_mask: {ref_spec_decoding_packed_mask}")

        print(
            f"attn_metadata.spec_decoding_generation_lengths: {attn_metadata.spec_decoding_generation_lengths}"
        )
        print(
            f"ref_spec_decoding_generation_lengths: {ref_spec_decoding_generation_lengths}"
        )

        if is_spec_decoding_enabled:
            assert torch.all(attn_metadata.spec_decoding_position_offsets ==
                             ref_spec_decoding_position_offsets)
            assert torch.all(attn_metadata.spec_decoding_packed_mask ==
                             ref_spec_decoding_packed_mask)
            assert torch.all(attn_metadata.spec_decoding_generation_lengths ==
                             ref_spec_decoding_generation_lengths)
        else:
            assert attn_metadata.spec_decoding_position_offsets is None
            assert attn_metadata.spec_decoding_packed_mask is None
            assert attn_metadata.spec_decoding_generation_lengths is None

    ################## CASE 1 is_spec_decoding_enabled = False ##########################
    max_batch_size = 1
    is_spec_decoding_enabled = False
    max_draft_len = 3
    max_total_draft_tokens = 12
    cur_draft_layer_idx = 0
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]]
    is_draft_model = False
    ref_spec_decoding_position_offsets = None
    ref_spec_decoding_packed_mask = None
    ref_spec_decoding_generation_lengths = None

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens,
             cur_draft_layer_idx, eagle_choices, is_draft_model,
             is_spec_decoding_enabled, ref_spec_decoding_position_offsets,
             ref_spec_decoding_packed_mask,
             ref_spec_decoding_generation_lengths)

    ################## CASE 2 target model ##########################
    max_batch_size = 1
    is_spec_decoding_enabled = True
    max_draft_len = 3
    max_total_draft_tokens = 12
    cur_draft_layer_idx = 0
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]]
    is_draft_model = False  # i.e, target model
    ref_spec_decoding_position_offsets = torch.tensor(
        [[0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]],
        dtype=torch.int,
        device='cuda')
    ref_spec_decoding_packed_mask = torch.tensor(
        [1, 3, 5, 9, 19, 35, 67, 133, 261, 521, 1043, 2083, 4229],
        dtype=torch.int,
        device='cuda').reshape(1, max_total_draft_tokens + 1, 1)
    ref_spec_decoding_generation_lengths = torch.tensor([13],
                                                        dtype=torch.int,
                                                        device='cuda')

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens,
             cur_draft_layer_idx, eagle_choices, is_draft_model,
             is_spec_decoding_enabled, ref_spec_decoding_position_offsets,
             ref_spec_decoding_packed_mask,
             ref_spec_decoding_generation_lengths)

    ################## CASE 3 target model, batch_size = 2 ##########################
    max_batch_size = 2
    is_spec_decoding_enabled = True
    max_draft_len = 3
    max_total_draft_tokens = 12
    cur_draft_layer_idx = 0
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]]
    is_draft_model = False  # i.e, target model
    ref_spec_decoding_position_offsets = torch.tensor(
        [[0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]],
        dtype=torch.int,
        device='cuda').repeat(2, 1, 1)
    ref_spec_decoding_packed_mask = torch.tensor(
        [1, 3, 5, 9, 19, 35, 67, 133, 261, 521, 1043, 2083, 4229],
        dtype=torch.int,
        device='cuda').reshape(1, max_total_draft_tokens + 1,
                               1).repeat(2, 1, 1)
    ref_spec_decoding_generation_lengths = torch.tensor([13],
                                                        dtype=torch.int,
                                                        device='cuda').repeat(
                                                            2, 1)

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens,
             cur_draft_layer_idx, eagle_choices, is_draft_model,
             is_spec_decoding_enabled, ref_spec_decoding_position_offsets,
             ref_spec_decoding_packed_mask,
             ref_spec_decoding_generation_lengths)

    ################## CASE 4 target model, bigger tree ##########################
    max_batch_size = 1
    is_spec_decoding_enabled = True
    max_draft_len = 4
    max_total_draft_tokens = 20
    cur_draft_layer_idx = 0
    eagle_choices = [[0], [1], [2], [3], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [3, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 2, 0],
                     [1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                     [0, 1, 0, 0]]
    is_draft_model = False  # i.e, target model
    ref_spec_decoding_position_offsets = torch.tensor(
        [[0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]],
        dtype=torch.int,
        device='cuda')
    ref_spec_decoding_packed_mask = torch.tensor(
        [
            1, 3, 5, 9, 17, 35, 67, 131, 261, 517, 1033, 2065, 4131, 8227,
            16451, 32899, 65797, 135203, 266275, 532515, 1065027
        ],
        dtype=torch.int,
        device='cuda').reshape(1, max_total_draft_tokens + 1, 1)
    ref_spec_decoding_generation_lengths = torch.tensor([21],
                                                        dtype=torch.int,
                                                        device='cuda')

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens,
             cur_draft_layer_idx, eagle_choices, is_draft_model,
             is_spec_decoding_enabled, ref_spec_decoding_position_offsets,
             ref_spec_decoding_packed_mask,
             ref_spec_decoding_generation_lengths)

    ################## CASE 5 drafter model, drafter_layer_idx = 0 ##########################
    max_batch_size = 1
    is_spec_decoding_enabled = True
    max_draft_len = 3
    max_total_draft_tokens = 12
    cur_draft_layer_idx = 0
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]]
    is_draft_model = True  # i.e, drafter model
    # These tensors are not used in the first drafter layer.
    ref_spec_decoding_position_offsets = torch.tensor(
        [[0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.int,
        device='cuda')
    ref_spec_decoding_packed_mask = torch.tensor(
        [[1, 3, 7, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.int,
        device='cuda').reshape(1, max_total_draft_tokens + 1, 1)
    ref_spec_decoding_generation_lengths = torch.tensor([4],
                                                        dtype=torch.int,
                                                        device='cuda')

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens,
             cur_draft_layer_idx, eagle_choices, is_draft_model,
             is_spec_decoding_enabled, ref_spec_decoding_position_offsets,
             ref_spec_decoding_packed_mask,
             ref_spec_decoding_generation_lengths)

    ################## CASE 6 drafter model, drafter_layer_idx = 1 ##########################
    max_batch_size = 1
    is_spec_decoding_enabled = True
    max_draft_len = 3
    max_total_draft_tokens = 12
    cur_draft_layer_idx = 1
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]]
    is_draft_model = True  # i.e, drafter model
    ref_spec_decoding_position_offsets = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.int,
        device='cuda')
    ref_spec_decoding_packed_mask = torch.tensor(
        [1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int,
        device='cuda').reshape(1, max_total_draft_tokens + 1, 1)
    ref_spec_decoding_generation_lengths = torch.tensor([3],
                                                        dtype=torch.int,
                                                        device='cuda')

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens,
             cur_draft_layer_idx, eagle_choices, is_draft_model,
             is_spec_decoding_enabled, ref_spec_decoding_position_offsets,
             ref_spec_decoding_packed_mask,
             ref_spec_decoding_generation_lengths)

    ################## CASE 7 drafter model, drafter_layer_idx = 2 ##########################
    max_batch_size = 1
    is_spec_decoding_enabled = True
    max_draft_len = 3
    max_total_draft_tokens = 12
    cur_draft_layer_idx = 2
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]]
    is_draft_model = True  # i.e, drafter model
    ref_spec_decoding_position_offsets = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]],
        dtype=torch.int,
        device='cuda')
    ref_spec_decoding_packed_mask = torch.tensor(
        [1, 2, 4, 9, 17, 33, 66, 130, 260, 0, 0, 0, 0],
        dtype=torch.int,
        device='cuda').reshape(1, max_total_draft_tokens + 1, 1)
    ref_spec_decoding_generation_lengths = torch.tensor([9],
                                                        dtype=torch.int,
                                                        device='cuda')

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens,
             cur_draft_layer_idx, eagle_choices, is_draft_model,
             is_spec_decoding_enabled, ref_spec_decoding_position_offsets,
             ref_spec_decoding_packed_mask,
             ref_spec_decoding_generation_lengths)


##############################################################################################################################


def _create_request(input_tokens, req_id: int, is_first_draft: bool):
    sampling_params = SamplingParams()
    kwargs = {
        "request_id":
        req_id,
        "max_new_tokens":
        128,
        "input_tokens":
        input_tokens,
        "sampling_config":
        tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        "is_streaming":
        False,
    }
    request = LlmRequest(**kwargs)
    request.paged_kv_block_ids = []
    request.py_is_first_draft = is_first_draft
    request.py_seq_slot = req_id
    request.py_batch_idx = req_id

    return request


@dataclass
class Config:
    torch_dtype: torch.dtype
    num_key_value_heads: int = 16
    num_attention_heads: int = 16
    hidden_size: int = 256
    architectures: list[str] = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class DummyModel(torch.nn.Module):

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.model_config = ModelConfig(pretrained_config=Config(
            torch_dtype=dtype))
        self.recorded_position_ids = None

    def infer_max_seq_len(self):
        return 2048

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(self, *args, **kwargs) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        self.recorded_position_ids = kwargs["position_ids"]
        batch_size = input_ids.size(0)
        return {"logits": torch.randn((batch_size, 2048), device='cuda')}


class DummyModelEngine(PyTorchModelEngine):

    def __init__(self,
                 pytorch_backend_config: PyTorchConfig,
                 spec_config: DecodingBaseConfig,
                 batch_size: int,
                 dtype: torch.dtype,
                 max_seq_len: int = 128,
                 max_total_draft_tokens: int = 12,
                 is_draft_model: bool = False) -> None:
        self.dtype = dtype
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          rank=tensorrt_llm.mpi_rank())
        self.model_is_wrapped = False
        self.hidden_size = 2048
        self.max_num_tokens = max_seq_len
        self.max_seq_len = max_seq_len

        super().__init__(
            model_path="",
            pytorch_backend_config=pytorch_backend_config,
            checkpoint_loader=None,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            mapping=mapping,
            spec_config=spec_config,
            is_draft_model=is_draft_model,
            model=DummyModel(self.dtype),
        )
        self.max_total_draft_tokens = max_total_draft_tokens


def create_model_engine_and_kvcache(spec_config,
                                    max_num_requests,
                                    batch_size,
                                    use_cuda_graph,
                                    max_seq_len,
                                    max_total_draft_tokens,
                                    is_draft_model,
                                    config: PyTorchConfig = None):
    tokens_per_block = 1
    max_tokens = 258  # Atleast 1 more than the max seq len
    num_layers = 1

    config = config if config else PyTorchConfig(
        use_cuda_graph=use_cuda_graph,
        cuda_graph_padding_enabled=use_cuda_graph)

    if use_cuda_graph:
        config.cuda_graph_batch_sizes = [
            1, 2, 4, 8, 16, 32, 64, 128
        ] if config.cuda_graph_batch_sizes is None else config.cuda_graph_batch_sizes

    model_engine = DummyModelEngine(
        pytorch_backend_config=config,
        spec_config=spec_config,
        batch_size=max_num_requests,
        dtype=torch.half,
        max_seq_len=max_seq_len,
        max_total_draft_tokens=max_total_draft_tokens,
        is_draft_model=is_draft_model)

    kv_cache_config = KvCacheConfig(max_tokens=max_tokens)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=model_engine.model.config.num_key_value_heads,
        head_dim=model_engine.model.config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_tokens,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )

    return model_engine, kv_cache_manager


# from executor.test_pytorch_model_engine import create_model_engine_and_kvcache, _create_request
# Test the prepare_inputs function for static tree
def test_draft_token_static_tree_prepare_inputs():

    max_num_requests = 1
    max_batch_size = max_num_requests
    batch_size = 1
    use_cuda_graph = False
    max_num_tokens = 128
    max_seq_len = 128
    hidden_size = 1024

    # Use same tree
    max_draft_len = 3
    max_total_draft_tokens = 12
    eagle_model_dir = ""
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]]
    use_dynamic_tree = False

    def run_test(scheduled_requests, new_tensors_device, is_draft_model,
                 cur_draft_layer_idx, ref_input_ids, ref_position_ids,
                 ref_gather_ids):

        # 1) Create spec related config, resource managers
        spec_config = EagleDecodingConfig(
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            speculative_model_dir=eagle_model_dir,
            eagle3_one_model=False,
            eagle_choices=eagle_choices,
            use_dynamic_tree=use_dynamic_tree,
        )
        eagle3_resource_manager = Eagle3ResourceManager(
            spec_config,
            torch.half,
            hidden_size,
            max_num_requests,
            max_seq_len,
            max_num_tokens,
        )
        eagle3_resource_manager.spec_tree_manager.cur_draft_layer_idx = cur_draft_layer_idx

        # 2) Create model engine and kv cache manager
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(
            spec_config=spec_config,
            max_num_requests=max_num_requests,
            batch_size=batch_size,
            use_cuda_graph=use_cuda_graph,
            max_seq_len=max_seq_len,
            max_total_draft_tokens=max_total_draft_tokens,
            is_draft_model=is_draft_model)
        model_engine._disable_overlap_scheduler = True

        for req in scheduled_requests.all_requests():
            kv_cache_manager.add_dummy_requests([req.request_id],
                                                [len(req.get_tokens(0))])
            eagle3_resource_manager.add_dummy_requests([req.request_id])

        resource_manager = ResourceManager({
            ResourceManagerType.KV_CACHE_MANAGER:
            kv_cache_manager,
            ResourceManagerType.SPEC_RESOURCE_MANAGER:
            eagle3_resource_manager
        })

        # 3) Create attn metadata
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=max_batch_size,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=kv_cache_manager)
        attn_metadata.max_seq_len = max_seq_len
        attn_metadata._max_seq_len_storage = max_seq_len

        # 4) Create spec metadata
        spec_metadata = Eagle3SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_batch_size,
            num_layers=32,
            hidden_size=1024,
            max_num_tokens=max_num_tokens,
            dtype=torch.bfloat16,
            is_draft_model=is_draft_model,
            eagle3_resource_manager=eagle3_resource_manager,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            eagle_choices=spec_config.eagle_choices,
            is_spec_dec_tree=spec_config.eagle_choices is not None
            or spec_config.use_dynamic_tree,
            is_spec_dec_dynamic_tree=spec_config.use_dynamic_tree,
        )
        model_engine.spec_metadata = spec_metadata

        # 5) Run the prepare_tp_inputs function
        inputs, gather_ids = model_engine._prepare_tp_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            new_tensors_device=new_tensors_device,
            cache_indirection_buffer=None,
            resource_manager=resource_manager)

        print(f"inputs['input_ids']: {inputs['input_ids']}")
        print(f"ref_input_ids: {ref_input_ids}")
        assert torch.all(inputs['input_ids'] == ref_input_ids)

        print(f"inputs['position_ids']: {inputs['position_ids']}")
        print(f"ref_position_ids: {ref_position_ids}")
        assert torch.all(inputs['position_ids'].squeeze(0) == ref_position_ids)

        print(f"gather_ids: {gather_ids}")
        print(f"ref_gather_ids: {ref_gather_ids}")
        assert torch.all(gather_ids == ref_gather_ids)

    ################## CASE 1 target model, the generation phase ##########################
    is_draft_model = False
    scheduled_requests = ScheduledRequests()
    target_gen_request = _create_request(
        input_tokens=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        req_id=0,
        is_first_draft=False)
    target_gen_request.py_draft_tokens = [
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ]
    scheduled_requests.generation_requests.append(target_gen_request)
    cur_draft_layer_idx = 0

    ref_input_ids = torch.tensor(
        [14, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        dtype=torch.int,
        device='cuda')
    ref_position_ids = torch.tensor(
        [14, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17],
        dtype=torch.int,
        device='cuda')
    ref_gather_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                  dtype=torch.int,
                                  device='cuda')

    run_test(scheduled_requests, None, is_draft_model, cur_draft_layer_idx,
             ref_input_ids, ref_position_ids, ref_gather_ids)

    ################## CASE 2 drafter model, context phase, the first drafter layer ##########################
    is_draft_model = True
    scheduled_requests = ScheduledRequests()
    # '[1:]prompt + new token' already done in model_drafter.py::_prepare_draft_batch()
    # The input request here are the draft batch.
    drafter_request = _create_request(
        input_tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        req_id=0,
        is_first_draft=False)
    scheduled_requests.context_requests.append(drafter_request)
    cur_draft_layer_idx = 0

    ref_input_ids = torch.tensor(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        dtype=torch.int,
        device='cuda')
    ref_position_ids = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        dtype=torch.int,
        device='cuda')
    ref_gather_ids = torch.tensor([14], dtype=torch.int, device='cuda')

    run_test(scheduled_requests, None, is_draft_model, cur_draft_layer_idx,
             ref_input_ids, ref_position_ids, ref_gather_ids)

    ################## CASE 3 drafter model, the first drafter layer ##########################
    is_draft_model = True
    scheduled_requests = ScheduledRequests()
    # the input_toeksn already be pad to max_draft_len + 1
    drafter_request = _create_request(input_tokens=[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0
    ],
                                      req_id=0,
                                      is_first_draft=True)
    drafter_request.py_num_accepted_draft_tokens = 1  #
    scheduled_requests.generation_requests.append(drafter_request)
    cur_draft_layer_idx = 0  # Prepare to execute the 0-th drafter layer

    ref_input_ids = torch.tensor([17, 18, 0, 0], dtype=torch.int,
                                 device='cuda')  # max_draft_len + 1
    ref_position_ids = torch.tensor([16, 17, 18, 19],
                                    dtype=torch.int,
                                    device='cuda')
    ref_gather_ids = torch.tensor([1], dtype=torch.int, device='cuda')

    run_test(scheduled_requests, None, is_draft_model, cur_draft_layer_idx,
             ref_input_ids, ref_position_ids, ref_gather_ids)

    ################## CASE 4 drafter model, the second drafter layer ##########################
    is_draft_model = True
    scheduled_requests = ScheduledRequests()
    drafter_request = _create_request(input_tokens=[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    ],
                                      req_id=0,
                                      is_first_draft=False)
    scheduled_requests.generation_requests.append(drafter_request)
    cur_draft_layer_idx = 1  # Prepare to execute the 1-st drafter layer

    new_tensors = torch.zeros((max_total_draft_tokens + 1, max_num_requests, 1),
                              dtype=torch.int,
                              device='cuda')
    new_tensors[:3, 0, 0] = torch.tensor([30, 31, 32],
                                         dtype=torch.int,
                                         device='cuda')
    new_tensors_device = SampleStateTensors(new_tokens=new_tensors)

    ref_input_ids = torch.tensor([30, 31, 32], dtype=torch.int, device='cuda')
    ref_position_ids = torch.tensor([17, 17, 17],
                                    dtype=torch.int,
                                    device='cuda')
    ref_gather_ids = torch.tensor([0, 1, 2], dtype=torch.int, device='cuda')

    run_test(scheduled_requests, new_tensors_device, is_draft_model,
             cur_draft_layer_idx, ref_input_ids, ref_position_ids,
             ref_gather_ids)

    ################## CASE 5 drafter model, the third drafter layer ##########################
    is_draft_model = True
    scheduled_requests = ScheduledRequests()
    drafter_request = _create_request(
        # 30, 31, 32 are from the previous drafter layer
        input_tokens=[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 30, 31,
            32
        ],
        req_id=0,
        is_first_draft=False)
    scheduled_requests.generation_requests.append(drafter_request)
    cur_draft_layer_idx = 2  # Prepare to execute the 2-nd drafter la

    new_tensors = torch.zeros((max_total_draft_tokens + 1, max_num_requests, 1),
                              dtype=torch.int,
                              device='cuda')
    new_tensors[:6, 0, 0] = torch.tensor([40, 41, 42, 43, 44, 45],
                                         dtype=torch.int,
                                         device='cuda')
    new_tensors_device = SampleStateTensors(new_tokens=new_tensors)

    # 30, 31, 32 are from the previous drafter layer
    ref_input_ids = torch.tensor([30, 31, 32, 40, 41, 42, 43, 44, 45],
                                 dtype=torch.int,
                                 device='cuda')
    ref_position_ids = torch.tensor([17, 17, 17, 18, 18, 18, 18, 18, 18],
                                    dtype=torch.int,
                                    device='cuda')
    ref_gather_ids = torch.tensor([3, 4, 6], dtype=torch.int, device='cuda')

    run_test(scheduled_requests, new_tensors_device, is_draft_model,
             cur_draft_layer_idx, ref_input_ids, ref_position_ids,
             ref_gather_ids)


if __name__ == "__main__":
    unittest.main()
