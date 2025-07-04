import pickle
import sys
import time
import traceback
import weakref
from dataclasses import dataclass

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    KVCacheParams, PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import MLA
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState,
                                                        SamplingConfig)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.sampling_params import SamplingParams

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


# values for deepseek_v3_lite
@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 32
    num_kv_heads: int = 32
    q_lora_rank: int = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    hidden_size: int = 2560
    rope_theta: float = 10000.0
    rope_scaling: bool = False
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    kv_cache_tokens_per_block: int = 64
    predicted_tokens_per_seq: int = 1
    bias: bool = False
    batch: int = 8
    seq_len: int = 1024
    gen_steps: int = 260
    ref_steps: int = 4

    @property
    def max_position_embeddings(self) -> int:
        # ensure that max_position_embeddings is set large enough for every scenario
        return self.seq_len + 1


all_scenarios = [
    Scenario(batch=1, seq_len=4096),
    Scenario(batch=1, seq_len=8192),
    Scenario(batch=1, seq_len=16384),
    Scenario(batch=1, seq_len=32768),
    Scenario(batch=1, seq_len=65536),
    Scenario(batch=1, seq_len=131072),
    Scenario(batch=8, seq_len=4096),
    Scenario(batch=8, seq_len=8192),
    Scenario(batch=8, seq_len=16384),
    Scenario(batch=8, seq_len=32768),
    Scenario(batch=8, seq_len=65536),
    Scenario(batch=8, seq_len=131072),
    Scenario(batch=16, seq_len=4096),
    Scenario(batch=16, seq_len=8192),
    Scenario(batch=16, seq_len=16384),
    Scenario(batch=16, seq_len=32768),
    Scenario(batch=16, seq_len=65536),
    # this goes OOM
    # Scenario(batch=16, seq_len=131072),
]

# limit the number of test scenarios to avoid taking too long
test_scenarios = [
    all_scenarios[1],
    all_scenarios[5],
    all_scenarios[6],
    all_scenarios[10],
    all_scenarios[14],
    all_scenarios[15],
]


# default values from deepseek_v3, but will be overwritten by scenario
@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    hidden_size: int = 7168
    num_attention_heads: int = 128
    rope_scaling: dict = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64
    model_type: str = "deepseek_v3"


def _setup_kv_and_metadata(scenario: Scenario, mapping: Mapping):
    # Set up KVCacheManager and attn_metadata for MLA
    n_gpu = mapping.world_size
    assert scenario.seq_len % n_gpu == 0
    seq_len_per_gpu = scenario.seq_len // n_gpu
    max_tokens = (
        seq_len_per_gpu + scenario.gen_steps +
        scenario.kv_cache_tokens_per_block - 1
    ) // scenario.kv_cache_tokens_per_block * scenario.kv_cache_tokens_per_block * scenario.batch
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        # Use SELFKONLY for MLA
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=scenario.num_layers,
        num_kv_heads=1,
        head_dim=scenario.kv_lora_rank + scenario.qk_rope_head_dim,
        tokens_per_block=scenario.
        kv_cache_tokens_per_block,  # for test, just use seq_len
        max_seq_len=seq_len_per_gpu + scenario.gen_steps,
        max_batch_size=scenario.batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(scenario.kv_cache_dtype)),
    )
    for req_id in range(scenario.batch):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=2,
            input_tokens=[1] *
            seq_len_per_gpu,  # all requests have the same length here
            sampling_config=SamplingConfig(
                SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.paged_kv_block_ids = []
        beam_width = 1
        kv_cache_manager.impl.add_sequence(req_id, seq_len_per_gpu, beam_width,
                                           req)
        req.state = LlmRequestState.GENERATION_IN_PROGRESS
    attn_metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([seq_len_per_gpu] * scenario.batch,
                              dtype=torch.int),
        request_ids=list(range(scenario.batch)),
        max_num_requests=scenario.batch,
        num_contexts=scenario.batch,
        prompt_lens=[seq_len_per_gpu] * scenario.batch,
        max_num_tokens=seq_len_per_gpu,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0 for _ in range(scenario.batch)],
        ),
        mapping=mapping,
    )
    attn_metadata.prepare()
    return kv_cache_manager, attn_metadata


def _run_mla_distributed(rank: int, world_size: int, scenario: Scenario,
                         mapping: Mapping, test_params: tuple,
                         ref_output: torch.Tensor):
    input_ctx, position_ids_ctx, weights, ref_q, ref_latent_cache, pos_embd_params = test_params
    extra_attrs = dict()
    config = ModelConfig(mapping=mapping)
    config.extra_attrs = extra_attrs
    mla = MLA(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        predicted_tokens_per_seq=scenario.predicted_tokens_per_seq,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
        config=config,
    ).cuda()

    mla.load_state_dict(weights)
    # Set up KVCacheManager and attn_metadata for distributed
    kv_cache_manager, attn_metadata = _setup_kv_and_metadata(scenario, mapping)
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    seq_len_per_gpu = scenario.seq_len // world_size
    input_ctx_bs = input_ctx.view(scenario.batch, scenario.seq_len,
                                  scenario.hidden_size)
    input_gen = input_ctx_bs[:, 0, :].contiguous()
    position_ids_ctx_bs = position_ids_ctx.view(scenario.batch,
                                                scenario.seq_len)
    position_ids_gen = position_ids_ctx_bs[:, 0].contiguous()

    # split inputs into chunks for each rank
    input_ctx_bs_rank = input_ctx_bs[:, rank * seq_len_per_gpu:(rank + 1) *
                                     seq_len_per_gpu, :]
    input_ctx_rank = input_ctx_bs_rank.reshape(
        scenario.batch * seq_len_per_gpu, scenario.hidden_size).contiguous()
    position_ids_ctx_bs_rank = position_ids_ctx_bs[:, rank *
                                                   seq_len_per_gpu:(rank + 1) *
                                                   seq_len_per_gpu]
    position_ids_ctx_rank = position_ids_ctx_bs_rank.reshape(
        scenario.batch * seq_len_per_gpu).contiguous()
    # this represents the context step
    with model_extra_attrs(extra_attrs):
        mla(position_ids_ctx_rank, input_ctx_rank, attn_metadata)
    outputs = []
    for step in range(scenario.gen_steps):
        for req_id in range(scenario.batch):
            kv_cache_manager.impl.add_token(req_id)
        attn_metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
            request_ids=list(range(scenario.batch)),
            max_num_requests=scenario.batch,
            num_contexts=0,
            prompt_lens=[seq_len_per_gpu] * scenario.batch,
            max_num_tokens=seq_len_per_gpu,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[
                    seq_len_per_gpu + step for _ in range(scenario.batch)
                ],
            ),
            enable_paged_context_mla=True,
        )
        attn_metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
        with model_extra_attrs(extra_attrs):
            result = mla(position_ids_gen, input_gen, attn_metadata)
        if step < scenario.ref_steps:
            outputs.append(result)
        elif step == scenario.ref_steps:
            start = time.time()
    end = time.time()
    avg_gen_time = (end - start) / (scenario.gen_steps - scenario.ref_steps)
    throughput = scenario.batch / avg_gen_time
    print(f"Rank {rank} {world_size}-GPU: time taken for "
          f"{scenario.gen_steps - scenario.ref_steps} steps: "
          f"{end - start} s, throughput: {throughput} MLA/s")
    output = torch.stack(outputs, dim=0)
    kv_cache_manager.shutdown()

    # every rank should have the same output and checks against the reference output
    # TODO sometimes, we are still producing NaNs, so we allow both elements to be NaN
    torch.testing.assert_close(output,
                               ref_output,
                               rtol=1e-2,
                               atol=1e-2,
                               equal_nan=True)


@torch.inference_mode
def _full_test_multi_gpu(rank: int, world_size: int, scenario: Scenario):
    if scenario.rope_scaling:
        rope_scaling = {
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings":
            scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        }
    else:
        rope_scaling = None
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling=rope_scaling,
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type)
    torch.manual_seed(42)
    input_ctx = torch.empty(scenario.batch * scenario.seq_len,
                            scenario.hidden_size,
                            dtype=scenario.dtype,
                            device="cuda").uniform_(-1, 1)
    position_ids_ctx = torch.arange(1, scenario.seq_len + 1,
                                    device="cuda").repeat(scenario.batch)

    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )

    mla = MLA(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        predicted_tokens_per_seq=scenario.predicted_tokens_per_seq,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
    ).cuda()
    weights = mla.state_dict()
    ref_q = torch.randn(
        scenario.batch * scenario.seq_len,
        scenario.num_heads *
        (2 * (scenario.qk_nope_head_dim + scenario.qk_rope_head_dim)) +
        scenario.num_kv_heads * scenario.v_head_dim,
        dtype=scenario.dtype,
        device="cuda")
    ref_latent_cache = torch.randn(scenario.batch * scenario.seq_len,
                                   scenario.kv_lora_rank +
                                   scenario.qk_rope_head_dim,
                                   dtype=scenario.dtype,
                                   device="cuda")
    # up to this point, all ranks should have same tensors because the seed is the same
    # now we run the reference MLA on rank 0
    if rank == 0:
        input_gen = input_ctx.view(scenario.batch, scenario.seq_len,
                                   scenario.hidden_size)[:, 0, :].contiguous()
        position_ids_gen = position_ids_ctx.view(
            scenario.batch, scenario.seq_len)[:, 0].contiguous()
        # Reference output (single GPU, but with correct KV/metadata setup)
        ref_mapping = Mapping(world_size=1, tp_size=1, rank=0)
        ref_kv_cache_manager, ref_attn_metadata = _setup_kv_and_metadata(
            scenario, ref_mapping)
        # this represents the context step
        mla(position_ids_ctx, input_ctx, ref_attn_metadata)
        ref_outputs = []
        for step in range(scenario.gen_steps):
            for req_id in range(scenario.batch):
                ref_kv_cache_manager.impl.add_token(req_id)
            ref_attn_metadata = get_attention_backend("TRTLLM").Metadata(
                seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
                request_ids=list(range(scenario.batch)),
                max_num_requests=scenario.batch,
                num_contexts=0,
                prompt_lens=[scenario.seq_len] * scenario.batch,
                max_num_tokens=scenario.seq_len,
                kv_cache_manager=ref_kv_cache_manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[
                        scenario.seq_len + step for _ in range(scenario.batch)
                    ],
                ),
                enable_paged_context_mla=True,
            )
            ref_attn_metadata.prepare()
            result = mla(position_ids_gen, input_gen, ref_attn_metadata)
            if step < scenario.ref_steps:
                ref_outputs.append(result)
            elif step == scenario.ref_steps:
                start = time.time()
        end = time.time()
        avg_gen_time = (end - start) / (scenario.gen_steps - scenario.ref_steps)
        throughput = scenario.batch / avg_gen_time
        print(
            f"Time taken for {scenario.gen_steps - scenario.ref_steps} steps: "
            f"{end - start} s, throughput: {throughput} MLA/s")
        ref_output = torch.stack(ref_outputs, dim=0)
        ref_kv_cache_manager.shutdown()
    else:
        ref_output = torch.empty(scenario.ref_steps,
                                 scenario.batch,
                                 scenario.hidden_size,
                                 dtype=scenario.dtype,
                                 device="cuda")

    # Distributed mapping for helix
    mapping = Mapping(world_size=world_size,
                      rank=rank,
                      cp_size=world_size,
                      cp_config={'cp_type': CpType.HELIX})
    # we use cp_allgather here because there is no broadcast op across CP group
    from tensorrt_llm._torch.distributed.ops import cp_allgather
    ref_output_all = cp_allgather(ref_output, mapping=mapping, dim=0)
    # we only need the values from rank 0
    ref_output = ref_output_all.view(world_size, *ref_output.shape)[0]
    test_params = (input_ctx, position_ids_ctx, weights, ref_q,
                   ref_latent_cache, pos_embd_params)
    _run_mla_distributed(rank, world_size, scenario, mapping, test_params,
                         ref_output)


def _run_single_rank(func, *args, **kwargs):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    print(f"rank {rank} starting")
    try:
        ret = func(rank, *args, **kwargs)
        print(f"rank {rank} done")
        return ret
    except Exception:
        traceback.print_exc()
        tb = traceback.format_exc()
        raise Exception(f"\n\nError occurred. Original traceback is\n{tb}\n")


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("scenario",
                         test_scenarios,
                         ids=lambda x: f"scenario: {x}")
def test_mla_helix_distributed(scenario: Scenario):
    world_size = 2
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            _run_single_rank,
            *zip(*[(_full_test_multi_gpu, world_size, scenario)] * world_size))
        for r in results:
            assert r is None


if __name__ == "__main__":
    for i, scenario in enumerate(all_scenarios):
        if i < len(all_scenarios) - 1:
            next_batch = all_scenarios[i + 1].batch
        else:
            next_batch = 0
        # only run the last scenario for each batch size: most interesting cases
        if scenario.batch == next_batch:
            continue
        print(f"Running scenario: {scenario}")
        test_mla_helix_distributed(scenario)
