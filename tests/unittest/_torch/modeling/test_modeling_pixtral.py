import contextlib

import pytest
import torch
import transformers
from transformers.models.pixtral import modeling_pixtral as hf_modeling_pixtral
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm import mapping as mapping_lib
from tensorrt_llm._torch import metadata as metadata_lib
from tensorrt_llm._torch import model_config as model_config_lib
from tensorrt_llm._torch.attention_backend import utils as attention_utils
from tensorrt_llm._torch.models import modeling_pixtral
from tensorrt_llm._torch.pyexecutor import resource_manager
from tensorrt_llm.bindings import executor as executor_lib
from tensorrt_llm.models import modeling_utils


@pytest.fixture
def pixtral_vision_config():
    # Values taken from:
    # https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json
    return model_config_lib.ModelConfig(
        pretrained_config=transformers.PixtralVisionConfig(
            hidden_size=1024,
            num_attention_heads=16,
            torch_dtype=torch.bfloat16,
        ))


def _load_attention_weights(hf_pixtral_attention, trtllm_pixtral_attention):
    weights = hf_pixtral_attention.state_dict()

    concated_weights = {
        "qkv_proj.weight":
        torch.cat(
            [
                weights.pop("q_proj.weight"),
                weights.pop("k_proj.weight"),
                weights.pop("v_proj.weight"),
            ],
            dim=0,
        ),
        **weights
    }

    trtllm_pixtral_attention.load_state_dict(concated_weights)


@contextlib.contextmanager
def kv_cache_manager_context(kv_cache_manager):
    try:
        yield
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skip(
    reason=
    "Figure out what the expected inputs / outputs should be for each forward method."
)
@pytest.mark.parametrize("quant_algo", [None, "FP8"])
def test_pixtral_attention(pixtral_vision_config, quant_algo):
    if quant_algo == "FP8" and getSMVersion() < 89:
        pytest.skip("This test is not supported in pre-Ada architecture")

    if quant_algo:
        quant_config = modeling_utils.QuantConfig(quant_algo=quant_algo)
    else:
        pass

    dtype = torch.bfloat16
    device = torch.device("cuda")
    pretrained_config = pixtral_vision_config.pretrained_config

    num_blocks = 1
    tokens_per_block = 128
    pretrained_config.head_dim
    pretrained_config.num_hidden_layers
    max_seq_len = num_blocks * tokens_per_block
    batch_size = 1

    pixtral_attention = modeling_pixtral.PixtralAttention(
        model_config=pixtral_vision_config, layer_idx=0)
    hf_pixtral_attention = hf_modeling_pixtral.PixtralAttention(
        pretrained_config)
    _load_attention_weights(hf_pixtral_attention, pixtral_attention)

    if dtype == torch.half:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")

    mapping = mapping_lib.Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_config = executor_lib.KvCacheConfig(max_tokens=num_blocks *
                                                 tokens_per_block)
    kv_cache_manager = resource_manager.KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        # num_layers=num_layers,
        num_layers=1,
        # num_kv_heads=num_kv_heads,
        num_kv_heads=1,
        head_dim=pretrained_config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    metadata_cls = attention_utils.get_attention_backend("TRTLLM").Metadata
    with kv_cache_manager_context(kv_cache_manager):
        # context
        input_ids = torch.tensor([100, 200, 300, 100, 200, 100, 400, 500],
                                 dtype=torch.int,
                                 device=device)

        num_cached_tokens_per_seq = [0]
        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=metadata_lib.KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )

        position_ids = [torch.arange(0, input_ids.size(-1), device=device)]
        position_ids = torch.cat(position_ids).unsqueeze(0)
        hidden_states = torch.ones(
            batch_size,
            pretrained_config.hidden_size,
            dtype=dtype,
        )
        with torch.inference_mode():
            attn_metadata.prepare()
            out = pixtral_attention(position_ids=position_ids,
                                    hidden_states=hidden_states,
                                    attn_metadata=attn_metadata)
            hf_out = hf_pixtral_attention(
                hidden_states=hidden_states,
                position_embeddings=position_ids,
            )
        """
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = mistral.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )

        assert len(past_seen_tokens) == logits.shape[0]

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = mistral.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                return_context_logits=True,
            )


        assert input_ids.shape == logits.shape[:-1]
        """
