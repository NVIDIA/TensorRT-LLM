import unittest

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig

# isort: off
from tensorrt_llm._torch.models.modeling_nemotron_h import (NemotronHConfig,
                                                            NemotronHForCausalLM
                                                            )
# isort: on
from transformers import AutoTokenizer
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than

from tensorrt_llm._torch.pyexecutor.model_engine import load_weights
from tensorrt_llm._torch.pyexecutor.resource_manager import \
    MambaHybridCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping


def get_logprobs(token_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    raw_probs = torch.softmax(logits, dim=-1)
    index = token_ids.unsqueeze(1).cuda()
    token_probs = torch.gather(raw_probs, dim=1, index=index).squeeze(-1)
    return torch.log(token_probs)


class TestNemotronH(unittest.TestCase):

    @skip_gpu_memory_less_than(
        (2 * 8 + 1) * 2**30)  # 8B, bf16, plus 1 GB for good measure
    def test_nemotron_correctness(self):
        model_dir = f"{llm_models_root(check=True)}/Nemotron-H-8B-Base-8K"
        nemotron_h_config = NemotronHConfig.from_pretrained(model_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        dtype = nemotron_h_config.torch_dtype
        device = torch.device('cuda')
        assert dtype == torch.bfloat16
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16

        model_config = ModelConfig(pretrained_config=nemotron_h_config)
        nemotron_h = NemotronHForCausalLM(model_config).to(device)

        weights = load_weights(model_dir)
        nemotron_h.load_weights(weights)

        # These tokens are from the sentence: The future of AI is
        input_ids = torch.tensor([1784, 7147, 1307, 26554, 1395],
                                 dtype=torch.int64,
                                 device=device)

        num_cached_tokens_per_seq = [0]
        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]

        num_blocks = 100
        tokens_per_block = 128
        head_dim = nemotron_h.config.hidden_size // nemotron_h.config.num_attention_heads
        num_layers = nemotron_h.config.hybrid_override_pattern.count("*")
        mamba_num_layers = nemotron_h.config.hybrid_override_pattern.count("M")
        num_kv_heads = nemotron_h.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        max_batch_size = 1

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block,
                                        enable_block_reuse=False)
        kv_cache_manager = MambaHybridCacheManager(
            # mamba cache parameters
            nemotron_h.config.hidden_size,
            nemotron_h.config.ssm_state_size,
            nemotron_h.config.conv_kernel,
            nemotron_h.config.expand,
            nemotron_h.config.n_groups,
            nemotron_h.config.mamba_head_dim,
            mamba_num_layers,
            nemotron_h.config.torch_dtype,
            # kv cache parameters
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
            num_extra_kv_tokens=0,
        )
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)
        kv_cache_manager.prepare_mamba_cache_blocks(request_ids)

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )

        # prefill
        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = nemotron_h.forward(input_ids=input_ids,
                                        position_ids=position_ids,
                                        attn_metadata=attn_metadata,
                                        return_context_logits=True)

        # compute logprobs from logits
        prefill_logprobs = get_logprobs(input_ids[1:], logits)

        # reference logprobs from mcore
        prefill_logprobs_ref = torch.tensor([
            -7.415980815887451, -0.36192911863327026, -2.8658294677734375,
            -2.316344738006592
        ],
                                            device=device)

        # reference logprobs from initial implementation
        prefill_logprobs_ref_initial = torch.tensor([
            -7.4359540939331055, -0.37661877274513245, -2.8925108909606934,
            -2.268364906311035
        ],
                                                    device=device)

        # compare logprobs with mcore logprobs, check that the max error is less than 0.3
        torch.testing.assert_close(prefill_logprobs,
                                   prefill_logprobs_ref,
                                   atol=0.3,
                                   rtol=0.0)

        # compare logprobs with initial implementation, check that the max error is less than 0.1
        torch.testing.assert_close(prefill_logprobs,
                                   prefill_logprobs_ref_initial,
                                   atol=0.1,
                                   rtol=0.0)

        print("#" * 40 + " prefill logprobs error " + "#" * 40)
        print("            prefill_logprobs:", prefill_logprobs.cpu().numpy())
        print("        prefill_logprobs_ref:",
              prefill_logprobs_ref.cpu().numpy())
        print("prefill_logprobs_ref_initial:",
              prefill_logprobs_ref_initial.cpu().numpy())
        print(
            f"  max error mcore: {torch.max(torch.abs(prefill_logprobs - prefill_logprobs_ref)).item():.5f}"
        )
        print(
            f"max error initial: {torch.max(torch.abs(prefill_logprobs - prefill_logprobs_ref_initial)).item():.5f}"
        )
        print()

        # output tokens
        output = []
        decode_logprobs = []

        # sample token greedily
        sampled_tokens = torch.argmax(logits[-1]).unsqueeze(0)
        output.append(sampled_tokens)

        decode_logprobs.append(get_logprobs(sampled_tokens, logits[-1:]))

        # generate 8 more tokens
        max_tokens = 8
        for i in range(max_tokens):

            num_cached_tokens_per_seq = input_ids.shape[0] + i + 1

            attn_metadata = metadata_cls(
                seq_lens=torch.tensor([1], dtype=torch.int),
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=num_cached_tokens_per_seq,
                ),
                max_num_requests=1,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
                prompt_lens=prompt_lens,
            )

            with torch.inference_mode():
                attn_metadata.prepare()
                logits = nemotron_h.forward(input_ids=sampled_tokens,
                                            position_ids=position_ids,
                                            attn_metadata=attn_metadata)

            sampled_tokens = torch.argmax(logits).unsqueeze(0)
            output.append(sampled_tokens)

            decode_logprobs.append(get_logprobs(sampled_tokens, logits))

        output = torch.cat(output)
        decode_logprobs = torch.cat(decode_logprobs)
        completion = tokenizer.decode(output)

        decode_logprobs_ref_initial = torch.tensor([
            -2.2722280025482178, -0.5235245823860168, -0.8821321725845337,
            -1.9436249732971191, -0.07366813719272614, -0.4224405586719513,
            -0.3872227966785431, -0.0612114779651165, -1.0475994348526
        ],
                                                   device=device)

        self.assertEqual(
            output.tolist(),
            [18168, 1044, 1454, 58096, 32975, 1394, 32492, 1321, 6762])
        self.assertEqual(
            completion,
            " bright, with endless possibilities for innovation and growth")

        # compare logprobs with initial implementation, check that the max error is less than 0.1
        torch.testing.assert_close(decode_logprobs,
                                   decode_logprobs_ref_initial,
                                   atol=0.1,
                                   rtol=0.0)

        print("#" * 40 + " completion " + "#" * 40)
        print("completion ids:", output.tolist())
        print("sequence:", f"The future of AI is{completion}")
        print("            decode_logprobs:", decode_logprobs.cpu().numpy())
        print("decode_logprobs_ref_initial:",
              decode_logprobs_ref_initial.cpu().numpy())
        print(
            f"max decode error initial: {torch.max(torch.abs(decode_logprobs - decode_logprobs_ref_initial)).item():.5f}"
        )
        print()

        # now let's test that decodes match prefill logprobs

        input_ids = torch.cat([input_ids, output])
        prefill_decode_logprobs = torch.cat([prefill_logprobs, decode_logprobs])
        num_cached_tokens_per_seq = [0]
        request_ids = [1]
        prompt_lens = [input_ids.size(-1)]

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )

        # prefill
        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = nemotron_h.forward(input_ids=input_ids,
                                        position_ids=position_ids,
                                        attn_metadata=attn_metadata,
                                        return_context_logits=True)

        # compute logprobs from logits
        full_sequence_prefill_logprobs = get_logprobs(input_ids[1:], logits)

        # compare full sequence prefill logprobs with prefill + decode logprobs, check that the max error is less than 0.3
        torch.testing.assert_close(full_sequence_prefill_logprobs,
                                   prefill_decode_logprobs,
                                   atol=0.3,
                                   rtol=0.0)

        print("#" * 40 + " prefill vs decode logprobs " + "#" * 40)
        print("full_sequence_prefill_logprobs:",
              full_sequence_prefill_logprobs.cpu().numpy())
        print("       prefill_decode_logprobs:",
              prefill_decode_logprobs.cpu().numpy())
        print(
            f"max error: {torch.max(torch.abs(full_sequence_prefill_logprobs - prefill_decode_logprobs)).item():.5f}"
        )

        kv_cache_manager.shutdown()
