import os
import sys
import tempfile
import unittest

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (CudaGraphConfig, KvCacheConfig,
                                 SaveHiddenStatesDecodingConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_multi_save_state():
    use_cuda_graph = True
    attn_backend = "TRTLLM"
    disable_overlap_scheduler = False
    enable_block_reuse = False
    enable_chunked_prefill = False
    layers_to_capture = {10, 11, 12}

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 80:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    with tempfile.TemporaryDirectory() as temp_dir:

        target_model_dir = f"{models_path}/llama-3.2-models/Llama-3.2-1B-Instruct"

        max_batch_size = 16
        kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                        free_gpu_memory_fraction=0.5)
        cuda_graph_config = CudaGraphConfig(
            batch_sizes=[1, 2, 4]) if use_cuda_graph else None

        llm_common_config = dict(
            model=target_model_dir,
            attn_backend=attn_backend,
            disable_overlap_scheduler=disable_overlap_scheduler,
            cuda_graph_config=cuda_graph_config,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            enable_chunked_prefill=enable_chunked_prefill,
        )
        spec_config = SaveHiddenStatesDecodingConfig(
            output_directory=temp_dir,
            write_interval=1,
            file_prefix="data",
            eagle3_layers_to_capture=layers_to_capture)

        llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

        tok_ids = llm_spec.tokenizer.encode("The future of AI is")

        sampling_params = SamplingParams(max_tokens=32, temperature=0)
        for output in llm_spec.generate_async(tok_ids,
                                              sampling_params,
                                              streaming=True):
            pass
        llm_spec.shutdown()
        assert os.path.exists(os.path.join(temp_dir, "data_1.pt"))
        # Read in .pt file
        saved_data = torch.load(os.path.join(temp_dir, "data_1.pt"))[0]

        assert saved_data["aux_hidden_states"].shape == (len(tok_ids), 2048 *
                                                         len(layers_to_capture))
        assert saved_data["hidden_state"].shape == (len(tok_ids), 2048)
        assert saved_data["input_ids"].tolist() == tok_ids


@pytest.mark.parametrize("layers_to_capture", [{-1}, None])
def test_save_state(layers_to_capture):
    use_cuda_graph = True
    attn_backend = "TRTLLM"
    disable_overlap_scheduler = False
    enable_block_reuse = False
    enable_chunked_prefill = False

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 80:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    with tempfile.TemporaryDirectory() as temp_dir:

        target_model_dir = f"{models_path}/llama-3.2-models/Llama-3.2-1B-Instruct"

        max_batch_size = 16
        kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                        free_gpu_memory_fraction=0.5)
        cuda_graph_config = CudaGraphConfig(
            batch_sizes=[1, 2, 4]) if use_cuda_graph else None

        llm_common_config = dict(
            model=target_model_dir,
            attn_backend=attn_backend,
            disable_overlap_scheduler=disable_overlap_scheduler,
            cuda_graph_config=cuda_graph_config,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            enable_chunked_prefill=enable_chunked_prefill,
        )
        spec_config = SaveHiddenStatesDecodingConfig(
            output_directory=temp_dir,
            write_interval=1,
            file_prefix="data",
            eagle3_layers_to_capture=layers_to_capture)

        llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

        tok_ids = llm_spec.tokenizer.encode("The future of AI is")

        sampling_params = SamplingParams(max_tokens=32, temperature=0)
        for output in llm_spec.generate_async(tok_ids,
                                              sampling_params,
                                              streaming=True):
            pass
        llm_spec.shutdown()
        assert os.path.exists(os.path.join(temp_dir, "data_1.pt"))
        # Read in .pt file
        saved_data = torch.load(os.path.join(temp_dir, "data_1.pt"))[0]
        if layers_to_capture is None:
            assert saved_data["aux_hidden_states"].shape == (len(tok_ids),
                                                             2048 * 3)
            assert saved_data["hidden_state"].shape == (len(tok_ids), 2048)
            assert saved_data["input_ids"].tolist() == tok_ids
        else:
            assert "aux_hidden_states" not in saved_data
            assert saved_data["hidden_state"].shape == (len(tok_ids), 2048)
            assert saved_data["input_ids"].tolist() == tok_ids


if __name__ == "__main__":
    unittest.main()
