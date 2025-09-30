import contextlib
import os
import re
from typing import Any, Dict
from unittest import mock

import pytest
import torch
import transformers
import transformers.models.mistral3
from _torch.helpers import create_mock_engine
from PIL import Image
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm import mapping as mapping_lib
from tensorrt_llm._torch import metadata as metadata_lib
from tensorrt_llm._torch import model_config as model_config_lib
from tensorrt_llm._torch.attention_backend import utils as attention_utils
from tensorrt_llm._torch.models import modeling_mistral
from tensorrt_llm._torch.pyexecutor import resource_manager
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm.bindings import executor as executor_lib
from tensorrt_llm.models import modeling_utils

_PATCH_SIZE = 14


@pytest.fixture
def mistral_small_3_1_24b_config():
    return {
        "architectures": ["Mistral3ForConditionalGeneration"],
        "image_token_index": 10,
        "model_type": "mistral3",
        "multimodal_projector_bias": False,
        "projector_hidden_act": "gelu",
        "spatial_merge_size": 2,
        "text_config": {
            "attention_dropout": 0.0,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 32768,
            "max_position_embeddings": 131072,
            "model_type": "mistral",
            "num_attention_heads": 32,
            # Reduce this from the original 40 to relieve memory needs for CI.
            "num_hidden_layers": 4,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000000.0,
            "sliding_window": None,
            "use_cache": True,
            "vocab_size": 131072,
        },
        "torch_dtype": "bfloat16",
        "transformers_version": "4.50.0.dev0",
        "vision_config": {
            "attention_dropout": 0.0,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "image_size": 1540,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "model_type": "pixtral",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 24,
            "patch_size": _PATCH_SIZE,
            "rope_theta": 10000.0,
        },
        "vision_feature_layer": -1,
    }


def reduce_mistral_config(
    mem_for_full_model: int, config_dict: Dict[str, Any], default_num_layers: int = 4
):
    _, total_mem = torch.cuda.mem_get_info()
    if "text_config" in config_dict:
        config_dict = config_dict["text_config"]
    # scale model down if gpu memory is low
    if total_mem < mem_for_full_model:
        model_fraction = total_mem / mem_for_full_model
        num_layers = int(config_dict["num_hidden_layers"] * model_fraction)
        num_layers = min(num_layers, default_num_layers)
        config_dict["num_hidden_layers"] = num_layers


def init_hf_model(cls, config, dtype, device):
    """Helper function for initializing a model from `transformers`.

    The reason this function exists is: by default, instantiating a `transformers` model also
    eagerly initializes the model's weights on the CPU, which takes an absurdly long time to
    complete.

    Instead, we lazily instantiate the model, and initialize the weights only after moving it to
    the requested `device`.
    """
    from transformers import modeling_utils as t_modeling_utils

    with t_modeling_utils.no_init_weights():
        model = cls(config).eval()

    model.to(device=device)
    model.init_weights()
    model.to(dtype=dtype)

    return model


def convert_weights_names(weights: dict) -> dict:
    # Since transformers version >= 4.52.0, the default model architecture is changed.
    # We need to convert the weight names accordingly to match TRTLLM naming.
    _checkpoint_conversion_mapping = {
        "^model.language_model": "language_model.model",
        "^model.vision_tower": "vision_tower",
        "^model.multi_modal_projector": "multi_modal_projector",
        "^lm_head": "language_model.lm_head",
    }
    converted_weights = {}
    for weight_name, weight_value in weights.items():
        new_name = weight_name
        for pattern, replacement in _checkpoint_conversion_mapping.items():
            new_name = re.sub(pattern, replacement, new_name)
        converted_weights[new_name] = weight_value
    return converted_weights


@pytest.fixture(autouse=True)
def empty_cuda_cache():
    torch.cuda.empty_cache()


@contextlib.contextmanager
def kv_cache_manager_context(kv_cache_manager):
    try:
        yield
    finally:
        kv_cache_manager.shutdown()


def test_mistral_3_vlm_rejects_disagg(mistral_small_3_1_24b_config):
    with (
        mock.patch.dict(os.environ, {"TLLM_MULTIMODAL_DISAGGREGATED": "1"}),
        pytest.raises(NotImplementedError, match="disaggregated inference"),
    ):
        modeling_mistral.Mistral3VLM(
            model_config=model_config_lib.ModelConfig(
                pretrained_config=transformers.Mistral3Config.from_dict(
                    mistral_small_3_1_24b_config
                )
            ),
        )


@pytest.mark.parametrize("quant_algo", [None, "FP8"])
def test_mistral_3_vlm_sanity(mistral_small_3_1_24b_config, quant_algo):
    if quant_algo == "FP8" and getSMVersion() < 89:
        pytest.skip("This test is not supported in pre-Ada architecture")

    config_dict = mistral_small_3_1_24b_config
    # 24B * sizeof(float16) plus some extra for activations
    mem_for_full_model = int(2.1 * 24 * 2 ** (30))
    reduce_mistral_config(mem_for_full_model, config_dict)

    if config_dict["text_config"]["num_hidden_layers"] <= 0:
        pytest.skip("Insufficient memory for a single Mistral layer")

    mistral_3_config = transformers.Mistral3Config.from_dict(config_dict)
    if quant_algo:
        quant_config = modeling_utils.QuantConfig(quant_algo=quant_algo)
    else:
        quant_config = None

    dtype = mistral_3_config.torch_dtype
    device = torch.device("cuda")

    model_config = model_config_lib.ModelConfig(
        pretrained_config=mistral_3_config,
        quant_config=quant_config,
    )
    mistral = modeling_mistral.Mistral3VLM(model_config).to(device)

    input_ids = torch.tensor(
        [100, 200, 300, 100, 200, 100, 400, 500], dtype=torch.int, device=device
    )

    context_sequence_lengths = [3, 2, 1]
    sequence_lengths = context_sequence_lengths + [1, 1]
    past_seen_tokens = [0, 0, 0, 62, 75]
    request_ids = list(range(len(sequence_lengths)))
    token_nums = (torch.tensor(past_seen_tokens) + torch.tensor(sequence_lengths)).tolist()
    prompt_lens = token_nums[:3] + past_seen_tokens[3:]

    num_blocks = 100
    tokens_per_block = 128
    head_dim = mistral.config.head_dim
    num_layers = mistral.config.num_hidden_layers
    num_kv_heads = mistral.config.num_key_value_heads
    max_seq_len = num_blocks * tokens_per_block
    batch_size = len(context_sequence_lengths) + 2

    if dtype == torch.half:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")

    mapping = mapping_lib.Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_config = executor_lib.KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
    kv_cache_manager = resource_manager.KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )
    with kv_cache_manager_context(kv_cache_manager):
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = attention_utils.get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor(sequence_lengths, dtype=torch.int),
            num_contexts=len(context_sequence_lengths),
            kv_cache_params=metadata_lib.KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=past_seen_tokens,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=len(context_sequence_lengths) + 2,
            max_num_tokens=8192,
        )

        position_ids = []
        for i, tokens in enumerate(past_seen_tokens):
            seq_len = context_sequence_lengths[i] if i < len(context_sequence_lengths) else 1
            position_id = torch.arange(tokens, tokens + seq_len, device=input_ids.device)
            position_ids.append(position_id)

        position_ids = torch.cat(position_ids).unsqueeze(0)

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


@pytest.mark.parametrize(
    "backend, use_cuda_graph",
    [
        ("VANILLA", False),
        ("FLASHINFER", False),
        ("FLASHINFER", True),
        ("TRTLLM", False),
        ("TRTLLM", True),
    ],
)
@torch.no_grad()
def test_mistral_3_vlm_allclose_to_hf(mistral_small_3_1_24b_config, backend, use_cuda_graph):
    metadata_cls = attention_utils.get_attention_backend(backend).Metadata

    torch.random.manual_seed(0)
    config_dict = mistral_small_3_1_24b_config
    # 24B * sizeof(float16) plus some extra for activations
    # times 2, since we'll need 2 of these
    mem_for_full_model = int(2.1 * 24 * 2 ** (30) * 2)
    reduce_mistral_config(mem_for_full_model, config_dict)
    if config_dict["text_config"]["num_hidden_layers"] <= 0:
        pytest.skip("Insufficient memory for a single Mistral layer")
    mistral_config = transformers.Mistral3Config.from_dict(config_dict)
    dtype = mistral_config.torch_dtype
    device = torch.device("cuda")

    hf_mistral = init_hf_model(
        cls=transformers.Mistral3ForConditionalGeneration,
        config=mistral_config,
        dtype=dtype,
        device=device,
    )

    model_config = model_config_lib.ModelConfig(
        pretrained_config=mistral_config,
        attn_backend=backend,
    )
    mistral = modeling_mistral.Mistral3VLM(model_config).to(dtype).to(device)
    mistral.load_weights(convert_weights_names(hf_mistral.state_dict()))

    num_blocks = 1
    tokens_per_block = 128
    head_dim = mistral.config.head_dim
    num_layers = mistral.config.num_hidden_layers
    num_kv_heads = mistral.config.num_key_value_heads
    max_seq_len = num_blocks * tokens_per_block
    batch_size = 1

    if dtype == torch.half:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")

    mapping = mapping_lib.Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_config = executor_lib.KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
    kv_cache_manager = resource_manager.KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    with kv_cache_manager_context(kv_cache_manager):
        # context
        input_ids = torch.tensor(
            [100, 200, 300, 100, 200, 100, 400, 500], dtype=torch.int, device=device
        )

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

        # Note: no CUDA graphs for prefill, the graph runner is built for
        # decoding only.
        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = mistral.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )
            ref = hf_mistral.forward(
                input_ids=input_ids.unsqueeze(0), position_ids=position_ids, use_cache=True
            )

        torch.testing.assert_close(logits, ref.logits[:, -1].float(), atol=0.4, rtol=0.4)

        # gen
        gen_input_ids = torch.tensor([600], dtype=torch.int, device=device)

        num_cached_tokens_per_seq = [input_ids.size(-1)]

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([gen_input_ids.size(-1)], dtype=torch.int),
            num_contexts=0,
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

        gen_position_ids = [
            torch.arange(input_ids.size(-1), input_ids.size(-1) + gen_input_ids.size(-1))
        ]
        gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()

        graph_runner = None
        if use_cuda_graph:
            mock_engine = create_mock_engine(1)
            graph_runner = CUDAGraphRunner(mock_engine)

        def run_forward(input_ids, position_ids, attn_metadata):
            attn_metadata.prepare()
            if not use_cuda_graph:
                return mistral.forward(
                    input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
                )
            else:
                inputs = {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attn_metadata": attn_metadata,
                }
                key = (1, 0, False)
                graph_runner.capture(key, lambda inputs: mistral.forward(**inputs), inputs)

                for _ in range(2):
                    # Run it twice. This helps us catch problems if buffers are accidentally reallocated
                    # in prepare().
                    attn_metadata.prepare()
                    logits = graph_runner.replay(key, inputs)
                return logits

        if use_cuda_graph:
            attn_metadata = attn_metadata.create_cuda_graph_metadata(1)

        with torch.inference_mode():
            logits = run_forward(
                input_ids=gen_input_ids, position_ids=gen_position_ids, attn_metadata=attn_metadata
            )
            ref = hf_mistral.forward(
                input_ids=gen_input_ids.unsqueeze(0),
                position_ids=gen_position_ids,
                past_key_values=ref.past_key_values,
                use_cache=True,
            )

        torch.testing.assert_close(logits, ref.logits[:, -1].float(), atol=0.4, rtol=0.4)
        if graph_runner is not None:
            graph_runner.clear()


@pytest.mark.parametrize(
    "in_shapes, image_sizes, expected_out_shape",
    [
        (
            [(2, 3, 100, 150), (1, 3, 200, 100), (3, 3, 120, 180)],
            [
                [[92, 150], [100, 73]],
                [[200, 100]],
                [[37, 130], [120, 83], [73, 180]],
            ],
            [6, 3, 200, 180],
        ),
        # Single batch, single image.
        (
            [(1, 3, 64, 128)],
            [[[64, 128]]],
            [1, 3, 64, 128],
        ),
        # Same max size across batches.
        (
            [(2, 3, 59, 59), (1, 3, 59, 59), (5, 3, 59, 59)],
            [
                [[13, 59], [59, 17]],
                [[59, 59]],
                [[19, 29], [59, 31], [17, 54], [13, 59], [11, 37]],
            ],
            [8, 3, 59, 59],
        ),
    ],
)
def test_batch_pixel_values(in_shapes, image_sizes, expected_out_shape):
    # Test case 1: Basic functionality with different sized images
    pixel_values = [torch.randn(*shape) for shape in in_shapes]
    image_sizes = [torch.tensor(size) for size in image_sizes]

    batched_pixels, batched_sizes = modeling_mistral.Mistral3VLM.batch_pixel_values(
        pixel_values, image_sizes
    )

    # Check output shapes
    assert list(batched_pixels.shape) == expected_out_shape
    assert list(batched_sizes.shape) == [expected_out_shape[0], 2]

    # Check that the original image data is preserved (with padding).
    start_idx = 0
    for original_values in pixel_values:
        batch_size = original_values.shape[0]
        end_idx = start_idx + batch_size
        orig_h, orig_w = original_values.shape[-2:]
        padded_values = batched_pixels[start_idx:end_idx, :, :orig_h, :orig_w]
        torch.testing.assert_close(padded_values, original_values)

        start_idx += batch_size


@pytest.mark.parametrize("height, width", [(37, 91), (128, 256), (512, 512)])
@torch.no_grad()
def test_processor_get_num_tokens_per_image(
    tmp_path,
    mistral_small_3_1_24b_config,
    height,
    width,
):
    # NOTES:
    # 1. Setting up a fake model checkpoint that `AutoTokenizer.from_pretrained` /
    #    `AutoProcessor.from_pretrained` is too involved (need at least several config JSONs, as well
    #    as tokenizer files like vocab files, etc.).
    # 2. On the other hand, using an actual model checkpoint for what should be a simple unit test
    #    feels overkill.
    # We therefore settle for this intermediate approach where we check that the expected calls are
    # are made by mocking the auto classes out.

    mistral_3_config = transformers.Mistral3Config.from_dict(mistral_small_3_1_24b_config)

    with mock.patch(
        "tensorrt_llm._torch.models.modeling_mistral.AutoProcessor"
    ) as mocked_auto_processor:
        input_processor = modeling_mistral.Mistral3InputProcessor(
            model_path=str(tmp_path),
            model_config=mistral_3_config,
            tokenizer=mock.MagicMock(),
        )

    input_processor.get_num_tokens_per_image(
        image=Image.new("RGB", (width, height), color=(255, 128, 0))
    )

    mocked_auto_processor.from_pretrained.return_value._get_num_multimodal_tokens.assert_called_once_with(
        [(height, width)]
    )
