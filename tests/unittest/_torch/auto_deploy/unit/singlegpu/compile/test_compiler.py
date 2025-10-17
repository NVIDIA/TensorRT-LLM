import pytest
import torch
from _model_test_utils import (
    TransformerLikeModel,
    VisionTransformerLikeModel,
    generate_dynamic_shapes,
)
from torch.nn import Module

from tensorrt_llm._torch.auto_deploy.compile import CompileBackendRegistry
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm


@pytest.mark.parametrize(
    "model_type, model_cls, input_shape, output_shape_fn",
    [
        ("llm", TransformerLikeModel, (32, 10), lambda b, s, v: (b, s, v)),
        ("vit", VisionTransformerLikeModel, (32, 4096, 16), lambda b, s, c: (b, s, c)),
    ],
)
@pytest.mark.parametrize(
    "backend_cls",
    [
        ("torch-simple"),
        ("torch-compile"),
        ("torch-cudagraph"),
        ("torch-opt"),
    ],
)
def test_compile_and_capture(model_type, model_cls, input_shape, output_shape_fn, backend_cls):
    """
    Test the `compile_and_capture` function for both LLM and ViT-like models.
    """
    batch_size, *seq_shape = input_shape

    if model_type == "llm":
        vocab_size = 100
        embed_dim = 32
        hidden_dim = 64
        mod = model_cls(vocab_size, embed_dim, hidden_dim).to("cuda")
        sample_input = torch.randint(0, vocab_size, input_shape).to("cuda")
        output_shape = output_shape_fn(batch_size, seq_shape[0], vocab_size)

    elif model_type == "vit":
        channels = 16
        hidden_dim = 64
        mod = model_cls(channels, hidden_dim).to("cuda")
        sample_input = torch.randn(*input_shape).to("cuda")
        output_shape = output_shape_fn(batch_size, seq_shape[0], channels)

    mod.eval()
    dynamic_shapes = generate_dynamic_shapes(batch_size, seq_shape[0])
    graph_module = torch_export_to_gm(mod, args=(sample_input,), dynamic_shapes=dynamic_shapes)

    with torch.inference_mode():
        compiler_cls = CompileBackendRegistry.get(backend_cls)
        compiled_model = compiler_cls(
            graph_module,
            args=(sample_input,),
            num_batched_inputs=1,
            max_batch_size=batch_size,
        ).compile()

        assert isinstance(compiled_model, Module), "Compiled model is not a valid nn.Module."
        output = compiled_model(sample_input)
        assert output is not None, "Compiled model forward pass failed."
        print("out shape", output.shape)
        assert output.shape == output_shape, "Output shape mismatch."
