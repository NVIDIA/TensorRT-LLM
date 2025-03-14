import pytest
import torch
from _model_test_utils import TransformerLikeModel, generate_dynamic_shapes
from torch.nn import Module

from tensorrt_llm._torch.auto_deploy.compile import compile_and_capture
from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export_to_gm


@pytest.mark.parametrize(
    "backend_cls",
    [
        ("torch-simple"),
        ("torch-opt"),
    ],
)
def test_compile_and_capture(backend_cls):
    """
    Test the `compile_and_capture` function with the `torch-simple` backend.
    """
    vocab_size = 100
    embed_dim = 32
    hidden_dim = 64
    seq_length = 10
    batch_size = 32

    mod = TransformerLikeModel(vocab_size, embed_dim, hidden_dim).to("cuda")
    mod.eval()

    sample_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to("cuda")  # Random input
    dynamic_shapes = generate_dynamic_shapes(batch_size, seq_length)
    graph_module = torch_export_to_gm(mod, args=(sample_input,), dynamic_shapes=dynamic_shapes)

    with torch.inference_mode():
        compiled_model = compile_and_capture(
            graph_module,
            backend_cls,
            args=(sample_input,),
            dynamic_shapes=dynamic_shapes,
        )

        assert isinstance(compiled_model, Module), "Compiled model is not a valid nn.Module."
        output = compiled_model(sample_input)
        assert output is not None, "Compiled model forward pass failed."
        print("out shape", output.shape)
        assert output.shape == (batch_size, seq_length, vocab_size), "Output shape mismatch."
