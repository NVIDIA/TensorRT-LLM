import pytest
import torch
from _model_test_utils import (
    TransformerLikeModel,
    VisionTransformerLikeModel,
    generate_dynamic_shapes,
)

from tensorrt_llm._torch.auto_deploy.compile.backends.torch_opt import CompiledGraph
from tensorrt_llm._torch.auto_deploy.compile.compiler import _flatten_args
from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export_to_gm


# Using pytest.mark.parametrize to test multiple cases
@pytest.mark.parametrize(
    "lst, value, expected",
    [
        ([10, 22, 14, 7, 35, 42], 22, 22),  # Case 1: value exactly matches an element
        ([10, 22, 14, 7, 35, 42], 5, 7),  # Case 2: value is smaller than all elements
        ([10, 22, 14, 7, 35, 42], 7, 7),  # Case 3: value is exactly min element
        ([10, 22, 14, 7, 35, 42], 50, None),  # Case 4: value is larger than all elements
        ([10, 22, 14, 7, 35, 42], 15, 22),  # Case 5: value is between two elements
        ([10, 22, 14, 7, 35, 42], 42, 42),  # Case 6: value is exactly the max element
        ([10], 5, 10),  # Case 7a: single-element list with value smaller than element
        ([10], 10, 10),  # Case 7b: single-element list with value equal to element
        ([10], 15, None),  # Case 7c: single-element list with value larger than element
        ([], 15, None),  # Case 8: empty list should return None
    ],
)
def test_round_up_to_closest(lst, value, expected):
    assert CompiledGraph.round_up_to_closest(lst, value) == expected


@pytest.mark.parametrize(
    "model_type, model_cls, input_shape, captured_shape_fn, atol",
    [
        ("llm", TransformerLikeModel, (32, 10), lambda b, s: (b, s), 1e-5),
        ("vit", VisionTransformerLikeModel, (32, 4096, 16), lambda b, s, c: (b, s, c), 1e-3),
    ],
)
def test_cudagraph_capture_replay(model_type, model_cls, input_shape, captured_shape_fn, atol):
    batch_size, *seq_shape = input_shape

    if model_type == "llm":
        vocab_size = 100  # Vocabulary size
        embed_dim = 32  # Embedding dimension
        hidden_dim = 64  # Hidden layer dimension
        model = model_cls(vocab_size, embed_dim, hidden_dim).to("cuda")
        input_data = torch.randint(0, vocab_size, input_shape).to("cuda")
        captured_shape = captured_shape_fn(batch_size, seq_shape[0])

    elif model_type == "vit":
        channels = 16  # Number of channels
        hidden_dim = 64  # Hidden layer dimension
        model = model_cls(channels, hidden_dim).to("cuda")
        input_data = torch.randn(*input_shape).to("cuda")
        captured_shape = captured_shape_fn(batch_size, seq_shape[0], channels)

    model.eval()
    dynamic_shapes = generate_dynamic_shapes(batch_size, seq_shape[0])
    graph_module = torch_export_to_gm(model, args=(input_data,), dynamic_shapes=dynamic_shapes)
    compiled_model = CompiledGraph(graph_module, max_batch_size=batch_size)

    with torch.inference_mode():
        full_args = (input_data,)
        compiled_model.capture_graph(*full_args)

        # Ensure the graph is stored for the batch size
        assert captured_shape in compiled_model.graphs, "Graph for batch size was not captured."

        input_data_replay = input_data[:, :1] if model_type == "llm" else input_data[:, :1, :]

        graph = compiled_model.graphs[captured_shape]
        input_data_flatten, _ = _flatten_args(compiled_model._in_spec, input_data_replay)
        compiled_model._input_buffer[:] = input_data_flatten  # Update input buffer
        graph.replay()

        replay_output = compiled_model._out_spec.unflatten(
            [buf[:batch_size].detach().clone() for buf in compiled_model._out_buffer_flat]
        )
        replay_output2 = compiled_model.forward(input_data_replay)
        assert torch.allclose(replay_output, replay_output2, atol=atol), (
            "CUDAGraph replay output mismatch"
        )

        original_output = compiled_model.gm_compiled(input_data_replay)

        assert torch.allclose(original_output, replay_output, atol=atol), (
            "CUDAGraph replay output mismatch"
        )
