import pytest
import torch
from _model_test_utils import (
    TransformerLikeModel,
    VisionTransformerLikeModel,
    generate_dynamic_shapes,
)

from tensorrt_llm._torch.auto_deploy.compile.backends.torch_cudagraph import (
    CapturedGraph,
    _args_kwargs_flatten_spec,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm


class ModelWithMultipleInputs(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x0, x1=None, x2=None):
        out = self.base_model(x0)
        if x1 is not None:
            out = out + self.base_model(x1)
        if x2 is not None:
            out = out + self.base_model(x2)
        return out


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
    assert CapturedGraph.round_up_to_closest(lst, value) == expected


@pytest.mark.parametrize("num_inputs", [1, 2, 3])
@pytest.mark.parametrize(
    "model_type, model_cls, input_shape, atol",
    [
        ("llm", TransformerLikeModel, (32, 10), 1e-5),
        ("vit", VisionTransformerLikeModel, (32, 4096, 16), 1e-3),
    ],
)
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_cudagraph_capture_replay(
    model_type, model_cls, input_shape, atol, num_inputs, use_torch_compile
):
    batch_size, *seq_shape = input_shape

    if model_type == "llm":
        vocab_size = 100  # Vocabulary size
        embed_dim = 32  # Embedding dimension
        hidden_dim = 64  # Hidden layer dimension
        base_model = model_cls(vocab_size, embed_dim, hidden_dim).to("cuda")
        model = ModelWithMultipleInputs(base_model).to("cuda")

        # Create inputs for the model
        input_data = [
            torch.randint(0, vocab_size, input_shape).to("cuda") for _ in range(num_inputs)
        ]

    elif model_type == "vit":
        channels = 16  # Number of channels
        hidden_dim = 64  # Hidden layer dimension
        base_model = model_cls(channels, hidden_dim).to("cuda")
        model = ModelWithMultipleInputs(base_model).to("cuda")

        # Create inputs for the model
        input_data = [torch.randn(*input_shape).to("cuda") for _ in range(num_inputs)]

    combined_shape = input_shape * num_inputs

    model.eval()
    dynamic_shapes = generate_dynamic_shapes(batch_size, seq_shape[0]) * num_inputs

    # Prepare args - include only the number of inputs needed
    args = tuple(input_data[:num_inputs])
    print(args)
    print(dynamic_shapes)

    graph_module = torch_export_to_gm(model, args=args, dynamic_shapes=dynamic_shapes)

    # Apply torch.compile if needed
    if use_torch_compile:
        graph_module = torch.compile(graph_module, dynamic=True)

    compiled_model = CapturedGraph(
        graph_module,
        cuda_graph_batch_sizes=[batch_size],
        num_batched_inputs=num_inputs,
    )

    with torch.inference_mode():
        # Capture graph with all inputs
        compiled_model.capture_graph(*args)

        # Ensure the graph is stored for the combined shape of all inputs
        assert combined_shape in compiled_model.cudagraphs, (
            f"Graph for combined shape {combined_shape} was not captured."
        )

        # Create smaller inputs for replay
        if model_type == "llm":
            replay_input_data = [x[:, :1] for x in input_data[:num_inputs]]
        else:  # vit
            replay_input_data = [x[:, :1, :] for x in input_data[:num_inputs]]

        # Prepare replay args - include only the number of inputs needed
        replay_args = tuple(replay_input_data)

        # Get flat inputs for manual replay
        all_args_flat = _args_kwargs_flatten_spec(compiled_model._in_spec, *replay_args)
        input_args_flat = all_args_flat[:num_inputs]  # Extract just the batched inputs

        # Update input buffers for replay
        for i, input_tensor in enumerate(input_args_flat):
            compiled_model._input_buffers[i][: input_tensor.shape[0]] = input_tensor

        # Get the appropriate graph and replay
        graph = compiled_model.cudagraphs[combined_shape]
        graph.replay()

        # Get output from manual replay
        replay_output = compiled_model._out_spec.unflatten(
            [buf[:batch_size].detach().clone() for buf in compiled_model._out_buffer_flat]
        )

        # Get output from forward method
        replay_output2 = compiled_model.forward(*replay_args)

        # Compare outputs
        assert torch.allclose(replay_output, replay_output2, atol=atol), (
            "CUDAGraph replay output mismatch"
        )

        # Compare with original model output
        original_output = compiled_model.model(*replay_args)
        assert torch.allclose(original_output, replay_output, atol=atol), (
            "CUDAGraph replay output mismatch"
        )
