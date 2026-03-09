import operator
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from _model_test_utils import (
    TransformerLikeModel,
    VisionTransformerLikeModel,
    generate_dynamic_shapes,
)
from torch.fx import Graph, GraphModule

from tensorrt_llm._torch.auto_deploy.compile.backends.torch_cudagraph import (
    CapturedGraph,
    DualModeCapturedGraph,
    PiecewiseCapturedGraph,
    _args_kwargs_flatten_spec,
    _submod_has_cuda_ops,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import _round_up_to_closest
from tensorrt_llm._torch.auto_deploy.transform.library.compile_model import (
    _generate_default_piecewise_num_tokens,
)


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
    assert _round_up_to_closest(lst, value) == expected


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
        num_batched_inputs=num_inputs,
    )

    # Create a get_args_kwargs function for capture_graph
    def get_args_kwargs(bs):
        if model_type == "llm":
            return tuple(x[:bs] for x in input_data[:num_inputs]), {}
        else:  # vit
            return tuple(x[:bs] for x in input_data[:num_inputs]), {}

    with torch.inference_mode():
        # Capture graph with batch sizes
        compiled_model.capture_graph(get_args_kwargs, [batch_size])

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


# ============================================================================
# Helpers for piecewise / _submod_has_cuda_ops tests
# ============================================================================


def _build_trivial_graphmodule():
    """Build a GraphModule with only trivial ops (getitem, view)."""
    graph = Graph()
    x = graph.placeholder("x")
    # getitem is trivial
    item = graph.call_function(operator.getitem, args=(x, 0))
    # view is a trivial call_method
    viewed = graph.call_method("view", args=(item, -1))
    graph.output(viewed)
    root = nn.Module()
    return GraphModule(root, graph)


def _build_graphmodule_with_linear():
    """Build a GraphModule that calls a Linear submodule (has CUDA ops)."""

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    model = SmallModel()
    from torch.fx import symbolic_trace

    gm = symbolic_trace(model)
    return gm


# ============================================================================
# Tests for _submod_has_cuda_ops
# ============================================================================


class TestSubmodHasCudaOps:
    """Tests for _submod_has_cuda_ops."""

    def test_trivial_graphmodule_returns_false(self):
        gm = _build_trivial_graphmodule()
        assert _submod_has_cuda_ops(gm) is False

    def test_graphmodule_with_linear_returns_true(self):
        gm = _build_graphmodule_with_linear()
        assert _submod_has_cuda_ops(gm) is True

    def test_non_graphmodule_returns_true(self):
        """Non-FX modules are conservatively treated as having CUDA ops."""
        module = nn.Linear(4, 4)
        assert _submod_has_cuda_ops(module) is True

    def test_graphmodule_with_nontrivial_call_function(self):
        """A graph with torch.add (non-trivial call_function) should return True."""
        graph = Graph()
        x = graph.placeholder("x")
        y = graph.call_function(torch.add, args=(x, x))
        graph.output(y)
        gm = GraphModule(nn.Module(), graph)
        assert _submod_has_cuda_ops(gm) is True

    def test_graphmodule_with_nontrivial_call_method(self):
        """A graph with 'matmul' method call should return True."""
        graph = Graph()
        x = graph.placeholder("x")
        y = graph.call_method("matmul", args=(x, x))
        graph.output(y)
        gm = GraphModule(nn.Module(), graph)
        assert _submod_has_cuda_ops(gm) is True

    def test_graphmodule_with_only_trivial_methods(self):
        """A graph with only trivial call_methods should return False."""
        graph = Graph()
        x = graph.placeholder("x")
        y = graph.call_method("view", args=(x, -1))
        z = graph.call_method("contiguous", args=(y,))
        graph.output(z)
        gm = GraphModule(nn.Module(), graph)
        assert _submod_has_cuda_ops(gm) is False


# ============================================================================
# Tests for DualModeCapturedGraph routing logic
# ============================================================================


class TestDualModeCapturedGraphRouting:
    """Tests for DualModeCapturedGraph routing logic (no actual graph capture)."""

    def _make_dual_mode(self, piecewise_num_tokens=None):
        """Create a DualModeCapturedGraph with mock monolithic and piecewise."""
        if piecewise_num_tokens is None:
            piecewise_num_tokens = [64, 128, 256]

        monolithic = MagicMock(spec=nn.Module)
        monolithic.return_value = torch.tensor([1.0])

        piecewise = MagicMock(spec=PiecewiseCapturedGraph)
        piecewise.piecewise_num_tokens = piecewise_num_tokens
        piecewise.original_model = MagicMock(return_value=torch.tensor([2.0]))
        piecewise.return_value = torch.tensor([3.0])

        dual = DualModeCapturedGraph(monolithic, piecewise)
        return dual

    def test_is_decode_only_with_batch_info_host_zero(self):
        dual = self._make_dual_mode()
        # num_prefill=0 → decode-only
        batch_info = torch.tensor([0, 0, 4])  # [num_prefill, num_prefill_tokens, num_decode]
        assert dual._is_decode_only(batch_info_host=batch_info) is True

    def test_is_decode_only_with_batch_info_host_nonzero(self):
        dual = self._make_dual_mode()
        # num_prefill=2 → not decode-only
        batch_info = torch.tensor([2, 100, 3])
        assert dual._is_decode_only(batch_info_host=batch_info) is False

    def test_is_decode_only_fallback_heuristic_decode(self):
        dual = self._make_dual_mode()
        # No batch_info_host; input_ids shape [4, 1] → decode (seq_dim == 1)
        input_ids = torch.randint(0, 100, (4, 1))
        assert dual._is_decode_only(input_ids=input_ids) is True

    def test_is_decode_only_fallback_heuristic_prefill(self):
        dual = self._make_dual_mode()
        # No batch_info_host; input_ids shape [1, 128] → prefill (seq_dim > 1)
        input_ids = torch.randint(0, 100, (1, 128))
        assert dual._is_decode_only(input_ids=input_ids) is False

    def test_is_decode_only_default_no_info(self):
        dual = self._make_dual_mode()
        # No batch_info_host and no batched inputs → defaults to True
        assert dual._is_decode_only() is True

    def test_get_num_tokens_flat_layout(self):
        dual = self._make_dual_mode()
        input_ids = torch.randint(0, 100, (1, 200))
        assert dual._get_num_tokens(input_ids=input_ids) == 200

    def test_get_num_tokens_1d_layout(self):
        dual = self._make_dual_mode()
        input_ids = torch.randint(0, 100, (150,))
        assert dual._get_num_tokens(input_ids=input_ids) == 150

    def test_get_num_tokens_no_input(self):
        dual = self._make_dual_mode()
        assert dual._get_num_tokens() == 0

    @pytest.mark.parametrize(
        "num_tokens, expected_bucket",
        [
            (10, 64),
            (64, 64),
            (65, 128),
            (128, 128),
            (200, 256),
            (256, 256),
            (257, None),  # exceeds largest bucket
        ],
    )
    def test_find_nearest_bucket(self, num_tokens, expected_bucket):
        dual = self._make_dual_mode(piecewise_num_tokens=[64, 128, 256])
        assert dual._find_nearest_bucket(num_tokens) == expected_bucket

    def test_find_nearest_bucket_empty(self):
        dual = self._make_dual_mode(piecewise_num_tokens=[])
        assert dual._find_nearest_bucket(100) is None


# ============================================================================
# Tests for PiecewiseCapturedGraph.prepare
# ============================================================================


class TestPiecewiseCapturedGraphPrepare:
    """Tests for PiecewiseCapturedGraph.prepare."""

    def test_non_graphmodule_sets_split_gm_none(self):
        """When model is not a GraphModule, split_gm should remain None."""
        model = nn.Linear(4, 4)
        pcg = PiecewiseCapturedGraph(model, piecewise_num_tokens=[8, 16])
        pcg.prepare()

        assert pcg._is_prepared is True
        assert pcg.split_gm is None

    def test_prepare_is_idempotent(self):
        """Calling prepare() twice should not re-split."""
        model = nn.Linear(4, 4)
        pcg = PiecewiseCapturedGraph(model, piecewise_num_tokens=[8])
        pcg.prepare()
        pcg.prepare()  # Should be a no-op
        assert pcg._is_prepared is True


# ============================================================================
# Tests for _generate_default_piecewise_num_tokens (compile_model.py)
# ============================================================================


class TestGenerateDefaultPiecewiseNumTokens:
    """Tests for _generate_default_piecewise_num_tokens."""

    def test_power_of_two_max(self):
        result = _generate_default_piecewise_num_tokens(8192)
        assert result == [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    def test_non_power_of_two_appended(self):
        result = _generate_default_piecewise_num_tokens(100)
        assert result == [64, 100]

    def test_zero_returns_empty(self):
        result = _generate_default_piecewise_num_tokens(0)
        assert result == []

    def test_negative_returns_empty(self):
        result = _generate_default_piecewise_num_tokens(-10)
        assert result == []

    def test_exactly_64(self):
        result = _generate_default_piecewise_num_tokens(64)
        assert result == [64]

    def test_less_than_64(self):
        result = _generate_default_piecewise_num_tokens(32)
        assert result == [32]

    def test_256(self):
        result = _generate_default_piecewise_num_tokens(256)
        assert result == [64, 128, 256]

    def test_large_non_power_of_two(self):
        result = _generate_default_piecewise_num_tokens(5000)
        # Powers of 2 from 64: 64, 128, 256, 512, 1024, 2048, 4096
        # Then append 5000
        assert result == [64, 128, 256, 512, 1024, 2048, 4096, 5000]

    def test_result_is_sorted(self):
        result = _generate_default_piecewise_num_tokens(10000)
        assert result == sorted(result)

    def test_no_duplicates_when_max_is_power_of_two(self):
        result = _generate_default_piecewise_num_tokens(4096)
        # 4096 is already a power of 2, should not be duplicated
        assert result.count(4096) == 1
