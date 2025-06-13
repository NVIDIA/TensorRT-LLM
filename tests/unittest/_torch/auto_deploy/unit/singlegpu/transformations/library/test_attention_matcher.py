from typing import Callable

import pytest
import torch
from _graph_test_helpers import run_test
from torch.export import Dim
from transformers.integrations.sdpa_attention import repeat_kv as hf_repeat_kv

from tensorrt_llm._torch.auto_deploy.transformations.library.attention import (
    match_attention_layout,
    match_causal_attn_mask,
    match_eager_attention,
    match_grouped_attention,
    match_repeat_kv,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

torch.manual_seed(0)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _repeat_kv2(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states.unsqueeze(2).expand(
        batch_size, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)


def _repeat_kv3(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    return _repeat_kv2(hidden_states, n_rep).contiguous()


class RepeatKVModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        # Define linear layers
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        return _repeat_kv(x, n_rep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Generate q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply repeat_kv pattern manually (this is what we want to detect and optimize)
        k = self._repeat_kv(k, self.num_heads // self.num_kv_heads)
        v = self._repeat_kv(v, self.num_heads // self.num_kv_heads)

        # Simple concatenation to return a result
        output = torch.cat([q, k, v], dim=1)
        return output

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", min=4, max=16)}


class RepeatKVModel2(RepeatKVModel):
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        return _repeat_kv2(x, n_rep)


class HFRepeatKVModel(RepeatKVModel):
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        return hf_repeat_kv(x, n_rep)


class RepeatKVModel3(RepeatKVModel):
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        return _repeat_kv3(x, n_rep)


class EagerAttentionModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        has_mask: bool = False,
        dropout: float = 0.0,
        use_division: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.has_mask = has_mask
        self.dropout = dropout
        self.use_division = use_division

        # Define the scaling approach based on use_division flag
        if use_division:
            # For division pattern, store the inverse of the scaling factor
            self.inv_scaling = self.head_dim**0.5  # sqrt(head_dim)
        else:
            # For multiplication pattern, store the scaling directly
            self.scaling = 1.0 / (self.head_dim**0.5)  # 1/sqrt(head_dim)

        # Define linear layers
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.out_proj = torch.nn.Linear(num_heads * self.head_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Generate q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores using the eager attention pattern
        # This is the pattern we want to detect and replace with SDPA
        if self.use_division:
            # Division pattern
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / self.inv_scaling
        else:
            # Multiplication pattern
            attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        # Add attention mask if enabled
        if self.has_mask:
            # Create a simple causal mask for testing - make sure all tensors are on the same device
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            attn_mask = torch.zeros_like(attn_weights, device=device)
            attn_mask = attn_mask.masked_fill(mask, float("-inf"))
            attn_weights = attn_weights + attn_mask

        # Apply softmax, dtype conversion, and dropout
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(dtype)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape for output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", min=4, max=16)}


class ComplexEagerAttentionModel(torch.nn.Module):
    """
    A model that implements a complex eager attention pattern similar to the one in the user's graph.
    This includes additional to_dtype operations and different transpose patterns.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        has_mask: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.has_mask = has_mask
        self.dropout = dropout

        # Use a division for scaling instead of multiplication
        self.scale_divisor = self.head_dim**0.5  # sqrt(head_dim)

        # Define linear layers
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.out_proj = torch.nn.Linear(num_heads * self.head_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Generate q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use a standard transpose that will work correctly for matmul
        # We need the dimensions to align for the matrix multiplication
        k_transposed = k.transpose(2, 3)

        # Compute attention scores using division for scaling instead of multiplication
        attn_weights = torch.matmul(q, k_transposed) / self.scale_divisor

        # Add attention mask if enabled
        if self.has_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask = mask.unsqueeze(0).unsqueeze(0)
            attn_mask = torch.zeros_like(attn_weights, device=device)
            attn_mask = attn_mask.masked_fill(mask, float("-inf"))
            attn_weights = attn_weights + attn_mask

        # Add a to_dtype node before softmax to match pattern in the graph
        attn_weights = attn_weights.to(torch.float32)

        # Apply softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Add a to_dtype node after softmax to match pattern in the graph
        attn_weights = attn_weights.to(dtype)

        # Apply dropout
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape for output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", min=4, max=16)}


class CounterExampleModel(torch.nn.Module):
    """
    A model with similar operations (unsqueeze -> expand -> reshape) but with different
    dimensions that shouldn't match the repeat_kv pattern.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Define linear layers
        self.proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Generate feature tensor
        features = self.proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Case 1: Unsqueeze at dimension 1 instead of dimension 2
        # This shouldn't match because our repeat_kv pattern specifically looks for unsqueeze at dim 2
        features_case1 = features.transpose(1, 2)  # [batch, heads, seq, dim]
        features_case1 = features_case1.unsqueeze(1)  # Different dimension than repeat_kv pattern
        features_case1 = features_case1.expand(
            batch_size, 2, self.num_heads, seq_len, self.head_dim
        )
        features_case1 = features_case1.reshape(
            batch_size, 2 * self.num_heads, seq_len, self.head_dim
        )

        # Case 2: Same operations but with unsqueeze dimension 3 instead of 2
        # This shouldn't match because repeat_kv pattern looks for unsqueeze at dim 2
        features_case2 = features.transpose(1, 2)  # [batch, heads, seq, dim]
        features_case2 = features_case2.unsqueeze(3)  # Different dimension than repeat_kv pattern
        features_case2 = features_case2.expand(
            batch_size, self.num_heads, seq_len, 2, self.head_dim
        )
        features_case2 = features_case2.reshape(
            batch_size, self.num_heads, seq_len, 2 * self.head_dim
        )
        features_case2 = features_case2.permute(0, 1, 3, 2)  # [batch, heads, 2*dim, seq]

        # Return the first case for simplicity - both cases should not be matched
        return features_case1

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", min=4, max=16)}


class GroupedAttentionModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        has_mask: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.has_mask = has_mask
        self.dropout = dropout
        self.n_rep = num_heads // num_kv_heads if num_kv_heads > 0 else 1

        # Define linear layers
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.out_proj = torch.nn.Linear(num_heads * self.head_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Generate q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Manually apply repeat_kv to k and v
        if self.num_kv_heads != self.num_heads:
            k = torch.ops.attention.repeat_kv(k, self.n_rep)
            v = torch.ops.attention.repeat_kv(v, self.n_rep)

        # Create attention mask if needed
        attn_mask = None
        if self.has_mask:
            # Simple causal mask
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            attn_mask = torch.zeros(
                (batch_size, 1, seq_len, seq_len), device=device, dtype=dtype
            ).masked_fill(mask, float("-inf"))

        # Apply scaled dot product attention
        attn_output = torch.ops.attention.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=False,
            scale=1.0 / (self.head_dim**0.5),
        )

        # Reshape output for the linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", min=4, max=16)}


@pytest.mark.parametrize("num_heads, num_kv_heads", [(8, 8), (8, 4), (8, 2)])
@pytest.mark.parametrize(
    "model_cls", [RepeatKVModel, RepeatKVModel2, RepeatKVModel3, HFRepeatKVModel]
)
@torch.inference_mode()
def test_match_repeat_kv(num_heads, num_kv_heads, model_cls):
    batch_size, seq_len = 4, 12
    hidden_size = 512

    model = model_cls(hidden_size, num_heads, num_kv_heads).to("cuda", dtype=torch.float16)
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()

    # When num_heads == num_kv_heads, we don't expect any pattern match
    # Otherwise, we should find 2 instances (one for k and one for v)
    expected_matches = 0 if num_heads == num_kv_heads else 2

    def verify_matcher(gm):
        repeat_kv_nodes = [n for n in gm.graph.nodes if is_op(n, torch.ops.attention.repeat_kv)]

        # Check that we have the expected number of replacements
        if len(repeat_kv_nodes) != expected_matches:
            return False

        # If we don't expect any matches, we're done
        if expected_matches == 0:
            return True

        # Otherwise, check the shape metadata of all repeat_kv nodes
        for node in repeat_kv_nodes:
            # Check input tensor metadata
            input_tensor = node.args[0]
            n_rep = node.args[1]

            if "val" not in input_tensor.meta:
                return False

            input_shape = input_tensor.meta["val"].shape
            output_shape = node.meta.get("val", None).shape if "val" in node.meta else None

            # Input should be [batch, num_kv_heads, seq_len, head_dim]
            if len(input_shape) != 4:
                return False

            batch, input_heads, input_seq, input_dim = input_shape

            # Output should be [batch, num_heads, seq_len, head_dim]
            # where num_heads = num_kv_heads * n_rep
            if output_shape is None or len(output_shape) != 4:
                return False

            output_batch, output_heads, output_seq, output_dim = output_shape

            # Check shapes are consistent
            if (
                output_batch != batch
                or output_heads != input_heads * n_rep
                or output_seq != input_seq
                or output_dim != input_dim
            ):
                print(
                    f"Expected shape {(batch, input_heads * n_rep, input_seq, input_dim)}, got {output_shape}"
                )
                return False

        return True

    _ = run_test(
        model,
        x,
        match_repeat_kv,
        verify_matcher,
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )


@pytest.mark.parametrize("has_mask", [True, False])
@pytest.mark.parametrize("use_division", [False, True])
@pytest.mark.parametrize(
    "dropout, rtol, atol",
    [
        (0.0, 1e-3, 1e-3),  # (dropout, rtol, atol) for no dropout
        (0.1, float("inf"), float("inf")),  # (dropout, rtol, atol) for dropout=0.1
    ],
)
@pytest.mark.parametrize("model_type", ["standard", "complex"])
@torch.inference_mode()
def test_match_eager_attention(has_mask, use_division, dropout, rtol, atol, model_type):
    # Set a fixed seed for consistent dropout behavior in tests
    torch.manual_seed(0)

    batch_size, seq_len = 4, 12
    hidden_size = 512
    num_heads = 8

    # Create different model types based on the parameter
    if model_type == "standard":
        model = EagerAttentionModel(hidden_size, num_heads, has_mask, dropout, use_division).to(
            "cuda", dtype=torch.float16
        )
        # Print the original scaling approach and value
        if use_division:
            print(f"\nOriginal model using DIVISION with inv_scaling={model.inv_scaling}")
            expected_scale = 1.0 / model.inv_scaling
        else:
            print(f"\nOriginal model using MULTIPLICATION with scaling={model.scaling}")
            expected_scale = model.scaling
    else:  # complex
        # Complex model only uses division for scaling
        model = ComplexEagerAttentionModel(hidden_size, num_heads, has_mask, dropout).to(
            "cuda", dtype=torch.float16
        )
        expected_scale = 1.0 / model.scale_divisor
        # Override use_division and only run test once (ignore the parameterization)
        if not use_division:
            pytest.skip("Complex model only uses division scaling")

    print(f"Expected normalized scale: {expected_scale}")
    # Use fixed seed for input to ensure consistent results
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()

    # We should find 1 instance of the pattern
    expected_matches = 1

    def verify_matcher(gm):
        sdpa_nodes = [
            n for n in gm.graph.nodes if is_op(n, torch.ops.attention.scaled_dot_product_attention)
        ]

        # Check that we have the expected number of replacements
        if len(sdpa_nodes) != expected_matches:
            print(f"Expected {expected_matches} SDPA nodes, found {len(sdpa_nodes)}")
            return False

        # Print information about all SDPA nodes
        print("\nSDPA nodes found:")
        for i, node in enumerate(sdpa_nodes):
            print(f"Node {i + 1}:")
            print(f"  Args count: {len(node.args)}")
            print(f"  Kwargs: {node.kwargs}")

            # Extract scale value - check kwargs first, then args
            scale_value = node.kwargs.get("scale", None)
            if scale_value is None and len(node.args) >= 7:  # Assuming scale is the 7th argument
                scale_value = node.args[6]

            scale_value = "missing" if scale_value is None else scale_value
            print(f"  Scale value: {scale_value}")

            # Compare with expected
            if scale_value != "missing":
                if abs(scale_value - expected_scale) < 1e-6:
                    print("  ✅ Scale matches expected value")
                else:
                    print(f"  ❌ Scale mismatch: expected {expected_scale}, got {scale_value}")

        # Check expected validations
        valid = True
        for node in sdpa_nodes:
            # Basic check: should have at least 3 positional args (q, k, v)
            if len(node.args) < 3:
                print(f"❌ SDPA node has fewer than 3 args: {len(node.args)}")
                valid = False

            # Check keyword arguments based on the configuration
            kwargs = node.kwargs

            # Check dropout rate is correctly preserved - check both kwargs and args
            dropout_value = kwargs.get("dropout_p", None)
            if (
                dropout_value is None and len(node.args) >= 5
            ):  # Assuming dropout_p is the 5th argument
                dropout_value = node.args[4]

            if dropout_value != dropout:
                print(f"❌ Expected dropout_p={dropout}, got {dropout_value}")
                valid = False

            # Check that scale factor is set properly - in either kwargs or args
            scale_value = kwargs.get("scale", None)
            if scale_value is None and len(node.args) >= 7:  # Assuming scale is the 7th argument
                scale_value = node.args[6]

            if scale_value is None:
                print("❌ Missing scale parameter in SDPA node")
                valid = False
            # Check the actual value of the scale parameter - should be close to expected_scale
            # regardless of whether we used multiplication or division in the original pattern
            elif abs(scale_value - expected_scale) > 1e-6:
                print(f"❌ Expected scale={expected_scale}, got {scale_value}")
                valid = False

            # Check mask handling for masked attention
            if has_mask:
                has_mask_arg = "attn_mask" in kwargs
                if not has_mask_arg and len(node.args) >= 4:
                    has_mask_arg = node.args[3] is not None

                if not has_mask_arg:
                    print("❌ Missing mask information in SDPA node")
                    valid = False

        print("Graph verification successful" if valid else "Graph verification failed")
        return valid

    # Run the test with the run_test utility
    run_test(
        model,
        x,
        match_eager_attention,
        verify_matcher,
        lambda num_p_og: num_p_og,
        atol=atol,
        rtol=rtol,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )


@torch.inference_mode()
def test_counter_example():
    """Test that similar tensor operations with different patterns are not falsely matched"""
    batch_size, seq_len = 4, 12
    hidden_size = 512
    num_heads = 8

    model = CounterExampleModel(hidden_size, num_heads).to("cuda", dtype=torch.float16)
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()

    def verify_no_matches(gm):
        # No nodes should be replaced with torch.ops.attention.repeat_kv
        repeat_kv_nodes = [n for n in gm.graph.nodes if is_op(n, torch.ops.attention.repeat_kv)]
        return len(repeat_kv_nodes) == 0

    # Ensure the pattern matcher doesn't match our counter-examples
    _ = run_test(
        model,
        x,
        match_repeat_kv,
        verify_no_matches,
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )


@pytest.mark.parametrize("num_heads, num_kv_heads", [(8, 8), (8, 4), (8, 2), (8, 1)])
@pytest.mark.parametrize("has_mask", [True, False])
@torch.inference_mode()
def test_match_grouped_attention(num_heads, num_kv_heads, has_mask):
    batch_size, seq_len = 4, 12
    hidden_size = 512

    model = GroupedAttentionModel(hidden_size, num_heads, num_kv_heads, has_mask).to(
        "cuda", dtype=torch.float16
    )
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()

    # We should find 1 instance of the pattern if num_heads != num_kv_heads
    # Otherwise, no pattern should be matched (no grouped attention)
    expected_matches = 1 if num_heads != num_kv_heads else 0

    def verify_matcher(gm):
        grouped_sdpa_nodes = [
            n for n in gm.graph.nodes if is_op(n, torch.ops.attention.grouped_sdpa)
        ]

        # Check that we have the expected number of replacements
        if len(grouped_sdpa_nodes) != expected_matches:
            print(
                f"Expected {expected_matches} grouped SDPA nodes, found {len(grouped_sdpa_nodes)}"
            )
            return False

        # If we don't expect any matches, we're done
        if expected_matches == 0:
            return True

        # Otherwise, check the node properties
        for node in grouped_sdpa_nodes:
            # Basic checks: should have at least 3 positional args (q, k, v)
            if len(node.args) < 3:
                print(f"❌ Grouped SDPA node has fewer than 3 args: {len(node.args)}")
                return False

            # Check kwargs
            kwargs = node.kwargs

            # Mask handling should be preserved
            if has_mask:
                # Check if attn_mask is in kwargs or provided via args
                has_mask_arg = "attn_mask" in kwargs
                if (
                    not has_mask_arg and len(node.args) >= 4
                ):  # Assuming attn_mask is the 4th positional arg
                    has_mask_arg = node.args[3] is not None

                if not has_mask_arg:
                    print("❌ Expected attn_mask in args or kwargs but not found")
                    return False

        return True

    # Run the test
    _ = run_test(
        model,
        x,
        match_grouped_attention,
        verify_matcher,
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )


class CausalAttentionModel(torch.nn.Module):
    """Model that creates different types of causal attention masks for testing."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mask_type: str = "triu",  # Options: "triu", "negative_fill", "non_causal"
        use_grouped_sdpa: bool = False,
        num_kv_heads: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mask_type = mask_type
        self.head_dim = hidden_size // num_heads
        self.use_grouped_sdpa = use_grouped_sdpa
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        # Define linear layers
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = torch.nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = torch.nn.Linear(num_heads * self.head_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Generate q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # For grouped attention, repeat k and v
        if self.use_grouped_sdpa and self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = torch.ops.attention.repeat_kv(k, n_rep)
            v = torch.ops.attention.repeat_kv(v, n_rep)

        # Create attention mask based on mask_type
        if self.mask_type == "triu":
            # Classic triu-based causal mask
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            attn_mask = torch.zeros(
                (batch_size, 1, seq_len, seq_len), device=device, dtype=dtype
            ).masked_fill(mask, float("-inf"))

        elif self.mask_type == "negative_fill":
            # Causal mask with large negative values
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            attn_mask = torch.full(
                (batch_size, 1, seq_len, seq_len), -65504.0, device=device, dtype=dtype
            )  # -65504 is the min value for float16
            attn_mask = attn_mask.masked_fill(~mask, 0.0)

        elif self.mask_type == "non_causal":
            # A non-causal mask (e.g., attention to every other token)
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            mask[:, 1::2] = True  # Mask every other column
            mask = mask.unsqueeze(0).unsqueeze(0)
            attn_mask = torch.zeros(
                (batch_size, 1, seq_len, seq_len), device=device, dtype=dtype
            ).masked_fill(mask, float("-inf"))

        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        # Choose the appropriate attention implementation
        if self.use_grouped_sdpa:
            attn_output = torch.ops.attention.grouped_sdpa(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,  # Explicitly use mask rather than is_causal flag
                scale=1.0 / (self.head_dim**0.5),
            )
        else:
            attn_output = torch.ops.attention.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,  # Explicitly use mask rather than is_causal flag
                scale=1.0 / (self.head_dim**0.5),
            )

        # Reshape output for the linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", min=4, max=16)}


@pytest.mark.parametrize("mask_type", ["triu", "negative_fill", "non_causal"])
@pytest.mark.parametrize("use_grouped_sdpa", [False, True])
@torch.inference_mode()
def test_match_causal_attention(mask_type, use_grouped_sdpa):
    batch_size, seq_len = 4, 12
    hidden_size = 512
    num_heads = 8
    num_kv_heads = 4 if use_grouped_sdpa else num_heads

    model = CausalAttentionModel(
        hidden_size,
        num_heads,
        mask_type=mask_type,
        use_grouped_sdpa=use_grouped_sdpa,
        num_kv_heads=num_kv_heads,
    ).to("cuda", dtype=torch.float16)

    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()

    # We expect optimization (None mask + is_causal=True) when using causal masks
    should_optimize = mask_type in ["triu", "negative_fill"]

    def verify_matcher(gm):
        # Find attention operations
        if use_grouped_sdpa:
            attn_nodes = [n for n in gm.graph.nodes if is_op(n, torch.ops.attention.grouped_sdpa)]
        else:
            attn_nodes = [
                n
                for n in gm.graph.nodes
                if is_op(n, torch.ops.attention.scaled_dot_product_attention)
            ]

        if len(attn_nodes) != 1:
            print(f"Expected 1 attention node, found {len(attn_nodes)}")
            return False

        node = attn_nodes[0]

        # Check if attention mask was set to None and is_causal was set to True
        if should_optimize:
            # Attention mask (4th arg) should be None
            has_mask = (
                node.args[3] is not None if len(node.args) > 3 else "attn_mask" in node.kwargs
            )

            # is_causal (6th arg) should be True
            is_causal = node.args[5] if len(node.args) > 5 else node.kwargs.get("is_causal", False)

            # Check if optimization was correctly applied
            if has_mask or not is_causal:
                print("❌ Expected optimization: mask=None, is_causal=True")
                print(
                    f"   Got: mask={node.args[3] if len(node.args) > 3 else 'not in args'}, "
                    f"is_causal={is_causal}"
                )
                return False

            print("✅ Successfully optimized causal mask: mask=None, is_causal=True")
        else:
            # Non-causal masks should remain as is
            has_mask = (
                node.args[3] is not None if len(node.args) > 3 else "attn_mask" in node.kwargs
            )

            # Check if non-optimization was correctly preserved
            if not has_mask:
                print("❌ Expected non-causal mask to be preserved")
                return False

            print("✅ Successfully preserved non-causal mask")

        return True

    # Run the test
    _ = run_test(
        model,
        x,
        match_causal_attn_mask,
        verify_matcher,
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )


class Llama3CausalAttentionModel(torch.nn.Module):
    """Model that creates a causal attention mask mimicking the llama-3.1 pattern."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        use_grouped_sdpa: bool = False,
        num_kv_heads: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_grouped_sdpa = use_grouped_sdpa
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        # Define linear layers
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = torch.nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = torch.nn.Linear(num_heads * self.head_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Generate q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # For grouped attention, repeat k and v
        if self.use_grouped_sdpa and self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = torch.ops.attention.repeat_kv(k, n_rep)
            v = torch.ops.attention.repeat_kv(v, n_rep)

        # Create a llama-3.1 style causal mask
        # 1. Create a full tensor with a very negative value
        full_tensor = torch.full(
            [seq_len, seq_len + 1], -3.3895313892515355e38, device=device, dtype=torch.float32
        )

        # 2. Create the triu mask with offset 1
        triu_mask = torch.triu(full_tensor, diagonal=1)

        # 3. Create position indices
        arange = torch.arange(seq_len, device=device).reshape([-1, 1])
        arange_plus1 = torch.arange(seq_len + 1, device=device)

        # 4. Create comparison mask (gt)
        gt_mask = arange_plus1 > arange

        # 5. Apply mul_ operation to combine masks
        combined_mask = triu_mask.mul_(
            gt_mask
        )  # Using mul_ instead of * for exact match with llama-3.1

        # 6. Apply unsqueeze operations
        unsqueezed1 = combined_mask.unsqueeze(0)
        unsqueezed2 = unsqueezed1.unsqueeze(1)

        # 7. Expand to batch size
        expanded = unsqueezed2.expand([batch_size, 1, -1, -1])

        # 8. Slice to get the right shape
        attn_mask = expanded[:, :, :, :seq_len]

        # Choose the appropriate attention implementation
        if self.use_grouped_sdpa:
            attn_output = torch.ops.attention.grouped_sdpa(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,  # Explicitly use mask rather than is_causal flag
                scale=1.0 / (self.head_dim**0.5),
            )
        else:
            attn_output = torch.ops.attention.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,  # Explicitly use mask rather than is_causal flag
                scale=1.0 / (self.head_dim**0.5),
            )

        # Reshape output for the linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", min=4, max=16)}


@pytest.mark.parametrize("use_grouped_sdpa", [False, True])
@pytest.mark.skip(reason="Skip until we have more robust attention masking handling, see #4783")
@torch.inference_mode()
def test_match_llama3_causal_attention(use_grouped_sdpa):
    batch_size, seq_len = 4, 12
    hidden_size = 512
    num_heads = 8
    num_kv_heads = 4 if use_grouped_sdpa else num_heads

    model = Llama3CausalAttentionModel(
        hidden_size,
        num_heads,
        use_grouped_sdpa=use_grouped_sdpa,
        num_kv_heads=num_kv_heads,
    ).to("cuda", dtype=torch.float32)

    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)
    dynamic_shapes = model.get_dynamic_shapes()

    def verify_matcher(gm):
        # Find attention operations
        if use_grouped_sdpa:
            attn_nodes = [n for n in gm.graph.nodes if is_op(n, torch.ops.attention.grouped_sdpa)]
        else:
            attn_nodes = [
                n
                for n in gm.graph.nodes
                if is_op(n, torch.ops.attention.scaled_dot_product_attention)
            ]

        if len(attn_nodes) != 1:
            print(f"Expected 1 attention node, found {len(attn_nodes)}")
            return False

        node = attn_nodes[0]

        # Attention mask (4th arg) should be None
        has_mask = node.args[3] is not None if len(node.args) > 3 else "attn_mask" in node.kwargs

        # is_causal (6th arg) should be True
        is_causal = node.args[5] if len(node.args) > 5 else node.kwargs.get("is_causal", False)

        # Check if optimization was correctly applied
        if has_mask or not is_causal:
            print("❌ Expected optimization: mask=None, is_causal=True")
            print(
                f"   Got: mask={node.args[3] if len(node.args) > 3 else 'not in args'}, "
                f"is_causal={is_causal}"
            )
            return False

        print("✅ Successfully optimized llama-3.1 causal mask: mask=None, is_causal=True")
        return True

    # Run the test
    run_test(
        model,
        x,
        match_causal_attn_mask,
        verify_matcher,
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )


class MockAttentionDescriptor:
    """A mock class that mimics the AttentionDescriptor interface for testing."""

    layout: str = "bnsd"
    source_attention_op: Callable = torch.ops.attention.scaled_dot_product_attention

    @classmethod
    def get_attention_layout(cls) -> str:
        return cls.layout

    @classmethod
    def get_source_attention_op(cls) -> Callable:
        return cls.source_attention_op


class AttentionLayoutModel(torch.nn.Module):
    """Model that uses SDPA for testing the layout transformation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        use_grouped_sdpa: bool = False,
        has_mask: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_grouped_sdpa = use_grouped_sdpa
        self.has_mask = has_mask

        # Define linear layers
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.out_proj = torch.nn.Linear(num_heads * self.head_dim, hidden_size)

    def _get_attn_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create a deterministic pseudo-random attention mask in the shape [b, n, s, s]."""
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Create a deterministic pseudo-random attention mask
        row_indices = (
            torch.arange(seq_len, device=device)
            .view(1, 1, -1, 1)
            .expand(batch_size, self.num_heads, -1, seq_len)
        )
        col_indices = (
            torch.arange(seq_len, device=device)
            .view(1, 1, 1, -1)
            .expand(batch_size, self.num_heads, seq_len, -1)
        )
        return ((row_indices + col_indices) % 3 == 0).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Generate q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim] - SDPA expects this layout
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Create attention mask if needed
        attn_mask = self._get_attn_mask(x) if self.has_mask else None

        # Apply scaled dot product attention
        if self.use_grouped_sdpa:
            attn_output = torch.ops.attention.grouped_sdpa(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=True,
                scale=1.0 / (self.head_dim**0.5),
            )
        else:
            attn_output = torch.ops.attention.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=True,
                scale=1.0 / (self.head_dim**0.5),
            )

        # Reshape output for the linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", min=4, max=16)}


class BsndAttentionModel(AttentionLayoutModel):
    """Model that directly uses bsnd_grouped_sdpa for testing the layout transformation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Generate q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Important: No transpose here because bsnd_grouped_sdpa expects [batch, seq, heads, dim]

        # Create attention mask if needed
        attn_mask = self._get_attn_mask(x) if self.has_mask else None

        # Apply bsnd_grouped_sdpa directly
        attn_output = torch.ops.attention.bsnd_grouped_sdpa.default(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=True,
            scale=1.0 / (self.head_dim**0.5),
        )

        # Reshape output for the linear projection (no transpose needed)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output


@pytest.mark.parametrize("layout", ["bnsd", "bsnd"])
@pytest.mark.parametrize(
    "model_config",
    [
        {"type": "standard", "use_grouped_sdpa": False, "name": "SDPA"},
        {"type": "standard", "use_grouped_sdpa": True, "name": "GroupedSDPA"},
        {"type": "already_bsnd", "name": "DirectBSND"},
    ],
)
@pytest.mark.parametrize("has_mask", [False, True])
@torch.inference_mode()
def test_match_attention_layout(layout, model_config, has_mask):
    """Test the match_attention_layout transformation with various models and layouts."""
    batch_size, seq_len = 4, 12
    hidden_size = 512
    num_heads = 8

    # Set up the mock attention descriptor class with the specified layout
    MockAttentionDescriptor.layout = layout
    if layout == "bnsd":
        if model_config.get("use_grouped_sdpa"):
            source_op = torch.ops.attention.grouped_sdpa
        else:
            source_op = torch.ops.attention.scaled_dot_product_attention
    else:
        source_op = torch.ops.attention.bsnd_grouped_sdpa
    MockAttentionDescriptor.source_attention_op = source_op

    # Create appropriate model based on model_config
    if model_config["type"] == "standard":
        model = AttentionLayoutModel(
            hidden_size,
            num_heads,
            use_grouped_sdpa=model_config["use_grouped_sdpa"],
            has_mask=has_mask,
        ).to("cuda", dtype=torch.float16)
    else:  # already_bsnd
        model = BsndAttentionModel(
            hidden_size,
            num_heads,
            has_mask=has_mask,
        ).to("cuda", dtype=torch.float16)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()

    # Print test case info
    print(f"\nRunning test: {model_config['name']} model with {layout} layout, has_mask={has_mask}")

    # Define appropriate verification function
    def verify_matcher(gm):
        # Identify node types to check based on model
        if model_config["type"] == "standard":
            if model_config["use_grouped_sdpa"]:
                original_nodes = [
                    n for n in gm.graph.nodes if is_op(n, torch.ops.attention.grouped_sdpa)
                ]
            else:
                original_nodes = [
                    n
                    for n in gm.graph.nodes
                    if is_op(n, torch.ops.attention.scaled_dot_product_attention)
                ]
        else:  # already_bsnd
            original_nodes = []

        bsnd_nodes = [n for n in gm.graph.nodes if is_op(n, torch.ops.attention.bsnd_grouped_sdpa)]
        transpose_nodes = [n for n in gm.graph.nodes if is_op(n, torch.ops.aten.transpose.int)]

        # Different expectations based on model type and layout
        if model_config["type"] == "already_bsnd":
            # A model already using bsnd_grouped_sdpa should remain unchanged
            if len(bsnd_nodes) != 1:
                print(f"❌ Expected 1 bsnd_grouped_sdpa node, found {len(bsnd_nodes)}")
                return False

            # Should not have any transposes
            if len(transpose_nodes) > 0:
                print(f"❌ Expected no transpose nodes, found {len(transpose_nodes)}")
                return False

            # Check that the mask is correctly passed to the attention function
            if has_mask:
                node = bsnd_nodes[0]
                has_mask_arg = "attn_mask" in node.kwargs
                if not has_mask_arg and len(node.args) >= 4:
                    has_mask_arg = node.args[3] is not None

                if not has_mask_arg:
                    print("❌ Expected attention mask in args or kwargs but not found")
                    return False

            print(f"✅ Already bsnd model correctly handled for {layout} layout")
        elif layout == "bsnd":
            # For layout=bsnd, original SDPA should be replaced with bsnd_grouped_sdpa
            if len(original_nodes) > 0:
                print(
                    f"❌ Expected original attention nodes to be replaced, found {len(original_nodes)}"
                )
                return False

            if len(bsnd_nodes) != 1:
                print(f"❌ Expected 1 bsnd_grouped_sdpa node, found {len(bsnd_nodes)}")
                return False

            # Check that the mask is correctly passed to the attention function
            if has_mask:
                node = bsnd_nodes[0]
                has_mask_arg = "attn_mask" in node.kwargs
                if not has_mask_arg and len(node.args) >= 4:
                    has_mask_arg = node.args[3] is not None

                if not has_mask_arg:
                    print("❌ Expected attention mask in args or kwargs but not found")
                    return False

            # We expect at least 7 transpose nodes:
            # - 3 from original model (q,k,v)
            # - 3 for the new input transposes
            # - 1 for the new output transpose
            # Note: Some nodes may be optimized away, so we check for at least 7
            if len(transpose_nodes) < 7:
                print(f"❌ Expected at least 7 transpose nodes, found {len(transpose_nodes)}")
                return False

            print("✅ Successfully transformed graph for bsnd layout")
        else:  # layout = bnsd
            # For layout=bnsd, graph should remain unchanged
            if len(original_nodes) != 1:
                print(f"❌ Expected 1 original attention node, found {len(original_nodes)}")
                return False

            if len(bsnd_nodes) > 0:
                print(f"❌ Expected no bsnd_grouped_sdpa nodes, found {len(bsnd_nodes)}")
                return False

            # Check that the mask is correctly passed to the attention function
            if has_mask:
                node = original_nodes[0]
                has_mask_arg = "attn_mask" in node.kwargs
                if not has_mask_arg and len(node.args) >= 4:
                    has_mask_arg = node.args[3] is not None

                if not has_mask_arg:
                    print("❌ Expected attention mask in args or kwargs but not found")
                    return False

            # The model has 4 transposes: 3 for q,k,v inputs and 1 for output
            if len(transpose_nodes) != 4:
                print(f"❌ Expected 4 transpose nodes, found {len(transpose_nodes)}")
                return False

            print("✅ Graph correctly unchanged for bnsd layout")

        return True

    # Run the test
    run_test(
        model,
        x,
        lambda gm: match_attention_layout(gm, MockAttentionDescriptor),
        verify_matcher,
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )
