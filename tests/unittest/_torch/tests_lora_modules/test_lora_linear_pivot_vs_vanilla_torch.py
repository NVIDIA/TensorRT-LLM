import torch
from torch.nn import Linear

from tensorrt_llm._torch.modules.linear import Linear as PivotLinear


class TestLoraLinearPivotVsVanilla:
    """Test class for comparing LoRA linear layer with vanilla PyTorch implementation."""

    def setup_method(self, method):
        """Set up test parameters and resources."""
        # Test parameters
        self.in_dim = 64
        self.out_dim = 64
        self.input_length = 10
        self.batch_size = 4
        self.device = "cuda"

        # Create input tensor
        self.input_tensor = torch.randn(self.batch_size,
                                        self.input_length,
                                        self.in_dim,
                                        device=self.device)

    def _get_lora_params(self):
        """Create LoRA parameters for the test."""
        lora_rank = 8
        lora_weight_ins = torch.randn(self.in_dim,
                                      lora_rank,
                                      device=self.device)
        lora_weight_outs = torch.randn(lora_rank,
                                       self.out_dim,
                                       device=self.device)
        return lora_weight_ins, lora_weight_outs

    def _setup_linear_layers(self):
        """Set up both vanilla and pivot linear layers with same weights."""
        torch_linear = Linear(self.in_dim, self.out_dim).to(self.device)
        pivot_linear = PivotLinear(self.in_dim, self.out_dim)

        # Initialize pivot linear with same weights as torch linear
        pivot_linear.weight.data = torch_linear.weight.data
        pivot_linear.bias.data = torch_linear.bias.data

        return torch_linear, pivot_linear

    def test_compare_linear_torch_pivot_lora(self):
        """Test comparison between vanilla and pivot linear layers with LoRA."""
        # Get LoRA parameters
        lora_weight_ins, lora_weight_outs = self._get_lora_params()

        # Set up linear layers
        torch_linear, pivot_linear = self._setup_linear_layers()

        # Compute LoRA output
        lora_output = (
            self.input_tensor @ lora_weight_outs.T) @ lora_weight_ins.T

        # Run vanilla linear with LoRA
        torch_output = torch_linear(self.input_tensor)
        torch_output = torch_output + lora_output

        # Run pivot linear with LoRA
        pivot_output = pivot_linear(self.input_tensor,
                                    lora_params={
                                        "lora_weight_ins": lora_weight_outs,
                                        "lora_weight_outs": lora_weight_ins
                                    })

        # Compare outputs
        assert torch.allclose(torch_output, pivot_output)
        print("Test passed!")

    def test_compare_linear_torch_pivot(self):
        """Test comparison between vanilla and pivot linear layers without LoRA."""
        # Set up linear layers
        torch_linear, pivot_linear = self._setup_linear_layers()

        # Run vanilla linear
        torch_output = torch_linear(self.input_tensor)

        # Run pivot linear
        pivot_output = pivot_linear(self.input_tensor)

        # Compare outputs
        assert torch.allclose(torch_output, pivot_output)
        print("Test passed!")


if __name__ == "__main__":
    test = TestLoraLinearPivotVsVanilla()
    test.setup_method(None)  # None is passed as method parameter
    test.test_compare_linear_torch_pivot_lora()
    test.test_compare_linear_torch_pivot()
