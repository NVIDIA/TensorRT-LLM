"""Unit tests for diffusion quantization operations."""

import unittest

import torch

from tensorrt_llm._torch.visual_gen.quantization.ops import (
    quantize_fp8_blockwise,
    quantize_fp8_per_tensor,
)


def _dequant_fp8_per_tensor(qweight, scale):
    """Dequantize per-tensor FP8 weight."""
    return qweight.to(torch.float32) * scale


class TestQuantOps(unittest.TestCase):
    """Test quantization operations."""

    def setUp(self):
        """Set random seed for reproducibility."""
        torch.manual_seed(42)
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_fp8_per_tensor(self):
        """Test FP8 per-tensor quantization using CUDA kernel."""
        weight = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
        qweight, scale = quantize_fp8_per_tensor(weight)

        # Check output types
        self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
        self.assertEqual(qweight.shape, weight.shape)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(scale.shape, (1, 1))

        # Verify dequantization (approximate)
        dequant = _dequant_fp8_per_tensor(qweight, scale)
        error = (dequant - weight.to(torch.float32)).abs().mean()
        self.assertLess(error, 0.15)  # Reasonable quantization error

    def test_fp8_per_tensor_different_shapes(self):
        """Test FP8 per-tensor quantization with various shapes."""
        shapes = [(128, 256), (256, 512), (512, 1024), (1024, 2048)]
        for shape in shapes:
            with self.subTest(shape=shape):
                weight = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
                qweight, scale = quantize_fp8_per_tensor(weight)

                self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
                self.assertEqual(qweight.shape, weight.shape)
                self.assertEqual(scale.dtype, torch.float32)

    def test_fp8_blockwise(self):
        """Test FP8 128x128 blockwise quantization."""
        weight = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
        block_size = 128
        qweight, scales = quantize_fp8_blockwise(weight, block_size=block_size)

        # Check output types
        self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
        self.assertEqual(qweight.shape, weight.shape)
        self.assertEqual(scales.dtype, torch.float32)

        # Check scales shape: (num_blocks_out, num_blocks_in) for 128x128 blocks
        num_blocks_out = (512 + block_size - 1) // block_size  # 4
        num_blocks_in = (512 + block_size - 1) // block_size  # 4
        self.assertEqual(scales.shape, (num_blocks_out, num_blocks_in))

    def test_fp8_blockwise_non_divisible(self):
        """Test FP8 blockwise quantization with non-divisible dimensions."""
        weight = torch.randn(300, 500, dtype=torch.bfloat16, device="cuda")
        block_size = 128
        qweight, scales = quantize_fp8_blockwise(weight, block_size=block_size)

        # Check output types
        self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
        self.assertEqual(qweight.shape, weight.shape)

        # Check scales shape (should handle non-divisible dimensions)
        num_blocks_out = (300 + block_size - 1) // block_size  # 3
        num_blocks_in = (500 + block_size - 1) // block_size  # 4
        self.assertEqual(scales.shape, (num_blocks_out, num_blocks_in))

    def test_fp8_blockwise_different_block_sizes(self):
        """Test FP8 blockwise quantization with different block sizes."""
        weight = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

        for block_size in [64, 128, 256]:
            with self.subTest(block_size=block_size):
                qweight, scales = quantize_fp8_blockwise(weight, block_size=block_size)

                self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
                self.assertEqual(qweight.shape, weight.shape)

                num_blocks = (256 + block_size - 1) // block_size
                self.assertEqual(scales.shape, (num_blocks, num_blocks))

    def test_fp8_per_tensor_zero_weight(self):
        """Test FP8 per-tensor quantization with zero weight."""
        weight = torch.zeros(128, 256, dtype=torch.bfloat16, device="cuda")
        qweight, scale = quantize_fp8_per_tensor(weight)

        # Should handle zero weights gracefully
        self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.all(qweight.to(torch.float32) == 0))

    def test_fp8_blockwise_zero_weight(self):
        """Test FP8 blockwise quantization with zero weight."""
        weight = torch.zeros(256, 256, dtype=torch.bfloat16, device="cuda")
        qweight, scales = quantize_fp8_blockwise(weight, block_size=128)

        # Should handle zero weights gracefully
        self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.all(qweight.to(torch.float32) == 0))


if __name__ == "__main__":
    unittest.main()
