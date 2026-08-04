import os
import unittest
from unittest import mock

import torch

from tensorrt_llm._torch import pinned_weight_staging


class TestPinnedWeightStaging(unittest.TestCase):
    """Tests for the scoped pinned-staging weight-load path."""

    def test_disabled_by_default_is_noop(self):
        orig_to = torch.Tensor.to
        orig_copy = torch.Tensor.copy_
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRTLLM_PINNED_WEIGHT_STAGING", None)
            with pinned_weight_staging.staging_scope():
                self.assertIs(torch.Tensor.to, orig_to)
                self.assertIs(torch.Tensor.copy_, orig_copy)
        self.assertIs(torch.Tensor.to, orig_to)
        self.assertIs(torch.Tensor.copy_, orig_copy)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_enabled_installs_and_restores(self):
        orig_to = torch.Tensor.to
        orig_copy = torch.Tensor.copy_
        with mock.patch.dict(os.environ, {"TRTLLM_PINNED_WEIGHT_STAGING": "1"}):
            with pinned_weight_staging.staging_scope():
                self.assertIsNot(torch.Tensor.to, orig_to)
                self.assertIsNot(torch.Tensor.copy_, orig_copy)
                # Nested scopes share one installation.
                with pinned_weight_staging.staging_scope():
                    self.assertIsNot(torch.Tensor.to, orig_to)
            self.assertIs(torch.Tensor.to, orig_to)
            self.assertIs(torch.Tensor.copy_, orig_copy)
            self.assertEqual(len(pinned_weight_staging._bufs), 0)

    def test_cpu_paths_unaffected_inside_scope(self):
        # Also covers the CPU-only case where prefer_pinned() disables
        # staging and the scope is a no-op.
        with mock.patch.dict(os.environ, {"TRTLLM_PINNED_WEIGHT_STAGING": "1"}):
            with pinned_weight_staging.staging_scope():
                x = torch.randn(8, 8)
                y = x.to(torch.float64)
                self.assertEqual(y.dtype, torch.float64)
                z = torch.empty(8, 8)
                z.copy_(x)
                torch.testing.assert_close(z, x)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_staged_h2d_correctness_and_release(self):
        with mock.patch.dict(os.environ, {"TRTLLM_PINNED_WEIGHT_STAGING": "1"}):
            with pinned_weight_staging.staging_scope():
                for dtype in (torch.float32, torch.bfloat16, torch.uint8):
                    if dtype.is_floating_point:
                        src = torch.randn(257, 129).to(dtype)
                    else:
                        src = torch.randint(0, 255, (257, 129), dtype=dtype)
                    dev = src.to("cuda")
                    torch.testing.assert_close(dev.cpu(), src)
                    dst = torch.empty_like(src, device="cuda")
                    dst.copy_(src)
                    torch.testing.assert_close(dst.cpu(), src)
                # Non-contiguous and sliced sources.
                base = torch.randn(64, 64)
                view = base[::2, 1:33]
                torch.testing.assert_close(view.to("cuda").cpu(), view)
                # Buffers exist while the scope is open...
                self.assertGreater(len(pinned_weight_staging._bufs), 0)
            # ...and are freed on exit.
            self.assertEqual(len(pinned_weight_staging._bufs), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_pinned_source_bypasses_staging(self):
        with mock.patch.dict(os.environ, {"TRTLLM_PINNED_WEIGHT_STAGING": "1"}):
            with pinned_weight_staging.staging_scope():
                src = torch.randn(32, 32).pin_memory()
                dev = src.to("cuda")
                torch.testing.assert_close(dev.cpu(), src)
                self.assertEqual(len(pinned_weight_staging._bufs), 0)


if __name__ == "__main__":
    unittest.main()
