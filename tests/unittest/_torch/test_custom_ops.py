# Adapted from https://github.com/pytorch/pytorch/blob/393377d2156cf4dfb0a7d53c79a85a8b24055ae0/test/test_custom_ops.py

import unittest
from typing import *  # noqa: F403

import torch
import torch._library.utils as library_utils
import torch.testing._internal.optests as optests
from torch.testing._internal.common_utils import IS_WINDOWS

import tensorrt_llm  # noqa: F401


def requires_compile(fun):
    fun = unittest.skipIf(IS_WINDOWS,
                          "torch.compile doesn't work with windows")(fun)
    return fun


class CustomOpTestCaseBase(unittest.TestCase):
    test_ns = "_test_custom_op"

    # "auto_deploy" custom ops are not checked here.
    custom_op_namespaces = ("trtllm", )

    @classmethod
    def setUpClass(cls):
        cls.custom_ops = cls.discover_custom_ops()

    @classmethod
    def discover_custom_ops(cls):
        """Discover all custom ops in the codebase."""
        discovered_ops = []
        for namespace in cls.custom_op_namespaces:
            ops = cls._discover_namespace_ops(namespace)
            print(f"Total {len(ops)} custom ops in namespace {namespace}")
            discovered_ops.extend(ops)
        return discovered_ops

    @classmethod
    def _discover_namespace_ops(cls, namespace: str, prefix: str = ""):
        """Discover custom ops in a specific namespace."""
        # C++ custom ops are lazy loaded, cannot use torch.ops.x to discover all custom ops.
        # Use schemas to discover instead.
        ops_schemas = torch._C._jit_get_all_schemas()
        ops = []

        ns_prefix = f"{namespace}::{prefix}"
        print("Discovering custom ops:")
        for schema in ops_schemas:
            if not schema.name.startswith(ns_prefix):
                continue
            op = library_utils.lookup_op(schema.name)
            ops.append(op)
            print(f"    {op._name}")

        return ops

    def setUp(self):
        super().setUp()
        self.libraries = []

    def tearDown(self):
        super().tearDown()
        import torch._custom_op

        keys = list(torch._custom_op.impl.global_registry.keys())
        for key in keys:
            if not key.startswith(f"{self.test_ns}::"):
                continue
            torch._custom_op.impl.global_registry[key]._destroy()
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        for lib in self.libraries:
            lib._destroy()
        del self.libraries

    def ns(self):
        return getattr(torch.ops, self.test_ns)

    def lib(self):
        result = torch.library.Library(self.test_ns, "FRAGMENT")  # noqa: TOR901
        self.libraries.append(result)
        return result


@requires_compile
class TestCustomOp(CustomOpTestCaseBase):
    """
    Test suite to verify the functionality and correctness of the custom operators,
    ensuring that custom ops registered as expected and integrate properly with
    PyTorch's operator checking and testing infrastructure.
    """

    def test_missing_fake_impl(self):
        """Test custom operator missing fake impl."""
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                return 2 * gx

        def foo_impl(x):
            return torch.tensor(x.cpu().numpy()**2, device=x.device)

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")

        # ensure library_utils.has_fake_kernel works as expected
        x = torch.tensor([0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(
                optests.OpCheckError,
                "_test_custom_op.foo.default",
        ):
            torch.library.opcheck(op, (x, ), {})

        self.assertFalse(library_utils.has_fake_kernel(op))

    # Better to add OpInfo for each custiom op, and use opcheck to test the custom ops.
    # Currently OpInfo for custom ops are not available in the codebase.
    # As a trade-off, only fake registration is checked.
    def test_register_fake(self):
        """Test custom operator fake impl registration."""

        # Custom ops that are not required to have fake impl.
        waivers = {
            "trtllm::record_stream",
            "trtllm::wait_event",
            "trtllm::record_event",
            "trtllm::set_stream",
        }

        # TODO: add fake impl for these ops in follow-up PRs.
        to_fix = {
            "trtllm::lora_grouped_gemm",
            "trtllm::mtp_relaxed_acceptance_op",
            "trtllm::mtp_update_hidden_states_op",
            "trtllm::mtp_prepare_drafter_inputs_op",
            "trtllm::selective_scan",
            "trtllm::reducescatter_list",
            "trtllm::fp8_per_tensor_scale_moe_runner",
            "trtllm::migrate_to_host_accessible",
            "trtllm::mnnvl_moe_alltoallv_prepare_without_allgather",
            "trtllm::mamba_conv1d",
            "trtllm::llama4_moe_tp8ep1_min_latency",
            "trtllm::llama4_fp8_fp8_gemm_swiglu",
            "trtllm::llama4_fp8_bf16_gemm",
            "trtllm::llama4_bf16_bf16_gemm",
            "trtllm::fused_topk_softmax",
            "trtllm::fp8_batched_quantize_1x128_permute102",
            "trtllm::fp8_block_scaling_moe_gemm",
            "trtllm::fp8_block_scaling_bmm_out",
            "trtllm::fp8_block_scaling_bmm",
            "trtllm::fp4_batched_quantize",
            "trtllm::fp4_gemm_trtllmgen",
            "trtllm::fp4_bmm",
            "trtllm::merge_chunked_attention_for_mla",
            "trtllm::cuda_scaled_mm",
            "trtllm::cublas_mm_out",
            "trtllm::initialize_static_lowprecision_buffers",
            "trtllm::cutlass_scaled_mm",
            "trtllm::fp8_per_tensor_scaling_tllmg_gemm",
            "trtllm::load_chunked_kv_cache_for_mla",
            "trtllm::load_paged_kv_cache_for_mla",
            "trtllm::set_paged_kv_cache_for_mla",
            "trtllm::set_chunked_kv_cache_for_mla",
            "trtllm::mla_rope_append_paged_kv_assign_q",
            "trtllm::cublas_scaled_mm_out",
            "trtllm::fused_qk_norm_rope",
        }

        ops_missing_fake_impl = []

        for op in self.custom_ops:
            if op._name in waivers or op._name in to_fix:
                continue
            if not library_utils.has_fake_kernel(op):
                ops_missing_fake_impl.append(op)

        names = ", ".join(op._name for op in ops_missing_fake_impl)
        self.assertTrue(
            len(ops_missing_fake_impl) == 0,
            f"Custom ops missing fake impl: {names}")


if __name__ == "__main__":
    unittest.main()
