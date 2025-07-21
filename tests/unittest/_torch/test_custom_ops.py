# Adapted from https://github.com/pytorch/pytorch/blob/393377d2156cf4dfb0a7d53c79a85a8b24055ae0/test/test_custom_ops.py

import unittest
from typing import *  # noqa: F403

import torch
import torch._library.utils as library_utils
import torch.testing._internal.optests as optests
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.testing._internal.optests.generate_tests import \
    resolve_unique_overload_or_throw

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import tensorrt_llm._torch.custom_ops  # noqa: F401


def requires_compile(fun):
    fun = unittest.skipIf(IS_WINDOWS,
                          "torch.compile doesn't work with windows")(fun)
    return fun


class CustomOpTestCaseBase(unittest.TestCase):
    test_ns = "_test_custom_op"

    custom_op_namespaces = ("auto_deploy", "trtllm")

    @classmethod
    def setUpClass(cls):
        cls.custom_ops = cls.discover_custom_ops()

    @classmethod
    def discover_custom_ops(cls):
        """Discover all custom ops in the codebase."""
        discovered_ops = []
        for namespace in cls.custom_op_namespaces:
            ops = cls._discover_namespace_ops(namespace)
            discovered_ops.extend(ops)

        names = "\n\t".join(op._name for op in discovered_ops)
        print(f"Discovered {len(discovered_ops)} custom ops:\n\t{names}")
        return discovered_ops

    @classmethod
    def _discover_namespace_ops(cls, namespace: str, prefix: str = ""):
        """Discover custom ops in a specific namespace."""
        ops = []

        if not hasattr(torch.ops, namespace):
            return ops

        namespace_obj = getattr(torch.ops, namespace)

        for attr_name in dir(namespace_obj):
            if not attr_name.startswith(prefix):
                continue

            op = getattr(namespace_obj, attr_name)

            if isinstance(op, torch._library.custom_ops.CustomOpDef):
                op = op._opoverload
            elif isinstance(op, torch._ops.OpOverloadPacket):
                op = resolve_unique_overload_or_throw(op)
            elif isinstance(op, torch._ops.OpOverload):
                pass
            else:
                continue
            ops.append(op)

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
        ops_missing_fake_impl = []
        for op in self.custom_ops:
            if not library_utils.has_fake_kernel(op):
                ops_missing_fake_impl.append(op)

        names = ", ".join(op._name for op in ops_missing_fake_impl)
        self.assertTrue(
            len(ops_missing_fake_impl) == 0,
            f"Custom ops missing fake impl: {names}")


if __name__ == "__main__":
    unittest.main()
