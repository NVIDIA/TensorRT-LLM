"""Tests for ``create_derived_custom_op`` in ``_graph.py``."""

import torch
from torch._subclasses import FakeTensorMode

from tensorrt_llm._torch.auto_deploy.utils._graph import create_derived_custom_op

# ---------------------------------------------------------------------------
# Helpers – tiny custom ops used as base ops for the tests
# ---------------------------------------------------------------------------


@torch.library.custom_op("ad_test_derived::double", mutates_args=())
def _double(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@_double.register_fake
def _double_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("ad_test_derived::weighted_add", mutates_args=())
def _weighted_add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return x + alpha * y


@_weighted_add.register_fake
def _weighted_add_fake(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateDerivedCustomOp:
    """Tests for the ``create_derived_custom_op`` utility."""

    def test_basic_derived_op(self):
        """A derived op should be callable and produce correct results."""

        def make_impl(base_overload):
            # Wrapper that calls the base op then negates the result.
            def impl(*args, **kwargs):
                return -base_overload(*args, **kwargs)

            return impl

        base_op = torch.ops.ad_test_derived.double
        derived = create_derived_custom_op(base_op, "_neg", make_impl)

        x = torch.tensor([1.0, 2.0, 3.0])
        result = derived(x)
        expected = -(x * 2)
        torch.testing.assert_close(result, expected)

    def test_derived_op_is_registered(self):
        """The derived op must be accessible via ``torch.ops``."""

        def make_impl(base_overload):
            return lambda *a, **kw: base_overload(*a, **kw)

        create_derived_custom_op(torch.ops.ad_test_derived.double, "_registered", make_impl)
        assert hasattr(torch.ops.ad_test_derived, "double_registered")

    def test_caching(self):
        """Repeated calls with the same base_op + suffix must return the same object."""

        def make_impl(base_overload):
            return lambda *a, **kw: base_overload(*a, **kw)

        op1 = create_derived_custom_op(torch.ops.ad_test_derived.double, "_cached", make_impl)
        op2 = create_derived_custom_op(torch.ops.ad_test_derived.double, "_cached", make_impl)
        assert op1 is op2

    def test_different_suffix_produces_different_op(self):
        """Different suffixes must create distinct ops."""

        def make_impl(base_overload):
            return lambda *a, **kw: base_overload(*a, **kw)

        op_a = create_derived_custom_op(torch.ops.ad_test_derived.double, "_sfx_a", make_impl)
        op_b = create_derived_custom_op(torch.ops.ad_test_derived.double, "_sfx_b", make_impl)
        assert op_a is not op_b

    def test_default_fake_implementation(self):
        """When *make_fake* is None the default (empty_like) must be used."""

        def make_impl(base_overload):
            return lambda *a, **kw: base_overload(*a, **kw)

        derived = create_derived_custom_op(
            torch.ops.ad_test_derived.double, "_dflt_fake", make_impl
        )
        # Calling the Meta implementation via FakeTensorMode
        with FakeTensorMode():
            x = torch.empty(4)
            out = derived(x)
        assert out.shape == x.shape

    def test_custom_fake_implementation(self):
        """A user-supplied *make_fake* must override the default."""

        def make_impl(base_overload):
            return lambda *a, **kw: base_overload(*a, **kw)

        # Fake that always returns shape (1,) regardless of input shape.
        def make_fake(base_overload):
            def fake(*args, **kwargs):
                return args[0].new_empty(1)

            return fake

        derived = create_derived_custom_op(
            torch.ops.ad_test_derived.double,
            "_custom_fake",
            make_impl,
            make_fake=make_fake,
        )

        with FakeTensorMode():
            x = torch.empty(10)
            out = derived(x)
        assert out.shape == (1,)

    def test_preserves_schema_with_defaults(self):
        """Derived op must preserve the base op's argument defaults."""

        def make_impl(base_overload):
            def impl(*args, **kwargs):
                return base_overload(*args, **kwargs) * 10

            return impl

        base_op = torch.ops.ad_test_derived.weighted_add
        derived = create_derived_custom_op(base_op, "_x10", make_impl)

        x = torch.ones(3)
        y = torch.ones(3) * 2.0

        # With default alpha=1.0 → (x + 1.0*y) * 10 = 30
        result_default = derived(x, y)
        torch.testing.assert_close(result_default, torch.full((3,), 30.0))

        # With explicit alpha=0.5 → (x + 0.5*y) * 10 = 20
        result_alpha = derived(x, y, alpha=0.5)
        torch.testing.assert_close(result_alpha, torch.full((3,), 20.0))

    def test_accepts_op_overload(self):
        """The function should accept an OpOverload (e.g. ``.default``) as well."""

        def make_impl(base_overload):
            return lambda *a, **kw: base_overload(*a, **kw) + 1

        derived = create_derived_custom_op(
            torch.ops.ad_test_derived.double.default, "_from_overload", make_impl
        )

        x = torch.tensor([5.0])
        # double → 10, +1 → 11
        torch.testing.assert_close(derived(x), torch.tensor([11.0]))
