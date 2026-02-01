"""A set of transforms to handle SSM cache transforms."""

from ..interface import TransformRegistry
from .kvcache import _InsertCachedOperator


# TODO: think about separating valid attention backends per transform better in the future
@TransformRegistry.register("insert_cached_ssm_attention")
class SSMCacheTransform(_InsertCachedOperator):
    """A transform to handle SSM cache operations."""


@TransformRegistry.register("insert_cached_causal_conv")
class InitializeCausalConvCache(_InsertCachedOperator):
    """A transform to handle causal conv cache operations."""


@TransformRegistry.register("insert_cached_delta_rule")
class InsertCachedDeltaRule(_InsertCachedOperator):
    """A transform to handle delta rule cache operations."""
