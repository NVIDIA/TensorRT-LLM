"""A set of transforms to handle SSM cache transforms."""

from typing import Tuple

from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import SharedConfig, TransformInfo, TransformRegistry
from .kvcache import _InsertCachedOperator


# TODO: think about separating valid attention backends per transform better in the future
@TransformRegistry.register("insert_cached_ssm_attention")
class SSMCacheTransform(_InsertCachedOperator):
    """A transform to handle SSM cache operations."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        qcfg = factory.get_quant_config()
        is_nvfp4 = qcfg.get("quant_algo", "").upper() == "NVFP4"
        if is_nvfp4 and self.config.backend == "flashinfer_ssm":
            self._log_warning(
                f"SSM backend '{self.config.backend}' is not compatible with NVFP4 quantization. "
                f"Falling back to triton_ssm."
            )
            self.config.backend = "triton_ssm"
        return super()._apply(gm, cm, factory, shared_config)


@TransformRegistry.register("insert_cached_causal_conv")
class InitializeCausalConvCache(_InsertCachedOperator):
    """A transform to handle causal conv cache operations."""


@TransformRegistry.register("insert_cached_delta_rule")
class InsertCachedDeltaRule(_InsertCachedOperator):
    """A transform to handle delta rule cache operations."""
