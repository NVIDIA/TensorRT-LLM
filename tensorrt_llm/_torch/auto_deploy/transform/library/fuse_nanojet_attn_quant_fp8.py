"""Let the nanojet attention quantize in its epilogue when its reader wants e4m3.

Mirrors ``fuse_trtllm_attn_quant_fp8`` for the nanojet backend: the scale is recorded on the
source attention node, which ``NanojetAttention.get_constants`` reads when the cached node is
built. Recording it here, while shapes are still propagated, is also what lets the following
fusions see an e4m3 activation.
"""

from typing import Tuple, Type

import torch
from torch.fx import GraphModule

from ...custom_ops.attention.nanojet_attention import NANOJET_ATTENTION_INPUT_SCALE
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.nanojet_graph import per_tensor_scale, set_val_meta
from ...utils.node_utils import (
    collect_terminal_users_through_passthrough,
    get_shared_input_scale_for_fp8_linears,
    is_op,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class FuseNanojetAttnQuantFP8Config(TransformConfig):
    """Configuration for quantizing the nanojet attention output in its epilogue."""


@TransformRegistry.register("fuse_nanojet_attn_quant_fp8")
class FuseNanojetAttnQuantFP8(BaseTransform):
    """Fold the quantize before ``o_proj`` into the attention that feeds it."""

    config: FuseNanojetAttnQuantFP8Config

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseNanojetAttnQuantFP8Config

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not self.config.enabled:
            return gm, TransformInfo(skipped=True, num_matches=0)

        num_matches = 0
        for attn_node in list(gm.graph.nodes):
            if not is_op(attn_node, torch.ops.auto_deploy.torch_attention.default):
                continue
            attn_node.meta.pop(NANOJET_ATTENTION_INPUT_SCALE, None)

            readers, traversal_ok = collect_terminal_users_through_passthrough(attn_node)
            fp8_readers, scale = get_shared_input_scale_for_fp8_linears(readers)
            if not (traversal_ok and fp8_readers and len(fp8_readers) == len(readers)):
                continue
            if per_tensor_scale(gm, scale) is None:
                continue

            attn_node.meta[NANOJET_ATTENTION_INPUT_SCALE] = scale
            # The epilogue changes what this node produces, so say so: the fusions after this
            # one decide by the dtype of the activation they read.
            set_val_meta(attn_node, attn_node, dtype=torch.float8_e4m3fn)
            num_matches += 1

        if num_matches:
            ad_logger.info(
                f"fuse_nanojet_attn_quant_fp8: {num_matches} attention outputs quantized "
                "in the epilogue"
            )

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=True,
            has_valid_shapes=True,
        )
