"""The transform passes to capture the hidden states of the target model."""

from typing import Dict, List, Optional, Set, Tuple, Type

import torch
from torch._ops import OpOverloadPacket
from torch.fx import GraphModule, Node

from ...custom_ops.attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BufferInitializerDict,
    CacheConfig,
    CacheInitializerDict,
    Constant,
    MHACallable,
    PrepareMetadataCallable,
    SequenceInfo,
)
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import get_all_layer_subgraphs, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from .kvcache import InsertCachedAttention


@torch.library.custom_op("auto_deploy::residual_add_for_capture", mutates_args=())
def residual_add_for_capture(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.add(t1, t2)


@residual_add_for_capture.register_fake
def residual_add_for_capture_fake(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.add(t1, t2)


@torch.library.custom_op("auto_deploy::cached_residual_add", mutates_args=())
def cached_residual_add(
    # INPUTS
    t1: torch.Tensor,
    t2: torch.Tensor,
    # METADATA
    #
    # CACHES
    hidden_states_cache: torch.Tensor,
    # CONSTANTS
    #
) -> torch.Tensor:
    ret = torch.ops.aten.add(t1, t2)
    b, s, _ = ret.shape
    num_tokens = b * s
    hidden_states_cache[:num_tokens].copy_(ret.view(num_tokens, -1), non_blocking=True)
    return ret


@cached_residual_add.register_fake
def cached_residual_add_fake(
    t1: torch.Tensor,
    t2: torch.Tensor,
    # METADATA
    #
    # CACHES
    hidden_states_cache: torch.Tensor,
    # CONSTANTS
    #
) -> torch.Tensor:
    return torch.ops.aten.add(t1, t2)


@torch.library.custom_op("auto_deploy::cached_residual_add_prepare_metadata", mutates_args=())
def cached_residual_add_prepare_metadata(
    position_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    slot_idx: torch.Tensor,
    page_size: int,
    chunk_size: int,
) -> List[torch.Tensor]:
    return []


@cached_residual_add_prepare_metadata.register_fake
def cached_residual_add_prepare_metadata_fake(
    position_ids,
    seq_len,
    input_pos,
    cache_loc,
    pages_per_seq,
    slot_idx,
    page_size,
    chunk_size,
):
    return []


class DetectHiddenStatesForCaptureConfig(TransformConfig):
    """Configuration for the hidden states detection transform."""

    # TODO: figure out how to get the layers to capture...
    # Right now, it seems the default is None and then EagleSpecMetadata has a heuristic to extract
    # the layers indices to capture. This seems fragile. We should consider if we can use the layer
    # indices stored in the eagle checkpoints, e.g.,
    # https://huggingface.co/nvidia/gpt-oss-120b-Eagle3/blob/main/config.json#L9-L14
    # TODO: just used for testing, remove later
    eagle3_layers_to_capture: Optional[Set[int]] = {2, 10, 12}  # None


@TransformRegistry.register("detect_hidden_states_for_capture")
class DetectHiddenStatesForCapture(BaseTransform):
    """Detect the hidden states for capture in the graph."""

    config: DetectHiddenStatesForCaptureConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return DetectHiddenStatesForCaptureConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # nothing to do if no layers to capture
        if not self.config.eagle3_layers_to_capture:
            info = TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)
            return gm, info

        def _get_layer_number(lin_node: Node) -> Optional[int]:
            weight = lin_node.args[1]
            if weight.op == "get_attr":
                subnames = weight.target.split(".")
                for subname in subnames:
                    if subname.isdigit():
                        return int(subname)
            return None

        # find last closing linear node of each layer
        # from there we will find the residual add node for that layer
        layer_subgraphs, unprocessed_linear_nodes = get_all_layer_subgraphs(gm)
        residual_add_nodes: Dict[int, Node] = {}
        for _, _, lin_node_closing in layer_subgraphs:
            # need layer number to correctly identify the residual add node
            layer_number = _get_layer_number(lin_node_closing)
            if layer_number is None or layer_number not in self.config.eagle3_layers_to_capture:
                continue

            # Conditions to identify as the hidden states after the residual
            # 1. add node with >1 users
            # 2. last add node in 1 user chain (for last layer or layers with to)
            res_node = lin_node_closing
            while len(res_node.users) == 1:
                user_node = list(res_node.users)[0]
                if not is_op(user_node, torch.ops.aten.add):
                    break
                res_node = user_node

            if is_op(res_node, torch.ops.aten.add):
                # this naturally store the last residual add node encountered for each layer
                residual_add_nodes[layer_number] = res_node

        # check that we have captured all desired layers
        assert residual_add_nodes.keys() == self.config.eagle3_layers_to_capture, (
            f"Expected layers to capture: {self.config.eagle3_layers_to_capture}, "
            f"but got: {residual_add_nodes.keys()}"
        )

        # now replace resaidual add node with a special placeholder node
        for _, res_node in residual_add_nodes.items():
            with gm.graph.inserting_before(res_node):
                new_node = gm.graph.call_function(
                    torch.ops.auto_deploy.residual_add_for_capture.default,
                    args=res_node.args,
                    kwargs=res_node.kwargs,
                )
            res_node.replace_all_uses_with(new_node)
            gm.graph.erase_node(res_node)

        cnt = len(residual_add_nodes)
        info = TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=cnt == 0
        )

        return gm, info


@AttentionRegistry.register("cached_residual_add")
class CachedResidualAdd(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        return True

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 2

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.residual_add_for_capture

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.cached_residual_add

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        return torch.ops.auto_deploy.cached_residual_add_prepare_metadata, 0

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        hidden_size = source_attn_node.meta["val"].shape[-1]
        hidden_type = source_attn_node.meta["val"].dtype

        def _get_hidden_states_cache(si: SequenceInfo):
            return torch.empty(
                si.max_num_tokens,
                hidden_size,
                device=si.device,
                dtype=hidden_type,
            )

        return {"hidden_states_cache": _get_hidden_states_cache}

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        return {}

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        return []


@TransformRegistry.register("insert_cached_residual_add")
class InsertCachedResidualAdd(InsertCachedAttention):
    """A transform to handle residual add cache operations."""
