# """
# DEPRECATED: This file has been deprecated and all functionality has been moved to:
# tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py
#
# This file is kept temporarily for verification purposes and will be removed after testing.
# DO NOT USE THIS FILE - import from transform.library.sharding instead.
# """

# """Sharding config definitions for the inference optimizer."""
#
# import math
# import operator
# import re
# from abc import ABC, abstractmethod
# from enum import Enum, IntEnum
# from functools import partial
# from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple
#
# if TYPE_CHECKING:
#     from ..transform.library.sharding import ShardingTransformConfig
#
# import torch
# import torch.nn as nn
# from pydantic import BaseModel, ConfigDict, Field
# from torch.fx import GraphModule, Node
#
# from ..models.factory import ShardingConfigSource
# from ..utils.logger import ad_logger
# from .node_utils import (
#     bfs,
#     extract_param_names_from_node,
#     is_any_lin_op,
#     is_op,
#     num_users_of_weight_node,
#     subgraph,
# )
# from .quantization_utils import (
#     cutlass_fp4_scale_to_modelopt_fp4_scale,
#     modelopt_fp4_scale_to_cutlass_fp4_scale,
# )
#
#
# # ALL CONTENT BELOW HAS BEEN MOVED TO transform/library/sharding.py
# # This file will be deleted after verification
