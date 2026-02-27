# Ported from https://github.com/Lightricks/LTX-2
# packages/ltx-core/src/ltx_core/model/video_vae/enums.py

from enum import Enum


class NormLayerType(Enum):
    GROUP_NORM = "group_norm"
    PIXEL_NORM = "pixel_norm"


class PaddingModeType(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"
