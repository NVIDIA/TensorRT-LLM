# Ported from https://github.com/Lightricks/LTX-2
# packages/ltx-core/src/ltx_core/model/audio_vae/causality_axis.py

from enum import Enum


class CausalityAxis(Enum):
    NONE = None
    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_COMPATIBILITY = "width-compatibility"
