# TODO: When getting rid of the nemotron H patches, import `modeling_nemotron_h` here to ensure the
# custom model implementation is registered.
from . import custom, hf, nemotron_flash, patches
from .factory import *
