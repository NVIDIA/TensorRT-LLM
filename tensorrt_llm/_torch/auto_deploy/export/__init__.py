"""AutoDeploy's modular export patch system."""

from . import library  # ensure all patches are registered
from .export import *
from .interface import *
