"""Shim for making AutoDeploy models run in the same way as other models."""

from .ad_executor import create_autodeploy_executor
from .demollm import DemoLLM
from .interface import CachedSequenceInterface, GetInferenceModel
