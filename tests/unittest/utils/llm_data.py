import os
import sys

# Ensure tests/ directory is in path for test_common imports
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from test_common.llm_data import llm_datasets_root, llm_models_root

__all__ = [
    "llm_datasets_root",
    "llm_models_root",
]
