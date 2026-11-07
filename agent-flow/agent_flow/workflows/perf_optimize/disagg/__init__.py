"""Disaggregated serving support for perf-optimize.

Three ways to optimize a disaggregated deployment live here, by module:

- :mod:`.harness` -- the ``disagg:`` block. One campaign measures the whole
  cluster end to end through the TensorRT-LLM checkout's own harness.
- :mod:`.sol_track` and :mod:`.bench_cli` -- the ``sol_track:`` block. One
  campaign optimizes one half (ctx or gen) at a fixed operating point,
  measured through the ``ibc-bench`` harness.
- :mod:`.sol`, :mod:`.spawn` and :mod:`.sweep_design` -- the ``disagg_sol:``
  block. A supervisor establishes the operating point from a measured design,
  then starts one ``sol_track`` campaign per half.

The ``disagg:`` block's names are re-exported here because they predate this
package -- they lived in ``perf_optimize/disagg.py`` -- and every existing
``from .disagg import has_disagg`` keeps working unchanged.
"""

from .harness import (
    DISAGG_CONFIG_KEY,
    DISAGG_FIELD,
    DISAGG_PROFILE_METHODS,
    DisaggConfigError,
    apply_harness_conditions,
    disagg_config_path,
    has_disagg,
    load_disagg_config,
    user_set_benchmark_keys,
    worker_config_yaml,
)

__all__ = [
    "DISAGG_CONFIG_KEY",
    "DISAGG_FIELD",
    "DISAGG_PROFILE_METHODS",
    "DisaggConfigError",
    "apply_harness_conditions",
    "disagg_config_path",
    "has_disagg",
    "load_disagg_config",
    "user_set_benchmark_keys",
    "worker_config_yaml",
]
