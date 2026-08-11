# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading cost-table samples out of production iteration logs.

The in-process sweep measures an engine the profiler built itself, and when
that engine's configuration drifts from the deployment the table describes a
machine nobody is running -- a swept (bs=256, M=1536) cell read 150.2 ms
against 118.8 ms measured live at the same shape, with nothing in the
fingerprint to record the difference. ``print_iter_log`` already emits the row
count, the token total and the host step time for every step of the real
deployment, so a table can be fitted from traffic that actually ran.

What must hold: the token total identifies the verify length exactly (no
clustering, no guessing), steps that cannot be filed honestly are dropped
rather than mislabelled, and the first step of each executor instance -- whose
host time contains the wait for the first request, measured at 62 s live -- is
never treated as a measurement.
"""

import importlib.util
import pathlib
import sys
import tempfile

_SPEC = importlib.util.spec_from_file_location(
    "dspark_sps_profiler",
    pathlib.Path(__file__).resolve().parents[4] / "microbenchmarks" /
    "dspark_sps_profiler.py")
_PROFILER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROFILER   # @dataclass needs the module registered
_SPEC.loader.exec_module(_PROFILER)

samples_from_iter_log = _PROFILER.samples_from_iter_log
resolve_padded_shape = _PROFILER.resolve_padded_shape

# A deployment's captured ladder, abbreviated. Padding rounds a step's rows up
# to the smallest entry at or above them, which is what turns a padded token
# total into exactly one verify length.
LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256]


def _line(iteration, bs, step_ms, gen_tokens, rank="0"):
    return (f"{rank}: [08/04/2026-19:08:13] [TRT-LLM] [I] [_torch][RANK 0] "
            f"iter = {iteration}, global_rank = 0, rank = 0, "
            f"num_scheduled_requests = {bs}, kv_cache_util = 0.125, "
            f"currank_total_requests = 1/8, host_step_time = {step_ms}ms, "
            f"prev_device_step_time = {step_ms}ms, "
            f"timestamp = 2026-08-04 19:08:13, "
            f"states = {{'num_ctx_requests': 0, 'num_ctx_tokens': 0, "
            f"'num_generation_tokens': {gen_tokens}, "
            f"'cached_kv_tokens': 63821}}")


def _write(lines, name="gen.log"):
    # tempfile rather than pytest's tmp_path: these tests also run under the
    # fixture-less in-container runner used for the GPU images.
    path = pathlib.Path(tempfile.mkdtemp()) / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_verify_length_comes_from_the_padded_token_total():
    """The label is read out of the log, never inferred from step times.

    Steps the ladder cannot explain are skipped, not mislabelled: filing a
    step under a shape it did not run is the mislabelling that once shipped
    eight fictional cells with near-zero residuals.
    """
    log = _write([
        _line(1, 1, 62098.0, 6),        # executor's first step: dropped
        _line(2, 244, 79.2, 768),       # 244 rows padded to 256 -> rung-2
        _line(3, 242, 117.1, 1536),     # padded to 256 -> full block
        _line(4, 128, 61.0, 384),       # exactly 128 rows -> rung-2
        _line(5, 256, 300.0, 2048),     # L = 7, past the block: skipped
        _line(6, 300, 400.0, 1800),     # more rows than the ladder holds: skipped
    ])
    samples = samples_from_iter_log([log], max_draft_len=5,
                                    padded_batch_sizes=LADDER)
    got = sorted((s.batch_size, s.verify_len, s.step_time_ms)
                 for s in samples)
    assert got == [(128, 2, 61.0), (256, 2, 79.2), (256, 5, 117.1)]
    # The table is indexed by the padded token total, which each sample exposes.
    assert sorted(s.total_verify_tokens for s in samples) == [384, 768, 1536]


def test_the_padded_width_is_what_gets_recorded():
    """A 244-row step cost what 256 rows cost; the table is indexed by that.

    The ladder is also what removes the ambiguity: 768 tokens over 128 rows
    is L=5 at 128, L=3 at 192, or L=2 at 256 -- without the ladder all three
    are admissible and the step is dropped; with it, padding rounds 128 up to
    128 and the answer is L=5 (the first version of this collector kept 312
    of 5682 production steps for lack of this). And a step whose token total
    the padded width cannot explain (_get_padded_batch can bail) is dropped:
    nothing in the log says padding declined, so that is the only honest move.
    """
    assert resolve_padded_shape(num_rows=244, num_generation_tokens=768,
                                max_draft_len=5, max_batch_size=256,
                                padded_batch_sizes=LADDER) == (256, 2)
    assert resolve_padded_shape(num_rows=128, num_generation_tokens=768,
                                max_draft_len=5, max_batch_size=256) is None
    assert resolve_padded_shape(num_rows=128, num_generation_tokens=768,
                                max_draft_len=5, max_batch_size=256,
                                padded_batch_sizes=LADDER) == (128, 5)
    assert resolve_padded_shape(num_rows=200, num_generation_tokens=600,
                                max_draft_len=5, max_batch_size=256,
                                padded_batch_sizes=LADDER) is None


def test_the_first_step_of_each_executor_is_dropped():
    """A build spins up more than one executor, and each numbers from 1.

    The first step of each carries the wait for its first request inside
    host_step_time (62 s on a live server), which would otherwise enter the
    fit as a cell measurement.
    """
    log = _write([
        _line(1, 1, 200.0, 6),          # KV-estimation executor
        _line(2, 256, 79.0, 768),
        _line(1, 1, 62098.0, 6),        # real executor starts over
        _line(2, 256, 79.4, 768),
    ])
    times = [s.step_time_ms for s in samples_from_iter_log([log], max_draft_len=5)]
    assert times == [79.0, 79.4]


def test_only_the_requested_rank_is_read():
    """Every rank logs; counting all of them would multiply every cell.

    Outlier steps are bounded out on the same pass.
    """
    log = _write([
        _line(2, 256, 79.2, 768, rank="0"),
        _line(2, 256, 79.9, 768, rank="3"),   # another rank's copy: dropped
        _line(3, 256, 5000.0, 768),           # outlier past max_step_ms
    ])
    times = [s.step_time_ms for s in
             samples_from_iter_log([log], max_draft_len=5, max_step_ms=1000.0)]
    assert times == [79.2]


def test_several_logs_are_pooled():
    """One arm gives one rung; the table needs the ladder, so pool the runs."""
    a = _write([_line(1, 1, 100.0, 6), _line(2, 256, 117.1, 1536)],
               name="a.log")
    c = _write([_line(1, 1, 100.0, 6), _line(2, 256, 79.2, 768)],
               name="c.log")
    got = sorted(s.verify_len for s in
                 samples_from_iter_log([a, c], max_draft_len=5))
    assert got == [2, 5]


def test_a_log_without_the_states_dict_yields_nothing():
    """enable_iter_perf_stats off: fail loudly upstream, not with a bad table."""
    path = pathlib.Path(tempfile.mkdtemp()) / "bare.log"
    path.write_text(
        "0: [TRT-LLM] iter = 2, num_scheduled_requests = 256, "
        "host_step_time = 79.2ms\n")
    assert samples_from_iter_log([str(path)], max_draft_len=5) == []
