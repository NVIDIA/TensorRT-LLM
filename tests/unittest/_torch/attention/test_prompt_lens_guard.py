# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests of the opt-in prompt_lens host-buffer guard (TLLM_ATTN_PROMPT_LENS_GUARD).

A fake CUDA event stands in for the event recorded after the native
``prompt_lens`` copy: ``pending = True`` means the copy has not run yet when
the next ``prepare()`` is about to rewrite ``prompt_lens_cpu``.
"""

import importlib.util
import pathlib
import sys

import pytest
import torch

_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "tensorrt_llm/_torch/attention/backends/prompt_lens_guard.py"
)

try:
    from tensorrt_llm._torch.attention.backends import prompt_lens_guard as plg
except (ImportError, OSError):
    # Without the native bindings the package cannot import; the module itself
    # depends on torch only, so load it from its file.
    _SPEC = importlib.util.spec_from_file_location("prompt_lens_guard", _PATH)
    plg = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = plg
    _SPEC.loader.exec_module(plg)

pytestmark = pytest.mark.cpu_only


class FakeEvent:
    def __init__(self):
        self.pending = False
        self.records = 0
        self.syncs = 0

    def record(self):
        # The fake copy runs at once unless a test says otherwise.
        self.records += 1
        self.pending = False

    def query(self):
        return not self.pending

    def synchronize(self):
        self.syncs += 1
        self.pending = False


class Harness:
    """A guard with a fake event and captured log lines."""

    def __init__(self, mode, capturing=False):
        self.warnings = []
        self.infos = []
        self.events = []
        self.capturing = capturing
        self.guard = plg.PromptLensH2DGuard(
            mode,
            "TrtllmAttentionMetadata #0 (PP rank 3, eager)",
            warn=self.warnings.append,
            info=self.infos.append,
            pin_memory=False,
            event_factory=self._new_event,
            is_capturing=lambda: self.capturing,
        )

    def _new_event(self):
        self.events.append(FakeEvent())
        return self.events[-1]

    def prepare(self, prompt_lens, kv_lens, staged=False):
        """The guard calls of one TrtllmAttentionMetadata.prepare()."""
        prompt_lens = torch.tensor(prompt_lens, dtype=torch.int)
        outcome = self.guard.before_rewrite(prompt_lens)
        buffer = prompt_lens.clone()  # stands in for prompt_lens_cpu[:n]
        source = None
        if not staged:
            source = self.guard.copy_source(buffer)
            self.guard.after_native_copy(buffer)
        self.guard.note_kv_lens(torch.tensor(kv_lens, dtype=torch.int))
        return outcome, buffer, source

    def copy_not_run_yet(self):
        self.events[-1].pending = True


# An example: step N has requests A (first chunk 7136 after 128 cached
# tokens) and two more; step N+1 has only A's next chunk of 8192.
STEP_N_PROMPT = [7136, 1029, 0]
STEP_N_KV = [7264, 172884, 58912]
STEP_N1_PROMPT = [8192]
STEP_N1_KV = [15456]


@pytest.mark.parametrize("value", [None, "", "0", " 0 "])
def test_parse_off(value):
    assert plg.parse_guard_mode(value) is None


@pytest.mark.parametrize(
    "value,mode",
    [("probe", "probe"), ("fix", "fix"), ("fresh", "fresh"), (" FIX ", "fix"), ("Fresh", "fresh")],
)
def test_parse_modes(value, mode):
    assert plg.parse_guard_mode(value) == mode


@pytest.mark.parametrize("value", ["1", "on", "true", "yes", "fixed", "probe,fix"])
def test_parse_rejects_unknown_values(value):
    with pytest.raises(ValueError, match=plg.PROMPT_LENS_GUARD_ENV):
        plg.parse_guard_mode(value)


def test_mode_from_env_mapping():
    assert plg.guard_mode_from_env({}) is None
    assert plg.guard_mode_from_env({plg.PROMPT_LENS_GUARD_ENV: "0"}) is None
    assert plg.guard_mode_from_env({plg.PROMPT_LENS_GUARD_ENV: "probe"}) == "probe"


def _load_fresh_module(name):
    spec = importlib.util.spec_from_file_location(name, _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mode_is_read_once_at_import(monkeypatch):
    monkeypatch.setenv(plg.PROMPT_LENS_GUARD_ENV, "fresh")
    module = _load_fresh_module("prompt_lens_guard_env_fresh")
    assert module.PROMPT_LENS_GUARD_MODE == "fresh"
    monkeypatch.setenv(plg.PROMPT_LENS_GUARD_ENV, "probe")
    assert module.PROMPT_LENS_GUARD_MODE == "fresh"
    monkeypatch.delenv(plg.PROMPT_LENS_GUARD_ENV)
    assert _load_fresh_module("prompt_lens_guard_env_off").PROMPT_LENS_GUARD_MODE is None


def test_bad_env_value_fails_at_import(monkeypatch):
    monkeypatch.setenv(plg.PROMPT_LENS_GUARD_ENV, "on")
    with pytest.raises(ValueError):
        _load_fresh_module("prompt_lens_guard_env_bad")


def test_predict_rows_vcp_negative_past():
    rows = plg.predict_rows(STEP_N_PROMPT, STEP_N_KV, STEP_N1_PROMPT)
    # Row 0 would read 8192 against kv 7264 (past -928); rows 1, 2 are not
    # rewritten by a one-request step.
    assert rows == [plg.ROW_NEGATIVE_PAST, plg.ROW_SAME, plg.ROW_SAME]
    assert plg.worst_row(rows) == plg.ROW_NEGATIVE_PAST
    text = plg.describe_rows(STEP_N_PROMPT, STEP_N_KV, STEP_N1_PROMPT, rows)
    assert "row 0: q 7136 -> 8192, kv 7264, past -928 (negative past)" in text
    assert "row 1" not in text


def test_predict_rows_positive_past_mismatch():
    # An example: [57184, 334015] -> [8192, ...]; kv of row 0 is far larger.
    rows = plg.predict_rows([8171, 0], [65355, 334015], [8192])
    assert rows == [plg.ROW_MISMATCH, plg.ROW_SAME]
    assert plg.worst_row(rows) == plg.ROW_MISMATCH


def test_predict_rows_boundaries():
    # Equal is not negative: past == 0 is a full new prompt.
    assert plg.predict_rows([10], [12], [12]) == [plg.ROW_MISMATCH]
    assert plg.predict_rows([10], [12], [13]) == [plg.ROW_NEGATIVE_PAST]
    # Same length on a rewritten row is harmless.
    assert plg.predict_rows([8192, 8192], [16384, 8192], [8192, 8192, 64]) == [
        plg.ROW_SAME,
        plg.ROW_SAME,
    ]
    # The rewrite may be longer than the pending copy: only the copied rows count.
    assert plg.predict_rows([5], [5], [5, 9000]) == [plg.ROW_SAME]
    # KV lengths not recorded: a changed row is unknown, an unchanged one same.
    assert plg.predict_rows([5, 6], None, [5, 7]) == [plg.ROW_SAME, plg.ROW_UNKNOWN]
    assert plg.worst_row([plg.ROW_MISMATCH, plg.ROW_UNKNOWN]) == plg.ROW_UNKNOWN
    assert plg.worst_row([]) == plg.ROW_SAME
    assert plg.predict_rows([], [], [3]) == []
    assert plg.describe_rows([1], [1], [1], [plg.ROW_SAME]) == "no row differs"


def test_unknown_mode_rejected_by_guard():
    with pytest.raises(ValueError):
        plg.PromptLensH2DGuard("0", "x", warn=print, info=print)


def test_probe_detects_without_waiting():
    h = Harness("probe")
    assert h.prepare(STEP_N_PROMPT, STEP_N_KV)[0] is None
    assert h.infos and "1 prepares" in h.infos[0]
    h.copy_not_run_yet()
    outcome, _, _ = h.prepare(STEP_N1_PROMPT, STEP_N1_KV)
    assert outcome == plg.ROW_NEGATIVE_PAST
    assert h.events[0].syncs == 0
    assert len(h.warnings) == 1
    line = h.warnings[0]
    assert line.startswith(plg.GUARD_TAG)
    assert "PP rank 3" in line
    assert "Pending prompt_lens [7136, 1029, 0]" in line
    assert "pending kv_lens [7264, 172884, 58912]" in line
    assert "new prompt_lens [8192]" in line
    assert "past -928" in line
    assert "probe" in line
    assert h.guard.num_events == 1
    assert h.guard.num_negative_past == 1
    assert h.guard.num_waits == 0
    # One event object is reused for every copy.
    assert len(h.events) == 1 and h.events[0].records == 2


def test_fix_waits_only_when_the_copy_has_not_run():
    h = Harness("fix")
    h.prepare(STEP_N_PROMPT, STEP_N_KV)
    # Copy already ran: no event, no wait.
    assert h.prepare(STEP_N1_PROMPT, STEP_N1_KV)[0] is None
    assert h.events[0].syncs == 0 and not h.warnings
    h.copy_not_run_yet()
    assert h.prepare([4096, 100], [12288, 100])[0] == plg.ROW_MISMATCH
    assert h.events[0].syncs == 1
    assert h.guard.num_waits == 1
    assert "Waiting for the copy" in h.warnings[0]
    assert "row 0: q 8192 -> 4096, kv 15456, past 11360 (mismatch)" in h.warnings[0]


def test_fresh_copies_from_a_private_block():
    h = Harness("fresh")
    _, buffer, source = h.prepare(STEP_N_PROMPT, STEP_N_KV)
    assert source is not buffer
    assert source.data_ptr() != buffer.data_ptr()
    assert torch.equal(source, buffer)
    # Rewriting the persistent buffer does not reach the private source.
    buffer.fill_(8192)
    assert source.tolist() == STEP_N_PROMPT
    h.copy_not_run_yet()
    assert h.prepare(STEP_N1_PROMPT, STEP_N1_KV)[0] == plg.ROW_NEGATIVE_PAST
    assert h.events[0].syncs == 0
    assert "private pinned block" in h.warnings[0]


@pytest.mark.parametrize("mode", ["probe", "fix"])
def test_non_fresh_copy_source_is_the_buffer(mode):
    h = Harness(mode)
    _, buffer, source = h.prepare(STEP_N_PROMPT, STEP_N_KV)
    assert source is buffer


def test_staged_copy_keeps_the_last_native_copy():
    h = Harness("probe")
    h.prepare(STEP_N_PROMPT, STEP_N_KV)
    h.copy_not_run_yet()
    # A staged step: the window of the last native copy is still checked.
    assert h.prepare([8192, 1029, 0], [99999, 1, 1], staged=True)[0] == plg.ROW_NEGATIVE_PAST
    assert h.events[0].records == 1
    # Its KV lengths do not replace those of the pending native copy.
    assert h.prepare(STEP_N1_PROMPT, STEP_N1_KV)[0] == plg.ROW_NEGATIVE_PAST
    assert "pending kv_lens [7264, 172884, 58912]" in h.warnings[-1]


def test_kv_lens_not_recorded():
    h = Harness("probe")
    h.guard.before_rewrite(torch.tensor([5, 6], dtype=torch.int))
    h.guard.after_native_copy(torch.tensor([5, 6], dtype=torch.int))
    h.copy_not_run_yet()
    assert h.guard.before_rewrite(torch.tensor([5, 7], dtype=torch.int)) == plg.ROW_UNKNOWN
    assert "pending kv_lens not recorded" in h.warnings[0]
    assert h.guard.num_mismatch == 1


def test_capture_leaves_everything_as_it_is():
    h = Harness("fresh", capturing=True)
    outcome, buffer, source = h.prepare(STEP_N_PROMPT, STEP_N_KV)
    assert outcome is None
    assert source is buffer
    assert not h.events
    assert h.guard.num_prepares == 0 and not h.infos
    h.capturing = False
    h.prepare(STEP_N_PROMPT, STEP_N_KV)
    h.copy_not_run_yet()
    h.capturing = True
    assert h.prepare(STEP_N1_PROMPT, STEP_N1_KV)[0] is None
    assert h.events[0].records == 1 and not h.warnings


def test_warning_rate_limit_and_summary():
    h = Harness("probe")
    h.prepare([1], [1])
    for _ in range(plg._LOG_EVERY + 5):
        h.copy_not_run_yet()
        h.prepare([2], [2])
    assert h.guard.num_events == plg._LOG_EVERY + 5
    # The first _LOG_FIRST events, then every _LOG_EVERY-th.
    assert len(h.warnings) == plg._LOG_FIRST + 1
    assert h.guard.num_mismatch + h.guard.num_negative_past == 1
    for _ in range(plg._SUMMARY_EVERY):
        h.prepare([2], [2])
    assert len(h.infos) == 2
    assert f"{plg._SUMMARY_EVERY} prepares" in h.infos[1]
    assert f"{plg._LOG_EVERY + 5} open windows" in h.infos[1]


def test_metadata_default_off():
    trtllm = pytest.importorskip(
        "tensorrt_llm._torch.attention.backends.trtllm", exc_type=ImportError
    )
    if trtllm.PROMPT_LENS_GUARD_MODE is not None:
        pytest.skip("TLLM_ATTN_PROMPT_LENS_GUARD is set in this environment")
    assert trtllm.TrtllmAttentionMetadata._prompt_lens_guard is None
    assert trtllm._make_prompt_lens_guard(object()) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("mode", ["probe", "fix", "fresh"])
def test_real_event_and_copy(mode):
    warnings = []
    guard = plg.PromptLensH2DGuard(
        mode, "gpu", warn=warnings.append, info=lambda _: None, pin_memory=True
    )
    buffer = torch.empty(4, dtype=torch.int, pin_memory=True)
    device = torch.zeros(4, dtype=torch.int, device="cuda")
    for step in ([3, 4, 5, 6], [7, 8, 9, 10]):
        values = torch.tensor(step, dtype=torch.int)
        guard.before_rewrite(values)
        buffer.copy_(values)
        source = guard.copy_source(buffer)
        assert (source is buffer) == (mode != "fresh")
        device.copy_(source, non_blocking=True)
        guard.after_native_copy(buffer)
        guard.note_kv_lens(values)
    torch.cuda.synchronize()
    assert device.tolist() == [7, 8, 9, 10]
    assert guard.num_native_copies == 2
