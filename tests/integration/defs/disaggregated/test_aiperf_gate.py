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
"""GPU-free unit tests for the aiperf error-rate gate.

The gate (enforce_aiperf_error_rate) is applied by default to every
disaggregated stress config, so these synthetic profile_export.jsonl cases
prove it (a) fires on the server-error storm it exists to catch, (b) passes a
healthy run with intentional cancellations, and (c) refuses to treat a broken
or implausible export as a clean pass. Run with:

    pytest -sv disaggregated/test_aiperf_gate.py
"""

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest
from test_disaggregated import enforce_aiperf_error_rate


def _record(error: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Build a minimal aiperf MetricRecordInfo-shaped request record.

    Args:
        error: Optional ErrorDetails-shaped object ({"code", "type",
            "message"}) attached to the record.

    Returns:
        A dict with "metadata" and "metrics" keys, plus "error" when given.
    """
    rec: dict[str, Any] = {
        "metadata": {"x_request_id": "id"},
        "metrics": {"request_latency": 1.0},
    }
    if error is not None:
        rec["error"] = error
    return rec


_CANCEL = {
    "code": 499,
    "type": "RequestCancellationError",
    "message": "Request cancelled 0.500s after being sent",
}
_SERVER_500 = {
    "code": 500,
    "type": "InternalServerError",
    "message": '{"detail":"Internal server error Cluster is not ready"}',
}


def _write_export(
    tmp_path: Path,
    records: Sequence[dict[str, Any]],
    raw_lines: Sequence[str] = (),
) -> str:
    """Write a synthetic profile_export.jsonl into tmp_path.

    Args:
        tmp_path: Directory to write the export into (pytest tmp_path).
        records: Records serialized one-per-line as JSON.
        raw_lines: Extra lines appended verbatim (e.g. corrupt/truncated).

    Returns:
        The artifact directory path to pass to enforce_aiperf_error_rate.
    """
    export = tmp_path / "profile_export.jsonl"
    with open(export, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        for line in raw_lines:
            f.write(line + "\n")
    return str(tmp_path)


def test_fires_on_error_storm(tmp_path: Path) -> None:
    """Replay of the nvbugs/6472256 CI failure distribution: must fire."""
    records = (
        [_record(_CANCEL)] * 3038
        + [_record(_SERVER_500)] * 4359
        + [_record()] * (35000 - 3038 - 4359)
    )
    artifact_dir = _write_export(tmp_path, records)
    with pytest.raises(AssertionError, match="exceeds threshold"):
        enforce_aiperf_error_rate(artifact_dir, 0.05, expected_records=35000)


def test_passes_healthy_run_with_cancellations(tmp_path: Path) -> None:
    """A clean run with 10% intentional cancellations passes the 5% gate."""
    records = [_record()] * 898 + [_record(_CANCEL)] * 100 + [_record(_SERVER_500)] * 2
    artifact_dir = _write_export(tmp_path, records)
    enforce_aiperf_error_rate(artifact_dir, 0.05, expected_records=1000)


def test_missing_export_raises(tmp_path: Path) -> None:
    """A missing export file raises FileNotFoundError with the path."""
    with pytest.raises(FileNotFoundError, match=r"profile_export\.jsonl"):
        enforce_aiperf_error_rate(str(tmp_path), 0.05)


def test_empty_export_fails(tmp_path: Path) -> None:
    """An export with zero request records must not read as a clean pass."""
    artifact_dir = _write_export(tmp_path, [])
    with pytest.raises(AssertionError, match="no parseable request records"):
        enforce_aiperf_error_rate(artifact_dir, 0.05)


def test_all_cancelled_fails(tmp_path: Path) -> None:
    """An all-cancelled record set indicates a broken run and must fail."""
    artifact_dir = _write_export(tmp_path, [_record(_CANCEL)] * 50)
    with pytest.raises(AssertionError, match=r"classified as \W*cancelled"):
        enforce_aiperf_error_rate(artifact_dir, 0.05)


def test_corrupt_export_fails(tmp_path: Path) -> None:
    """Wholesale parse failure (format change) must not read as a clean run."""
    artifact_dir = _write_export(tmp_path, [_record()] * 10, raw_lines=["{not json"] * 10)
    with pytest.raises(AssertionError, match="failed to parse"):
        enforce_aiperf_error_rate(artifact_dir, 0.05)


def test_single_truncated_line_tolerated(tmp_path: Path) -> None:
    """One partial trailing line (killed writer) does not fail the gate."""
    artifact_dir = _write_export(tmp_path, [_record()] * 200, raw_lines=['{"metadata": {"x_req'])
    enforce_aiperf_error_rate(artifact_dir, 0.05, expected_records=200)


def test_incomplete_accounting_fails(tmp_path: Path) -> None:
    """Far fewer records than requests => refuse to compute a rate."""
    artifact_dir = _write_export(tmp_path, [_record()] * 100)
    with pytest.raises(AssertionError, match=r"accounting is \W*incomplete"):
        enforce_aiperf_error_rate(artifact_dir, 0.05, expected_records=1000)


def test_gate_disabled_paths_not_affected(tmp_path: Path) -> None:
    """expected_records=None skips the plausibility check (dataset-entry runs)."""
    artifact_dir = _write_export(tmp_path, [_record()] * 5)
    enforce_aiperf_error_rate(artifact_dir, 0.05, expected_records=None)


def test_was_cancelled_metadata_fallback(tmp_path: Path) -> None:
    """Cancellations are excluded even if the error shape drifts.

    A future aiperf may record cancellations with a different error type, a
    null code, or no error object at all; metadata.was_cancelled still
    classifies them as intentional cancellations rather than server errors.
    """
    drifted_error = {"code": None, "type": "ClientDisconnected", "message": "x"}
    records = [_record()] * 900
    for rec_error in ([drifted_error] * 50, [None] * 50):
        for err in rec_error:
            rec = _record(err)
            rec["metadata"]["was_cancelled"] = True
            records.append(rec)
    artifact_dir = _write_export(tmp_path, records)
    # 100 drifted cancellations at 10% must not trip the 5% threshold.
    enforce_aiperf_error_rate(artifact_dir, 0.05, expected_records=1000)


def test_non_request_records_excluded_from_denominator(tmp_path: Path) -> None:
    """Records without metrics/error (future metadata lines) do not dilute the rate."""
    records = [_record()] * 90 + [_record(_SERVER_500)] * 10
    non_request = [{"summary": {"total": 100}}] * 900
    artifact_dir = _write_export(tmp_path, records + non_request)
    # 10 errors over 100 requests = 10% — must fire even though 900 metadata
    # lines would dilute it to ~1% if they were counted as requests.
    with pytest.raises(AssertionError, match="exceeds threshold"):
        enforce_aiperf_error_rate(artifact_dir, 0.05, expected_records=100)
