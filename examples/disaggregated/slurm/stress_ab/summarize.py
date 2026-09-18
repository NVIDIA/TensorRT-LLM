# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reconcile the four fresh Slurm steps without turning missing data into a pass."""

import argparse
import csv
import json
from pathlib import Path

ORDER = ["control", "treatment", "treatment", "control"]


def summarize(root: Path) -> dict:
    """Return status and per-arm counts; no automatic causal claim is made."""
    with (root / "slurm-steps.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    step_rows = {int(row["index"]): row for row in rows}
    if len(step_rows) != len(rows) or set(step_rows) - {1, 2, 3, 4}:
        raise ValueError("Duplicate or unexpected step records")
    trials = []
    counts = {arm: dict.fromkeys(("pass", "fail", "invalid"), 0) for arm in set(ORDER)}
    for index, arm in enumerate(ORDER, 1):
        path = root / f"{index:02d}-{arm}" / "summary.json"
        step = step_rows.get(index)
        status = "invalid"
        reason = "Missing Slurm step or trial summary"
        if path.is_file() and step:
            data = json.loads(path.read_text())
            details = data.get("trials", [])
            if (
                len(details) == 1
                and details[0].get("arm") == arm
                and step["arm"] == arm
                and data.get("status") in ("pass", "fail", "invalid")
                and details[0].get("status") == data["status"]
            ):
                status = data["status"]
                expected_code = {"pass": 0, "fail": 1, "invalid": 2}[status]
                if int(step["exit_code"]) != expected_code:
                    status = "invalid"
                    reason = "Slurm step exit disagrees with trial summary"
                else:
                    reason = details[0].get("reason", "")
            else:
                reason = "Summary does not contain the expected single arm"
        counts[arm][status] += 1
        trials.append(
            {"index": index, "arm": arm, "status": status, "reason": reason, "summary": str(path)}
        )
    status = "pass"
    if any(item["status"] == "invalid" for item in trials):
        status = "invalid"
    elif any(item["status"] == "fail" for item in trials):
        status = "fail"
    return {
        "schema_version": 1,
        "status": status,
        "counts": counts,
        "trials": trials,
        "interpretation": "Compare initiating worker errors and request accounting; no causal verdict is automatic.",
    }


def main() -> int:
    """Write a persistent matrix result and preserve non-passing exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_root)
    (args.run_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return {"pass": 0, "fail": 1, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
