#!/usr/bin/env bash
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
#
# Control matrix for the conformance checks: replay a few corpus entries against the fake server in
# every mode and confirm that `ok` passes everything and each fault mode fails exactly its check.
# Run it before trusting a verdict, and after editing any check.
#
# usage: run_controls.sh <corpus.jsonl> [out_dir] [python]
# The corpus comes from ../build_replay_corpus.py (any recorded
# Responses requests; the fake server only reads their tool declarations).
set -u
CORPUS=${1:?corpus.jsonl}
OUT=${2:-/tmp/conformance_controls}
PY=${3:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
REPLAY=$HERE/../replay_requests.py
PORT=${CONTROLS_PORT:-8399}
mkdir -p "$OUT"
"$PY" "$HERE/fake_responses_server.py" --port "$PORT" > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
sleep 3
# mode : the check that must fail (stream checks or call checks)
EXPECT="ok:-
drop_part_done:unpaired_content_part
snapshot_text_differs:item_content_differs
delta_differs:delta_done_mismatch
no_terminal:no_terminal_event
custom_as_function:type_mismatch
two_calls:ptc_violation
prefixed_name:prefixed_name_of_declared
overlength:-"
bad=0
while IFS=: read -r mode want; do
  rm -rf "$OUT/$mode"
  "$PY" "$REPLAY" --base-url "http://127.0.0.1:$PORT" --out-dir "$OUT/$mode" --concurrency 4 --limit 5 \
    --header "x-fake-mode: $mode" "$CORPUS" > "$OUT/$mode.replay.log" 2>&1
  "$PY" "$HERE/check_stream.py" --out "$OUT/$mode.stream.json" "$OUT/$mode" > /dev/null 2>&1
  "$PY" "$HERE/check_tool_calls.py" --out "$OUT/$mode.calls.json" "$OUT/$mode" > /dev/null 2>&1
  "$PY" - "$OUT" "$mode" "$want" <<'PY' || bad=1
import json, sys
out, mode, want = sys.argv[1:]
failed = set()
for kind in ("stream", "calls"):
    report = json.load(open(f"{out}/{mode}.{kind}.json"))
    failed |= {k for k, v in report["verdicts"].items() if v == "FAIL"}
expected = set() if want == "-" else {want}
ok = expected <= failed and (want != "-" or not failed)
print(f"{'OK ' if ok else 'BAD'} {mode:22s} expected={sorted(expected)} failed={sorted(failed)}")
sys.exit(0 if ok else 1)
PY
done <<< "$EXPECT"
exit $bad
