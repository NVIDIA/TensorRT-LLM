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
# Run N real Codex sessions against a self-hosted Responses endpoint, CONCURRENCY at a time.
#
# usage: run_codex_sessions.sh <tasks.tsv> <out_dir>
#   tasks.tsv: one task per line, "<source dir>\t<prompt>". The source dir is copied into a fresh
#              per-session workspace; use "-" for an empty one.
# env:
#   BASE_URL     endpoint base, e.g. http://127.0.0.1:8000/v1   (required)
#   MODEL        model name to send (default: served)
#   CONCURRENCY  parallel sessions (default 4)
#   TIMEOUT_S    per-session wall clock (default 3600)
#   CODEX_BIN    codex executable (default: codex on PATH)
#   SANDBOX      codex sandbox mode: read-only | workspace-write | danger-full-access (default workspace-write).
#                danger-full-access only inside a disposable container or VM.
#   CONTEXT_WINDOW  optional model_context_window to tell Codex (it compacts when its count reaches it)
#   EFFORT       optional model_reasoning_effort
#   EXTRA_ARGS   extra arguments for `codex exec`
#
# Every session gets its own CODEX_HOME, so the user's ~/.codex (config, auth, history) is never read
# or written. The endpoint key is a dummy unless TRTLLM_API_KEY is set. Flags were checked against
# codex-cli 0.155.1: re-check `codex exec --help` for other versions.
#
# Output: <out_dir>/sNNNN/{events.jsonl,stderr.log,last_message.txt,workspace/} and <out_dir>/sessions.tsv
# (index, exit code, seconds). Harness-side signals come from the SERVER's request trace:
# run the endpoint with TRTLLM_REQUEST_TRACE_DIR set, then harness_signals.py and the conformance checks.
set -u
TASKS=${1:?tasks.tsv}
OUT=${2:?out_dir}
: "${BASE_URL:?set BASE_URL, e.g. http://127.0.0.1:8000/v1}"
MODEL=${MODEL:-served}
CONCURRENCY=${CONCURRENCY:-4}
TIMEOUT_S=${TIMEOUT_S:-3600}
CODEX_BIN=${CODEX_BIN:-codex}
SANDBOX=${SANDBOX:-workspace-write}
export TRTLLM_API_KEY=${TRTLLM_API_KEY:-dummy-local-key}
mkdir -p "$OUT"
: > "$OUT/sessions.tsv"

run_one() {
  local index=$1 src=$2 prompt=$3
  local dir
  dir=$(printf '%s/s%04d' "$OUT" "$index")
  mkdir -p "$dir/home" "$dir/workspace"
  if [ "$src" != "-" ]; then cp -a "$src/." "$dir/workspace/"; fi
  local extra=()
  [ -n "${CONTEXT_WINDOW:-}" ] && extra+=(-c "model_context_window=${CONTEXT_WINDOW}")
  [ -n "${EFFORT:-}" ] && extra+=(-c "model_reasoning_effort=\"${EFFORT}\"")
  local start=$SECONDS
  # shellcheck disable=SC2086
  CODEX_HOME="$dir/home" timeout "$TIMEOUT_S" "$CODEX_BIN" exec --json --skip-git-repo-check \
    -C "$dir/workspace" -s "$SANDBOX" -m "$MODEL" \
    -c 'model_provider="trtllm"' \
    -c "model_providers.trtllm={name=\"trtllm-serve\", base_url=\"${BASE_URL}\", wire_api=\"responses\", env_key=\"TRTLLM_API_KEY\"}" \
    "${extra[@]}" ${EXTRA_ARGS:-} -o "$dir/last_message.txt" "$prompt" \
    > "$dir/events.jsonl" 2> "$dir/stderr.log" < /dev/null
  local rc=$?
  printf '%d\t%d\t%d\n' "$index" "$rc" "$((SECONDS - start))" >> "$OUT/sessions.tsv"
}

index=0
while IFS=$'\t' read -r src prompt; do
  [ -z "${src}${prompt}" ] && continue
  index=$((index + 1))
  run_one "$index" "$src" "$prompt" &
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 1; done
done < "$TASKS"
wait
echo "sessions: $index; exit codes: $(cut -f2 "$OUT/sessions.tsv" | sort | uniq -c | tr '\n' ' ')"
