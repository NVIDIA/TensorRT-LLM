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
#              per-session workspace; use "-" for an empty one. Blank lines and lines starting
#              with "#" are skipped.
# env:
#   BASE_URL     endpoint base, e.g. http://127.0.0.1:8000/v1   (required)
#   MODEL        model name to send (default: served)
#   CONCURRENCY  parallel sessions (default 4)
#   TIMEOUT_S    per-session wall clock (default 3600)
#   CODEX_BIN    codex executable (default: codex on PATH)
#   SANDBOX      codex sandbox mode: read-only | workspace-write | danger-full-access (default
#                workspace-write); danger-full-access only inside a disposable container or VM.
#                Codex's own sandbox needs Linux namespaces: in a container that forbids them every
#                command fails (exit 101, or "bwrap: No permissions to create a new namespace").
#                Inside such an already isolated container, use SANDBOX=danger-full-access and let
#                the container be the sandbox.
#   CONTEXT_WINDOW  optional model_context_window to tell Codex (it compacts when its count reaches it)
#   EFFORT       optional model_reasoning_effort
#   EXTRA_ARGS   extra arguments for `codex exec`
#   HTTP_PROXY, HTTPS_PROXY, NO_PROXY, ALL_PROXY, and http_proxy, https_proxy, no_proxy, all_proxy
#                passed to the sessions only when set (curl reads only the lowercase http_proxy),
#                e.g. a proxy at a closed port (http://127.0.0.1:9) with the endpoint's host in
#                NO_PROXY and no_proxy, to cut other egress
# Relative paths (arguments, CODEX_BIN, source dirs) are taken from where the runner is started.
#
# Every session runs under `env -i` and sees only PATH, its own HOME, CODEX_HOME and TMPDIR, the
# endpoint key TRTLLM_API_KEY, and TERM and the proxy variables above when they are set. So the
# user's ~/.codex (config, auth, history) is never read or written, nothing else from the calling
# shell leaks in, and Codex's temporary files stay in the session's directory. PATH is the
# caller's: Codex's npm launcher runs `node` from it, and the commands Codex runs find their tools
# on it. The endpoint key is a dummy unless TRTLLM_API_KEY is set.
# Flags were checked against codex-cli 0.155.1: re-check `codex exec --help` for other versions.
#
# Output: <out_dir>/sNNNN/{events.jsonl,stderr.log,last_message.txt,workspace/,tmp/} and
# <out_dir>/sessions.tsv (index, exit code, seconds, sandbox failures: the lines of events.jsonl
# with "bwrap: " in them, i.e. commands Codex's sandbox could not start). A session that exited 101
# or has sandbox failures ran none of its commands: the runner then warns and exits 3.
# Otherwise the exit status is 0 when every session exited 0, and 1 when one did not.
# Judge the run from the trace of the recorder BASE_URL points at (references/smoke.md section 1).
# <out_dir> must be new or empty, or the runner exits 2 before it starts a session.
# A session whose task source cannot be copied whole is not started: its row records
# exit code 1, and its stderr.log says why.
set -u
TASKS=${1:?tasks.tsv}
OUT=${2:?out_dir}
: "${BASE_URL:?set BASE_URL, e.g. http://127.0.0.1:8000/v1}"
MODEL=${MODEL:-served}
CONCURRENCY=${CONCURRENCY:-4}
TIMEOUT_S=${TIMEOUT_S:-3600}
CODEX_BIN=${CODEX_BIN:-codex}
SANDBOX=${SANDBOX:-workspace-write}
TRTLLM_API_KEY=${TRTLLM_API_KEY:-dummy-local-key}
# Codex works in the session's workspace: the paths it is given are made absolute here.
abspath() { case $1 in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
TASKS=$(abspath "$TASKS")
OUT=$(abspath "$OUT")
case $CODEX_BIN in */*) CODEX_BIN=$(abspath "$CODEX_BIN") ;; esac
# Of the caller's variables, these reach a session only when set.
PASS_ENV=()
for name in TERM HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY http_proxy https_proxy no_proxy all_proxy; do
  [ -n "${!name+set}" ] && PASS_ENV+=("$name=${!name}")
done
# A run starts from nothing: an earlier run's session directories would carry their workspaces
# and homes into this one.
if [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "$OUT is not empty: give each run a new output directory" >&2
  exit 2
fi
mkdir -p "$OUT"
: > "$OUT/sessions.tsv"

run_one() {
  local index=$1 src=$2 prompt=$3
  local dir
  dir=$(printf '%s/s%04d' "$OUT" "$index")
  mkdir -p "$dir/home" "$dir/workspace" "$dir/tmp"
  if [ "$src" != "-" ] && ! cp -a "$src/." "$dir/workspace/" 2> "$dir/stderr.log"; then
    # A session on a partial copy would work on another task than its own.
    echo "cannot copy the task source $src: the session was not started" >> "$dir/stderr.log"
    printf '%d\t%d\t%d\t%d\n' "$index" 1 0 0 >> "$OUT/sessions.tsv"
    return
  fi
  local extra=()
  [ -n "${CONTEXT_WINDOW:-}" ] && extra+=(-c "model_context_window=${CONTEXT_WINDOW}")
  [ -n "${EFFORT:-}" ] && extra+=(-c "model_reasoning_effort=\"${EFFORT}\"")
  local start=$SECONDS
  # shellcheck disable=SC2086
  timeout "$TIMEOUT_S" env -i PATH="$PATH" HOME="$dir/home" CODEX_HOME="$dir/home" TMPDIR="$dir/tmp" \
    TRTLLM_API_KEY="$TRTLLM_API_KEY" "${PASS_ENV[@]}" \
    "$CODEX_BIN" exec --json --skip-git-repo-check \
    -C "$dir/workspace" -s "$SANDBOX" -m "$MODEL" \
    -c 'model_provider="trtllm"' \
    -c "model_providers.trtllm={name=\"trtllm-serve\", base_url=\"${BASE_URL}\", wire_api=\"responses\", env_key=\"TRTLLM_API_KEY\"}" \
    "${extra[@]}" ${EXTRA_ARGS:-} -o "$dir/last_message.txt" "$prompt" \
    > "$dir/events.jsonl" 2> "$dir/stderr.log" < /dev/null
  local rc=$?
  # Counted, never printed: the lines are the agent's tool output.
  local sandbox_failures
  sandbox_failures=$(grep -c 'bwrap: ' "$dir/events.jsonl")
  printf '%d\t%d\t%d\t%d\n' "$index" "$rc" "$((SECONDS - start))" "$sandbox_failures" >> "$OUT/sessions.tsv"
}

index=0
while IFS= read -r line || [ -n "$line" ]; do
  case ${line#"${line%%[![:space:]]*}"} in "" | "#"*) continue ;; esac  # blank or comment
  IFS=$'\t' read -r src prompt <<< "$line"
  index=$((index + 1))
  run_one "$index" "$src" "$prompt" &
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 1; done
done < "$TASKS"
wait
echo "sessions: $index; exit codes: $(cut -f2 "$OUT/sessions.tsv" | sort | uniq -c | tr '\n' ' ')"
broken=$(awk -F'\t' '$2 == 101 || $4 > 0' "$OUT/sessions.tsv" | wc -l)
if [ "$broken" -gt 0 ]; then
  echo "WARNING: Codex's sandbox did not start in $broken session(s) (exit 101, or 'bwrap: ' in" \
    "events.jsonl), so their commands never ran. See SANDBOX above and references/smoke.md section 0." >&2
  exit 3
fi
failed=$(awk -F'\t' '$2 != 0' "$OUT/sessions.tsv" | wc -l)
if [ "$failed" -gt 0 ]; then
  echo "$failed session(s) exited non-zero: see sessions.tsv and their stderr.log" >&2
  exit 1
fi
