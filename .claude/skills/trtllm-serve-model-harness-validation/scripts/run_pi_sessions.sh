#!/bin/bash
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
# Run real pi sessions against a self-hosted endpoint over one of pi's three wire APIs, each in a
# clean environment. Run it once per API with the same tasks for a protocol differential
# (scripts/validate/protocol_diff.py).
#
# usage: run_pi_sessions.sh <tasks.tsv> <out_dir>
#   tasks.tsv: one task per line, "<source dir>\t<prompt>". The source dir is copied into a fresh
#              per-session workspace; use "-" for an empty one. Blank lines and lines starting
#              with "#" are skipped.
# env:
#   BASE_URL       endpoint base WITH /v1, e.g. http://127.0.0.1:8000/v1   (required). pi appends
#                  /chat/completions or /responses to it, but /v1/messages for anthropic-messages,
#                  so for that API a trailing /v1 is removed here.
#   API            openai-completions (default), openai-responses or anthropic-messages
#   MODEL          model id sent to the server (required)
#   CONTEXT_WINDOW served max_seq_len (default 32768); pi's own default is 128000
#   MAX_TOKENS     output cap (default 4096)
#   REASONING      true (default) or false: whether pi treats the model as a thinking model
#   THINKING       pi --thinking level (default: pi's own)
#   COMPAT         JSON object for the model's `compat` (default {})
#   THINKING_LEVEL_MAP
#                  JSON object for the model's `thinkingLevelMap` (default: none, so pi sends the
#                  level name as is and runs xhigh and max as high). With COMPAT it decides which
#                  request field carries the level: per model, contracts/pi.md section 12
#   TOOLS          pi --tools allowlist (default read,ls). Allow bash/edit/write only inside a
#                  disposable container or VM.
#   WORKSPACE      a fixed root for the session workspaces, wiped and refilled per session. pi
#                  puts its working directory in the system prompt, so a protocol differential
#                  runs every API in the same path (default: <out_dir>/sNNNN/workspace)
#   CONCURRENCY    parallel sessions (default 1: a differential compares runs turn by turn)
#   TIMEOUT_S      per-session wall clock (default 1800)
#   PI_BIN         pi entry point (default: pi on PATH); NODE_BIN the node to run it with
#   EXTRA_ARGS     extra arguments for `pi -p`
#   HTTP_PROXY, HTTPS_PROXY, NO_PROXY, ALL_PROXY, and http_proxy, https_proxy, no_proxy, all_proxy
#                  passed to the sessions only when set (curl reads only the lowercase
#                  http_proxy), e.g. a proxy at a closed port (http://127.0.0.1:9) with the
#                  endpoint's host in NO_PROXY and no_proxy, to cut other egress
# Relative paths (arguments, WORKSPACE, PI_BIN, NODE_BIN, source dirs) are taken from where the
# runner is started.
#
# Every session runs under `env -i` with its own HOME, PI_CODING_AGENT_DIR and TMPDIR, so ~/.pi is
# never read, and of the caller's variables only the proxy variables above reach it. Startup network
# activity, version checks and telemetry are off. Extensions, MCP, skills, prompt templates and
# context files are disabled, so the request carries only pi's own prompt.
# Flags were checked against @earendil-works/pi-coding-agent 1.4.0: re-check `pi --help`.
#
# Output: <out_dir>/sNNNN/{events.jsonl,stderr.log,workspace/,agent/models.json,tmp/} and
# <out_dir>/sessions.tsv (index, exit code, seconds). Judge the run from the trace of the recorder
# BASE_URL points at (record_requests.py) with scripts/validate/verdict.py --population smoke.
# <out_dir> must be new or empty, or the runner exits 2 before it starts a session.
# A session whose task source cannot be copied whole is not started: its row records
# exit code 1, and its stderr.log says why.
# Exit status: 0 when every session exited 0, 1 otherwise. pi -p exits 0 even when a run ends
# on an error, so read its events too.
set -u
TASKS=${1:?tasks.tsv}
OUT=${2:?out_dir}
: "${BASE_URL:?set BASE_URL, e.g. http://127.0.0.1:8000/v1}"
: "${MODEL:?set MODEL to the served model id}"
API=${API:-openai-completions}
CONTEXT_WINDOW=${CONTEXT_WINDOW:-32768}
MAX_TOKENS=${MAX_TOKENS:-4096}
REASONING=${REASONING:-true}
COMPAT=${COMPAT:-"{}"}
LEVEL_MAP=""
[ -n "${THINKING_LEVEL_MAP:-}" ] && LEVEL_MAP=", \"thinkingLevelMap\": $THINKING_LEVEL_MAP"
TOOLS=${TOOLS:-read,ls}
CONCURRENCY=${CONCURRENCY:-1}
TIMEOUT_S=${TIMEOUT_S:-1800}
PI_BIN=${PI_BIN:-$(readlink -f "$(command -v pi)")}
NODE_BIN=${NODE_BIN:-$(readlink -f "$(command -v node)")}
[ "$API" = anthropic-messages ] && BASE_URL=${BASE_URL%/v1}
# A session runs in its own workspace: the paths it uses are made absolute here.
abspath() { case $1 in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
TASKS=$(abspath "$TASKS")
OUT=$(abspath "$OUT")
[ -n "${WORKSPACE:-}" ] && WORKSPACE=$(abspath "$WORKSPACE")
case $PI_BIN in */*) PI_BIN=$(abspath "$PI_BIN") ;; esac
case $NODE_BIN in */*) NODE_BIN=$(abspath "$NODE_BIN") ;; esac
# Of the caller's variables, these reach a session only when set.
PROXY_ENV=()
for name in HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY http_proxy https_proxy no_proxy all_proxy; do
  [ -n "${!name+set}" ] && PROXY_ENV+=("$name=${!name}")
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
  local ws="$dir/workspace"
  if [ -n "${WORKSPACE:-}" ]; then ws=$(printf '%s/s%04d' "$WORKSPACE" "$index"); rm -rf "$ws"; fi
  mkdir -p "$dir/home" "$dir/agent" "$dir/sessions" "$dir/tmp" "$ws"
  if [ "$src" != "-" ] && ! cp -a "$src/." "$ws/" 2> "$dir/stderr.log"; then
    # A session on a partial copy would work on another task than its own.
    echo "cannot copy the task source $src: the session was not started" >> "$dir/stderr.log"
    printf '%d\t%d\t%d\n' "$index" 1 0 >> "$OUT/sessions.tsv"
    return
  fi
  cat > "$dir/agent/models.json" <<JSON
{"providers": {"trtllm": {"baseUrl": "$BASE_URL", "api": "$API", "apiKey": "dummy-local-key",
  "models": [{"id": "$MODEL", "name": "trtllm-$MODEL", "reasoning": $REASONING, "input": ["text"],
              "contextWindow": $CONTEXT_WINDOW, "maxTokens": $MAX_TOKENS,
              "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
              "compat": $COMPAT$LEVEL_MAP}]}}}
JSON
  local thinking=()
  [ -n "${THINKING:-}" ] && thinking=(--thinking "$THINKING")
  local start=$SECONDS
  # shellcheck disable=SC2086
  (cd "$ws" && timeout "$TIMEOUT_S" env -i PATH=/usr/local/bin:/usr/bin:/bin \
    HOME="$dir/home" PI_CODING_AGENT_DIR="$dir/agent" TMPDIR="$dir/tmp" \
    PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0 \
    "${PROXY_ENV[@]}" \
    "$NODE_BIN" "$PI_BIN" -p --mode json --provider trtllm --model "$MODEL" "${thinking[@]}" \
    --tools "$TOOLS" --no-extensions --no-mcp --no-skills --no-prompt-templates --no-context-files \
    --no-themes --offline --session-dir "$dir/sessions" ${EXTRA_ARGS:-} -- "$prompt" \
    > "$dir/events.jsonl" 2> "$dir/stderr.log" < /dev/null)
  local rc=$?
  printf '%d\t%d\t%d\n' "$index" "$rc" "$((SECONDS - start))" >> "$OUT/sessions.tsv"
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
failed=$(awk -F'\t' '$2 != 0' "$OUT/sessions.tsv" | wc -l)
if [ "$failed" -gt 0 ]; then
  echo "$failed session(s) exited non-zero: see sessions.tsv and their stderr.log" >&2
  exit 1
fi
