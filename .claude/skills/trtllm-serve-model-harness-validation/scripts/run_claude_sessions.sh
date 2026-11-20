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
# Run N real Claude Code sessions against a self-hosted Anthropic Messages endpoint, CONCURRENCY at
# a time, each in a clean environment.
#
# usage: run_claude_sessions.sh <tasks.tsv> <out_dir>
#   tasks.tsv: one task per line, "<source dir>\t<prompt>". The source dir is copied into a fresh
#              per-session workspace; use "-" for an empty one. Blank lines and lines starting
#              with "#" are skipped.
# env:
#   BASE_URL     endpoint base WITHOUT /v1, e.g. http://127.0.0.1:8000   (required)
#   MODEL        model name to send (default: served); also used for background tasks
#   CONCURRENCY  parallel sessions (default 4)
#   TIMEOUT_S    per-session wall clock (default 3600)
#   CLAUDE_BIN   claude executable (default: claude on PATH, resolved before the env is cleared)
#   ALLOWED      --allowedTools value (default "Read Glob Grep")
#   DISALLOWED   --disallowedTools value (default "Bash Write Edit NotebookEdit WebFetch WebSearch")
#                Allow write/exec tools only inside a disposable container or VM.
#   ATTRIBUTION  CLAUDE_CODE_ATTRIBUTION_HEADER value (default: unset, i.e. Claude Code's default)
#   EXTRA_ARGS   extra arguments for `claude -p`
#   HTTP_PROXY, HTTPS_PROXY, NO_PROXY, ALL_PROXY, and http_proxy, https_proxy, no_proxy, all_proxy
#                passed to the sessions only when set (curl reads only the lowercase http_proxy),
#                e.g. a proxy at a closed port (http://127.0.0.1:9) with the endpoint's host in
#                NO_PROXY and no_proxy, to cut other egress
# Relative paths (arguments, CLAUDE_BIN, source dirs) are taken from where the runner is started.
#
# Every session runs under `env -i`: nothing from the calling shell leaks in but the proxy
# variables above (a Claude Code parent session exports CLAUDE_CODE_* variables, including a
# messaging socket and token). Each session gets its own HOME, CLAUDE_CONFIG_DIR and TMPDIR, so
# the user's ~/.claude (credentials, settings, hooks, plugins) is never read, and Claude Code's
# temporary files stay in the session's directory. The endpoint credential
# is a dummy unless ANTHROPIC_AUTH_TOKEN is set. Gateway hint headers are on, so the trace carries
# request class, prompt id and compaction marks. Flags were checked against Claude Code 2.1.289:
# re-check `claude --help` for other versions.
#
# Output: <out_dir>/sNNNN/{events.jsonl,stderr.log,workspace/,tmp/} and <out_dir>/sessions.tsv
# (index, exit code, seconds). Judge the run from the trace of the recorder BASE_URL points at
# (record_requests.py) with scripts/validate/verdict.py --population smoke.
# <out_dir> must be new or empty, or the runner exits 2 before it starts a session.
# A session whose task source cannot be copied whole is not started: its row records
# exit code 1, and its stderr.log says why.
# Exit status: 0 when every session exited 0, 1 otherwise.
set -u
TASKS=${1:?tasks.tsv}
OUT=${2:?out_dir}
: "${BASE_URL:?set BASE_URL, e.g. http://127.0.0.1:8000 (no /v1)}"
MODEL=${MODEL:-served}
CONCURRENCY=${CONCURRENCY:-4}
TIMEOUT_S=${TIMEOUT_S:-3600}
CLAUDE_BIN=${CLAUDE_BIN:-$(readlink -f "$(command -v claude)")}
ALLOWED=${ALLOWED:-Read Glob Grep}
DISALLOWED=${DISALLOWED:-Bash Write Edit NotebookEdit WebFetch WebSearch}
TOKEN=${ANTHROPIC_AUTH_TOKEN:-dummy-local-token}
# A session runs in its own workspace: the paths it uses are made absolute here.
abspath() { case $1 in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
TASKS=$(abspath "$TASKS")
OUT=$(abspath "$OUT")
case $CLAUDE_BIN in */*) CLAUDE_BIN=$(abspath "$CLAUDE_BIN") ;; esac
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
  mkdir -p "$dir/home/.claude" "$dir/workspace" "$dir/tmp"
  if [ "$src" != "-" ] && ! cp -a "$src/." "$dir/workspace/" 2> "$dir/stderr.log"; then
    # A session on a partial copy would work on another task than its own.
    echo "cannot copy the task source $src: the session was not started" >> "$dir/stderr.log"
    printf '%d\t%d\t%d\n' "$index" 1 0 >> "$OUT/sessions.tsv"
    return
  fi
  local env_extra=()
  [ -n "${ATTRIBUTION:-}" ] && env_extra+=("CLAUDE_CODE_ATTRIBUTION_HEADER=${ATTRIBUTION}")
  local start=$SECONDS
  # shellcheck disable=SC2086
  (cd "$dir/workspace" && timeout "$TIMEOUT_S" env -i PATH=/usr/local/bin:/usr/bin:/bin \
    HOME="$dir/home" CLAUDE_CONFIG_DIR="$dir/home/.claude" TMPDIR="$dir/tmp" \
    ANTHROPIC_BASE_URL="$BASE_URL" ANTHROPIC_AUTH_TOKEN="$TOKEN" \
    ANTHROPIC_MODEL="$MODEL" ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
    CLAUDE_CODE_GATEWAY_HINT_HEADERS=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    DISABLE_AUTOUPDATER=1 "${env_extra[@]}" "${PROXY_ENV[@]}" \
    "$CLAUDE_BIN" -p "$prompt" --output-format stream-json --verbose \
    --allowedTools $ALLOWED --disallowedTools $DISALLOWED ${EXTRA_ARGS:-} \
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
