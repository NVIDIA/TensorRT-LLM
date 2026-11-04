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
#              per-session workspace; use "-" for an empty one.
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
#
# Every session runs under `env -i`: nothing from the calling shell leaks in (a Claude Code parent
# session exports CLAUDE_CODE_* variables, including a messaging socket and token). Each session
# gets its own HOME and CLAUDE_CONFIG_DIR, so the user's ~/.claude (credentials, settings, hooks,
# plugins) is never read. The endpoint credential is a dummy unless ANTHROPIC_AUTH_TOKEN is set.
# Gateway hint headers are on, so the trace carries request class, prompt id and compaction marks.
# Flags were checked against Claude Code 2.1.289: re-check `claude --help` for other versions.
#
# Output: <out_dir>/sNNNN/{events.jsonl,stderr.log,workspace/} and <out_dir>/sessions.tsv
# (index, exit code, seconds). Judge the run from the SERVER side trace (request trace or
# record_requests.py) with scripts/validate/verdict.py --population smoke.
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
mkdir -p "$OUT"
: > "$OUT/sessions.tsv"

run_one() {
  local index=$1 src=$2 prompt=$3
  local dir
  dir=$(printf '%s/s%04d' "$OUT" "$index")
  mkdir -p "$dir/home/.claude" "$dir/workspace"
  if [ "$src" != "-" ]; then cp -a "$src/." "$dir/workspace/"; fi
  local env_extra=()
  [ -n "${ATTRIBUTION:-}" ] && env_extra+=("CLAUDE_CODE_ATTRIBUTION_HEADER=${ATTRIBUTION}")
  local start=$SECONDS
  # shellcheck disable=SC2086
  (cd "$dir/workspace" && timeout "$TIMEOUT_S" env -i PATH=/usr/local/bin:/usr/bin:/bin \
    HOME="$dir/home" CLAUDE_CONFIG_DIR="$dir/home/.claude" \
    ANTHROPIC_BASE_URL="$BASE_URL" ANTHROPIC_AUTH_TOKEN="$TOKEN" \
    ANTHROPIC_MODEL="$MODEL" ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
    CLAUDE_CODE_GATEWAY_HINT_HEADERS=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    DISABLE_AUTOUPDATER=1 "${env_extra[@]}" \
    "$CLAUDE_BIN" -p "$prompt" --output-format stream-json --verbose \
    --allowedTools $ALLOWED --disallowedTools $DISALLOWED ${EXTRA_ARGS:-} \
    > "$dir/events.jsonl" 2> "$dir/stderr.log" < /dev/null)
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
