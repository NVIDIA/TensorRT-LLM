#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Compose a trial build: a base commit plus open PRs and capture changes, one commit per item, and a
# provenance record of exactly what went in (references/smoke.md section 3d).
#
# usage: compose_trial_build.sh [-k] [-r REMOTE] [-m MAIN_REF] [-p FILE] <worktree> <branch> <base> <item>...
#   <item>      <label>=<source>. <source> is pr:<number> (the PR's head, fetched from REMOTE) or any
#               commit-ish: a branch (<remote>/<branch>) or a sha.
#   -k          keep <branch> as it is and add the items on top (after fixing a conflict by hand)
#   -r REMOTE   where pr:<number> heads are fetched from (default: origin)
#   -m MAIN_REF the upstream main each item's diff is taken against (default: origin/main)
#   -p FILE     the provenance file (default: <worktree>.<branch>.provenance.txt); lines are appended
#
# Each item becomes ONE commit: its diff against its merge-base with MAIN_REF, applied with a
# three-way merge, so a PR's own merge commits and its stale base do not come along. A conflict stops
# the script (exit 2) with the index left as it is: resolve it, commit with the printed message, then
# rerun with -k and the remaining items. Commits run the repository's hooks; a hook that rewrites a
# file is recorded as hooks_modified, because the served code then differs from the PR head.
set -u -o pipefail

usage() { sed -n '/^# usage:/,/^# Each item/p' "$0" | sed '$d'; exit 64; }

keep=0
remote=origin
main_ref=origin/main
prov=""
while getopts "kr:m:p:" opt; do
  case "$opt" in
    k) keep=1 ;;
    r) remote=$OPTARG ;;
    m) main_ref=$OPTARG ;;
    p) prov=$OPTARG ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 3 ] || usage
# Absolute paths: the worktree is added from the repository root and then entered, and the
# provenance file is written from inside it.
absolute() { case $1 in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "$PWD" "$1" ;; esac; }
wt=$(absolute "$1"); branch=$2; base=$3; shift 3
[ -n "$prov" ] || prov="$wt.${branch//\//_}.provenance.txt"  # a branch's / is not a directory
prov=$(absolute "$prov")

repo=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "run this inside a TensorRT-LLM checkout"; exit 64; }
base_sha=$(git -C "$repo" rev-parse --verify "$base^{commit}") || { echo "unknown base $base"; exit 64; }
if [ ! -d "$wt" ]; then
  git -C "$repo" worktree add -q --detach "$wt" "$base_sha" || exit 1
fi
cd "$wt" || exit 1
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "$wt has uncommitted changes; commit or discard them first"
  exit 1
fi
if [ "$keep" -eq 0 ]; then
  git checkout -q --detach "$base_sha" && git checkout -q -B "$branch" || exit 1
  {
    echo "# trial build $branch, composed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "base $base $base_sha"
    echo "main_ref $main_ref $(git rev-parse --verify "$main_ref^{commit}")"
  } >> "$prov"
elif [ "$(git rev-parse --abbrev-ref HEAD)" != "$branch" ]; then
  echo "-k: $wt is not on $branch"
  exit 1
fi

for item in "$@"; do
  label=${item%%=*}
  source=${item#*=}
  if [ -z "$label" ] || [ "$label" = "$item" ]; then
    echo "bad item '$item': expected <label>=<source>"
    exit 64
  fi
  case "$source" in
    pr:*)
      git fetch -q "$remote" "pull/${source#pr:}/head" || { echo "cannot fetch $source from $remote"; exit 1; }
      sha=$(git rev-parse FETCH_HEAD) ;;
    *)
      sha=$(git rev-parse --verify "$source^{commit}") || { echo "unknown source $source"; exit 1; } ;;
  esac
  mb=$(git merge-base "$main_ref" "$sha") || { echo "$source shares no history with $main_ref"; exit 1; }
  pid=$(git diff "$mb" "$sha" | git patch-id --stable | cut -d' ' -f1)
  if [ -z "$pid" ]; then
    echo "item $label $source head=$sha empty" >> "$prov"
    echo "skipped $label: no change against $main_ref"
    continue
  fi
  msg="[None][test] trial build: apply $label (${sha:0:10})"
  if ! git diff --binary "$mb" "$sha" | git apply -3 --index 2> "$prov.apply.err"; then
    echo "CONFLICT applying $label ($source ${sha:0:10}); unmerged paths:"
    git diff --name-only --diff-filter=U
    echo "resolve, then: git commit -s -m \"$msg\"; rerun with -k and the remaining items"
    echo "item $label $source head=$sha merge_base=$mb patch_id=$pid CONFLICT" >> "$prov"
    exit 2
  fi
  hooks=""
  if ! git commit -q -s -m "$msg" > "$prov.commit.log" 2>&1; then
    git add -u
    git commit -q -s -m "$msg" >> "$prov.commit.log" 2>&1 || { echo "commit of $label failed: $prov.commit.log"; exit 1; }
    hooks=" hooks_modified"
  fi
  echo "item $label $source head=$sha merge_base=$mb patch_id=$pid commit=$(git rev-parse --short=10 HEAD)$hooks" >> "$prov"
  echo "applied $label ${sha:0:10}$hooks"
done
echo "result $branch head=$(git rev-parse HEAD) tree=$(git rev-parse 'HEAD^{tree}')" >> "$prov"
echo "provenance: $prov"
