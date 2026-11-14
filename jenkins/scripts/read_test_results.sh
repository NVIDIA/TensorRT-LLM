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

# Reads the test-result archives of one CI run into a directory with the read helper,
# fetch_build_artifact.sh. The steps that need every archive of a run call it: it lists the run's
# test-results/ folder, reads every final archive, and reads a stage attempt's checkpoint only
# when the attempt's final archive could not be read, so that a checkpoint left beside its final
# archive never adds to or replaces the final archive's results. While aws is the primary, a
# failure of urm is logged and changes nothing else. Run it with --help for the interface and the
# exit codes.
#
# The script handles no credential: the helper takes it from the netrc file that the environment
# names.

set -uo pipefail

# Exit codes of fetch_build_artifact.sh that this script acts on, by the names the helper gives
# them. tests/unittest/scripts/test_fetch_build_artifact.py checks that the values match, and that
# the functions below check the helper's exit status by these names only.
FETCH_ABSENT=2
FETCH_REJECTED=3
FETCH_TRANSFER_FAILED=4
FETCH_SECONDARY_FAILED=5
# This script's exit codes other than success.
READ_INCOMPLETE=1
READ_USAGE=64
# The reading stops after this many reads in a row that the primary rejected or failed, since the
# rest would only wait out the helper's retries.
READ_FAILURES_BEFORE_STOP=3
readonly FETCH_ABSENT FETCH_REJECTED FETCH_TRANSFER_FAILED FETCH_SECONDARY_FAILED
readonly READ_INCOMPLETE READ_USAGE READ_FAILURES_BEFORE_STOP

# A checkpoint is named after its stage attempt's final archive, results-<x>.tar.gz, as
# results-<x>-progress.tar.gz.
readonly ARCHIVE_SUFFIX=".tar.gz"
readonly CHECKPOINT_SUFFIX="-progress.tar.gz"

# The text takes this script's exit codes and the stop's count from their names above.
usage() {
    cat <<EOF
Usage: read_test_results.sh <helper> <test-results dir> <dest dir> [<name ERE> [<exclude ERE>]]

Reads the test-result archives of one CI run into <dest dir> with the read helper. A stage
attempt's checkpoint is results-<x>-progress.tar.gz, and its final archive is
results-<x>.tar.gz.

  - Lists <test-results dir> on the locations that the helper asks.
  - Reads every listed name that is not a checkpoint.
  - For each listed checkpoint whose final archive was not read, reads the final archive by
    name, which may have landed after the listing, and the checkpoint only when that fails.
    A checkpoint that is gone by then was deleted after its final archive's upload, so the
    final archive is read once more.
  - Reads no checkpoint whose final archive was read.

While aws is the primary, the helper gets --secondary-optional, and a listing or a read that
fails on urm is logged and counted, and leaves the exit code alone. While urm is the primary,
the helper asks urm alone.

Arguments:
  <helper>            Path of fetch_build_artifact.sh, which is run with bash.
  <test-results dir>  The run's test-results/ folder, as a legacy path
                      (sw-tensorrt-generic/llm-artifacts/...) or a layout path
                      (llm-artifacts/...), which may end in "/".
  <dest dir>          An existing directory. Each file is written there under its own name.
  <name ERE>          Keeps only the listed names that this extended regular expression
                      matches. Without it, or when it is empty, every name is kept.
  <exclude ERE>       Drops the listed names that this extended regular expression matches.
                      Without it, or when it is empty, no name is dropped.

Environment, as the helper reads it:
  LLM_ARTIFACTS_PRIMARY      aws (the default) or urm.
  ARTIFACTORY_NETRC, ARTIFACTORY_CREDENTIAL_ID
                             The netrc file for aws, and the name of the CI credential behind
                             it.

Output:
  The helper's messages and one line on stderr for each name that was skipped, replaced,
  absent, or not read, and then, as the last line on stderr:
    [test-results] listed <n> (aws <a>, urm <u>), read <r>, checkpoints skipped <s>,
    replaced <g>, absent <x>, urm unreachable <m>, failed <f>
  on one line, with "aws not asked" in place of "aws <a>" while urm is the primary.
    listed           Names kept by the patterns, and how many of them each location listed.
    read             Files read into <dest dir>.
    skipped          Checkpoints not read because their final archive was read.
    replaced         Checkpoints whose final archive, read after the listing, took their place.
    absent           Listed final archives that neither location has, without a listed
                     checkpoint to stand in for them.
    urm unreachable  Listings and reads that failed on urm while aws is the primary.
    failed           Reads that failed in any other way, which the helper's message tells, and
                     checkpoints that are gone while neither location has their final archive.

Exit codes:
  0   Complete: the listing succeeded on the primary, or neither location has the folder, and
      absent and failed are both 0.
  $READ_INCOMPLETE   Incomplete: the listing failed on the primary, or absent or failed is not 0.
      The reading stops early, with this code, after $READ_FAILURES_BEFORE_STOP reads in a row
      that the primary rejected or failed.
  $READ_USAGE  Usage error: wrong arguments, a missing <helper> or <dest dir>, an invalid pattern,
      or a bad LLM_ARTIFACTS_PRIMARY.
EOF
}

log() {
    printf '[test-results] %s\n' "$*" >&2
}

usage_error() {
    log "$@"
    usage >&2
    exit "$READ_USAGE"
}

# Sets HELPER, RESULTS_DIR, DEST_DIR, NAME_ERE, and EXCLUDE_ERE from the arguments.
parse_arguments() {
    local wrong="wrong number of arguments"
    (( $# > 0 )) || usage_error "$wrong"
    HELPER="$1"
    shift
    (( $# > 0 )) || usage_error "$wrong"
    RESULTS_DIR="${1%/}"
    shift
    (( $# > 0 )) || usage_error "$wrong"
    DEST_DIR="$1"
    shift
    if (( $# > 0 )); then
        NAME_ERE="$1"
        shift
    fi
    if (( $# > 0 )); then
        EXCLUDE_ERE="$1"
        shift
    fi
    (( $# == 0 )) || usage_error "$wrong"
}

# valid_ere <pattern>: fails for a pattern that is not an extended regular expression. A match
# returns 0 and a mismatch 1; only a pattern that cannot be compiled returns more.
valid_ere() {
    [[ "" =~ $1 ]]
    (( $? <= 1 ))
}

# keeps <name>: succeeds for a name that <name ERE> keeps and <exclude ERE> does not drop.
keeps() {
    if [[ -n "$NAME_ERE" && ! "$1" =~ $NAME_ERE ]]; then
        return 1
    fi
    [[ -z "$EXCLUDE_ERE" || ! "$1" =~ $EXCLUDE_ERE ]]
}

# run_helper <argument>...: runs the read helper and returns its exit status. While aws is the
# primary, the helper reports a failure of urm with a status of its own.
run_helper() {
    if [[ "$PRIMARY" == aws ]]; then
        bash "$HELPER" --secondary-optional "$@"
    else
        bash "$HELPER" "$@"
    fi
}

# Lists the folder and sorts the names that the patterns keep into FINALS and CHECKPOINTS, each
# name once and in the listing's order. The helper prints "<location> <name>" for each file, the
# primary's files first. A listing that failed on the primary makes the read incomplete, and the
# names that the helper still printed are read.
list_results() {
    local listing status line location name
    listing="$(run_helper --list "$RESULTS_DIR")"
    status=$?
    if (( status == FETCH_SECONDARY_FAILED )); then
        URM_UNREACHABLE=$(( URM_UNREACHABLE + 1 ))
        log "urm could not be listed (fetch_build_artifact.sh exited $status);" \
            "going on with the names that aws listed"
    elif (( status != 0 && status != FETCH_ABSENT )); then
        INCOMPLETE=true
        log "the listing failed (fetch_build_artifact.sh exited $status);" \
            "some names may be missing"
    fi
    while IFS= read -r line; do
        location="${line%% *}"
        name="${line#* }"
        if [[ "$line" != *" "* || -z "$name" ]] || ! keeps "$name"; then
            continue
        fi
        case "$location" in
            aws) ON_AWS=$(( ON_AWS + 1 )) ;;
            urm) ON_URM=$(( ON_URM + 1 )) ;;
            *) continue ;;
        esac
        if [[ -n "${LISTED[$name]:-}" ]]; then
            continue
        fi
        # The first location that listed the name: the primary, when it lists the name at all.
        LISTED[$name]="$location"
        if [[ "$name" == *"$CHECKPOINT_SUFFIX" ]]; then
            CHECKPOINTS+=("$name")
        else
            FINALS+=("$name")
        fi
    done <<<"$listing"
}

# read_archive <name>: reads one file of the folder into DEST_DIR, counts the result, and returns
# the helper's exit status. After a read that succeeded, SERVED_BY is the location that the helper
# names as the one that served the file. A read that the primary rejected or failed counts toward
# the stop, and any other result starts that count again; STOPPED is set once
# READ_FAILURES_BEFORE_STOP such reads came in a row. A name that both locations lack is left to
# the caller, which knows what its absence means.
read_archive() {
    local name="$1" status messages
    SERVED_BY=""
    # A download prints nothing on stdout, and the helper's messages go on to stderr.
    messages="$(run_helper "$RESULTS_DIR/$name" "$DEST_DIR/$name" 2>&1)"
    status=$?
    if [[ -n "$messages" ]]; then
        printf '%s\n' "$messages" >&2
    fi
    case "$status" in
        0)
            if [[ "$messages" =~ served-by=(aws|urm)[[:space:]] ]]; then
                SERVED_BY="${BASH_REMATCH[1]}"
            fi
            READ[$name]=1
            READ_COUNT=$(( READ_COUNT + 1 ))
            FAILURES_IN_A_ROW=0
            ;;
        "$FETCH_ABSENT") FAILURES_IN_A_ROW=0 ;;
        "$FETCH_SECONDARY_FAILED")
            URM_UNREACHABLE=$(( URM_UNREACHABLE + 1 ))
            FAILURES_IN_A_ROW=0
            log "urm unreachable: aws does not have $name, and urm could not answer" \
                "(fetch_build_artifact.sh exited $status); not read"
            ;;
        *)
            FAILED=$(( FAILED + 1 ))
            if (( status == FETCH_REJECTED || status == FETCH_TRANSFER_FAILED )); then
                FAILURES_IN_A_ROW=$(( FAILURES_IN_A_ROW + 1 ))
            else
                FAILURES_IN_A_ROW=0
            fi
            log "failed to read $name (fetch_build_artifact.sh exited $status)"
            ;;
    esac
    if (( FAILURES_IN_A_ROW >= READ_FAILURES_BEFORE_STOP )); then
        STOPPED=true
        log "$FAILURES_IN_A_ROW reads in a row failed; not reading the remaining names"
    fi
    return "$status"
}

# Reads every listed name that is not a checkpoint. A final archive that both locations lack is
# absent unless its checkpoint is listed, which read_checkpoints then reads in its place.
read_finals() {
    local name
    (( ${#FINALS[@]} > 0 )) || return 0
    for name in "${FINALS[@]}"; do
        if [[ "$STOPPED" == true ]]; then
            return 0
        fi
        read_archive "$name"
        if (( $? == FETCH_ABSENT )) \
            && [[ -z "${LISTED[${name%"$ARCHIVE_SUFFIX"}$CHECKPOINT_SUFFIX]:-}" ]]; then
            ABSENT=$(( ABSENT + 1 ))
            log "absent: neither location has $name, which was listed," \
                "and no checkpoint stands in for it"
        fi
    done
}

# replaced <checkpoint> <final archive>: counts a checkpoint whose final archive was read in its
# place.
replaced() {
    REPLACED=$(( REPLACED + 1 ))
    log "replaced checkpoint $1: read its final archive $2 instead"
}

# Skips each listed checkpoint whose final archive was read. For any other, reads the final
# archive, which may have landed after the listing, and the checkpoint only when that fails. A
# stage deletes its checkpoint only after its final archive's upload, so a checkpoint that is gone
# sends the read to its final archive once more, and the attempt's results are lost only when
# that fails too. A checkpoint is gone when both locations lack it, and also when aws listed it
# and lacks it now, whether urm could not answer or served a copy of its own: a stage deletes its
# checkpoint on aws first, so urm's copy is one that the stage has not deleted yet or could not
# delete. That copy is kept only when the final archive cannot be read.
read_checkpoints() {
    local checkpoint final status gone_from_aws
    (( ${#CHECKPOINTS[@]} > 0 )) || return 0
    for checkpoint in "${CHECKPOINTS[@]}"; do
        final="${checkpoint%"$CHECKPOINT_SUFFIX"}$ARCHIVE_SUFFIX"
        if [[ -n "${READ[$final]:-}" ]]; then
            SKIPPED=$(( SKIPPED + 1 ))
            log "skipped checkpoint $checkpoint: its final archive $final was read"
            continue
        fi
        if [[ "$STOPPED" == true ]]; then
            continue
        fi
        if read_archive "$final"; then
            replaced "$checkpoint" "$final"
            continue
        fi
        if [[ "$STOPPED" == true ]]; then
            continue
        fi
        read_archive "$checkpoint"
        status=$?
        gone_from_aws=false
        if [[ "${LISTED[$checkpoint]}" == aws ]]; then
            if (( status == FETCH_SECONDARY_FAILED )) || { (( status == 0 )) && [[ "$SERVED_BY" == urm ]]; }; then
                gone_from_aws=true
            fi
        fi
        if (( status != FETCH_ABSENT )) && [[ "$gone_from_aws" == false ]]; then
            continue
        fi
        read_archive "$final"
        status=$?
        if (( status == 0 )); then
            if [[ -n "${READ[$checkpoint]:-}" ]]; then
                rm -f -- "$DEST_DIR/$checkpoint"
                unset 'READ[$checkpoint]'
                READ_COUNT=$(( READ_COUNT - 1 ))
            fi
            replaced "$checkpoint" "$final"
        elif [[ -n "${READ[$checkpoint]:-}" ]]; then
            log "kept urm's copy of checkpoint $checkpoint: aws no longer has it, and its final" \
                "archive $final could not be read"
        elif (( status == FETCH_ABSENT )); then
            FAILED=$(( FAILED + 1 ))
            log "failed: checkpoint $checkpoint is gone, and neither location has its final" \
                "archive $final"
        elif (( status == FETCH_SECONDARY_FAILED )) && [[ "${LISTED[$checkpoint]}" == aws ]]; then
            # aws listed the checkpoint, so the stage wrote there, and aws has neither file now.
            FAILED=$(( FAILED + 1 ))
            log "failed: checkpoint $checkpoint is gone from aws, which does not have its final" \
                "archive $final either, and urm could not answer"
        fi
    done
}

print_summary() {
    local on_aws="aws $ON_AWS"
    if [[ "$PRIMARY" == urm ]]; then
        on_aws="aws not asked"
    fi
    log "listed ${#LISTED[@]} ($on_aws, urm $ON_URM), read $READ_COUNT," \
        "checkpoints skipped $SKIPPED, replaced $REPLACED, absent $ABSENT," \
        "urm unreachable $URM_UNREACHABLE, failed $FAILED"
}

case "${1:-}" in
    -h | --help)
        usage
        exit 0
        ;;
    *) ;;
esac

HELPER=""
RESULTS_DIR=""
DEST_DIR=""
NAME_ERE=""
EXCLUDE_ERE=""
parse_arguments "$@"

if [[ ! -f "$HELPER" || ! -r "$HELPER" ]]; then
    usage_error "<helper> is not a readable file: '$HELPER'"
fi
if [[ -z "$RESULTS_DIR" ]]; then
    usage_error "<test-results dir> is empty"
fi
if [[ ! -d "$DEST_DIR" ]]; then
    usage_error "<dest dir> is not a directory: '$DEST_DIR'"
fi
if [[ -n "$NAME_ERE" ]] && ! valid_ere "$NAME_ERE"; then
    usage_error "<name ERE> is not an extended regular expression: '$NAME_ERE'"
fi
if [[ -n "$EXCLUDE_ERE" ]] && ! valid_ere "$EXCLUDE_ERE"; then
    usage_error "<exclude ERE> is not an extended regular expression: '$EXCLUDE_ERE'"
fi

PRIMARY="${LLM_ARTIFACTS_PRIMARY:-aws}"
if [[ "$PRIMARY" != aws && "$PRIMARY" != urm ]]; then
    usage_error "LLM_ARTIFACTS_PRIMARY must be 'aws' or 'urm', not '$PRIMARY'"
fi

# The names to read, in the order of the listing, and the sets of names listed and read.
FINALS=()
CHECKPOINTS=()
declare -A LISTED=()
declare -A READ=()
ON_AWS=0
ON_URM=0
READ_COUNT=0
# The location that served the last file read, as the helper names it.
SERVED_BY=""
SKIPPED=0
REPLACED=0
ABSENT=0
URM_UNREACHABLE=0
FAILED=0
FAILURES_IN_A_ROW=0
INCOMPLETE=false
STOPPED=false

list_results
read_finals
read_checkpoints
print_summary

if [[ "$INCOMPLETE" == true || "$STOPPED" == true ]] || (( ABSENT > 0 || FAILED > 0 )); then
    exit "$READ_INCOMPLETE"
fi
exit 0
