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

# Keeps a checkpoint of a stage's results on each upload target, from the background of the step
# that runs the tests. Every PROGRESS_INTERVAL seconds it prepares a checkpoint with
# progress_snapshot.sh, after copying a Slurm batch job's latest results back with
# progress_collect_slurm.sh, and uploads it to each target that lacks its content. Once
# PROGRESS_DONE_FILE exists, it takes the final snapshot, uploads it, and exits. A step that cannot
# create that file sends the watcher USR1 instead, to the same effect, so that the step never waits
# for a watcher that cannot learn that the tests have ended. It always exits 0: every failure is
# logged, and none fails the step.
#
# The credentials come from a netrc that the pipeline writes for this watcher alone. The watcher
# reads it into memory and removes it before anything else, and the step starts the tests only once
# it is gone. Each request then gets the credentials through a pipe. While the tests run, no secret
# is therefore in a file on disk, in an environment, on a command line, or in a trace, so the tests
# cannot inherit, print, or archive one. A test that sets out to take the credentials still can,
# except on a Slurm batch stage: elsewhere the tests run beside the watcher, as its user. Never turn
# on set -x here.
#
# Required env vars:
#   WORKSPACE, STAGE_NAME, PROGRESS_TAR  as for progress_snapshot.sh
#   CHECKPOINT_NETRC_FILE  the netrc, alone in its directory, with an entry for each target's host;
#                          without it, the watcher prepares its snapshots but uploads none
#   CHECKPOINT_TARGETS     space-separated <location>=<upload URL> pairs, in upload order
#   PROGRESS_DONE_FILE     once this file exists, or once USR1 arrives, the watcher takes the final
#                          snapshot and exits
#   PROGRESS_INTERVAL      seconds between checkpoints
#   LABEL_PREFIX           log label of a checkpoint, e.g. "checkpoint" or "rerun2 checkpoint"
#   FINAL_LABEL            log label of the final snapshot, e.g. "run1 final snapshot"
#
# Optional:
#   POST_TAG, TIMEOUT_XML_SCRIPT     passed on to progress_snapshot.sh
#   XML_PATH            a checkpoint is taken only when this file's mtime has advanced, or while a
#                       target still lacks the content of the last one
#   SNAPSHOT_REQUIRES   no checkpoint is taken while the tests run unless this file exists; on a
#                       Slurm batch stage it is checked after the job's results are copied back
#   FINAL_REQUIRES      no final snapshot is taken unless this file exists
#   CHECKPOINT_REJECTED_<location>
#                       the message for a 401 or 403 from that location, with HTTPSTATUS in place
#                       of the status
#   PROGRESS_STOP_FILE  once this file exists, no new checkpoint starts, and while one is under way
#                       ${PROGRESS_STOP_FILE}.busy exists. A Slurm batch stage's step creates it
#                       before it copies the job's last results back, and waits for .busy to go, so
#                       that the copy and a checkpoint never write the results at the same time.
#   SLURM_SSH_STAT_CMD, the other SLURM_* commands, and PROGRESS_MTIME_FILE
#                       a Slurm batch stage: each checkpoint first runs progress_collect_slurm.sh
#                       with them

set +e
set +x

log() {
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: $*"
}

# --- The end of the tests: PROGRESS_DONE_FILE, or USR1 from a step that cannot create it ---
# The trap is set before anything else, since USR1 ends a process that does not trap it. bash runs
# a trap only once the command under way returns, so the signal cuts no checkpoint short.
tests_ended=0
trap 'tests_ended=1' USR1

tests_done() {
    [ "$tests_ended" = 1 ] || [ -e "$PROGRESS_DONE_FILE" ]
}

# --- Credentials: read the netrc into a variable that is not exported, then remove it ---
netrc=""
if [ -n "$CHECKPOINT_NETRC_FILE" ]; then
    [ -r "$CHECKPOINT_NETRC_FILE" ] && netrc=$(< "$CHECKPOINT_NETRC_FILE")
    rm -f -- "$CHECKPOINT_NETRC_FILE"
fi
if [ -z "$netrc" ]; then
    log "no credentials, no checkpoint uploads in this step"
elif [ -e "$CHECKPOINT_NETRC_FILE" ]; then
    log "credentials read; the netrc could not be removed"
else
    log "credentials read; netrc removed"
fi
# The pipeline made the netrc's directory for it alone, and the step starts the tests once the
# directory is gone, so it goes after the line above.
if [ -n "$CHECKPOINT_NETRC_FILE" ]; then
    rmdir -- "$(dirname -- "$CHECKPOINT_NETRC_FILE")" 2>/dev/null
fi

if [ -z "$PROGRESS_DONE_FILE" ]; then
    log "PROGRESS_DONE_FILE is not set, so no checkpoint is taken in this step"
    exit 0
fi

scripts_dir=$(dirname "${BASH_SOURCE[0]}")
hash_file="${WORKSPACE}/${PROGRESS_TAR}.last_hash"
interval=$PROGRESS_INTERVAL
case $interval in
    "" | *[!0-9]* | 0) interval=600 ;;
esac
targets=""
[ -n "$netrc" ] && targets=$CHECKPOINT_TARGETS
# The mtime of XML_PATH at the last checkpoint, and 1 while a target lacks the content of the last
# checkpoint prepared.
last_mtime=0
behind=0

# Uploads the tarball to one URL and prints two words: the HTTP status of the last try, or 000 when
# no response arrived, and curl's exit code. A try is repeated only for a status that may clear on
# its own. Each try opens a new netrc stream, since a stream can be read only once. -q must come
# first: it keeps curl from reading a curlrc, which could turn on output that shows the
# Authorization header.
#
# curl's own messages stay out of the log, and its exit code is logged in their place. The pipeline
# searches the log of the step that runs the tests for signs of an infrastructure failure, and a
# message such as "Operation timed out" from an upload would make it take a failed test for one.
put_tarball() {
    local url=$1 attempt status curl_exit
    for attempt in 1 2 3; do
        status=$(curl -q -s -o /dev/null -w '%{http_code}' --max-time 300 \
            --netrc-file <(printf '%s\n' "$netrc") -T "${WORKSPACE}/${PROGRESS_TAR}" "$url" \
            2> /dev/null)
        curl_exit=$?
        case $status in
            "" | 000 | 408 | 429 | 5??) ;;
            *) break ;;
        esac
        [ "$attempt" -lt 3 ] && sleep 10
    done
    echo "${status:-000} ${curl_exit}"
}

# Uploads the checkpoint that progress_snapshot.sh just prepared to each target, in order, that
# lacks its content, and records the content each target received. The content becomes the last
# checkpoint's once every target has it. Without credentials there is no target, and the content is
# not recorded: a later step in this workspace that has credentials must still upload it.
upload_checkpoint() {
    local label=$1 pending target location url received status curl_exit message rejected_var
    local every_target=1
    [ -z "$targets" ] && return
    pending=$(cat "${hash_file}.pending" 2>/dev/null)
    for target in $targets; do
        location=${target%%=*}
        url=${target#*=}
        received=$(cat "${hash_file}.${location}" 2>/dev/null)
        if [ -n "$pending" ] && [ "$received" = "$pending" ]; then
            continue
        fi
        read -r status curl_exit <<< "$(put_tarball "$url")"
        if [ "$status" = 200 ] || [ "$status" = 201 ]; then
            printf '%s\n' "$pending" > "${hash_file}.${location}"
            log "${label} uploaded to ${location}"
            continue
        fi
        every_target=0
        message="${label} upload to ${location} failed (non-fatal): HTTP ${status}"
        [ "$curl_exit" != 0 ] && message="${message} (curl exit ${curl_exit})"
        if [ "$status" = 401 ] || [ "$status" = 403 ]; then
            rejected_var="CHECKPOINT_REJECTED_${location}"
            if [[ $rejected_var =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -n "${!rejected_var}" ]; then
                message="${message}. ${!rejected_var//HTTPSTATUS/$status}"
            fi
        fi
        log "$message"
    done
    if [ "$every_target" = 1 ]; then
        mv -f "${hash_file}.pending" "$hash_file"
        behind=0
    else
        behind=1
    fi
}

# Prepares a checkpoint and uploads it; "final" takes the final snapshot, which skips the checks for
# new results.
take_checkpoint() {
    local final=$1 label=$LABEL_PREFIX mtime collected prepared
    if [ "$final" = final ]; then
        label=$FINAL_LABEL
        if [ -n "$FINAL_REQUIRES" ] && [ ! -e "$FINAL_REQUIRES" ]; then
            return
        fi
    elif [ -n "$SLURM_SSH_STAT_CMD" ]; then
        bash "${scripts_dir}/progress_collect_slurm.sh"
        collected=$?
        # 3 means nothing new: the snapshot still runs, so that a target that missed the last
        # checkpoint receives it. After a failed copy the results may be partial, so this pass takes
        # no checkpoint.
        if [ "$collected" -ne 0 ] && [ "$collected" -ne 3 ]; then
            log "${label} skipped: copying the results back failed" \
                "(progress_collect_slurm.sh exited ${collected})"
            return
        fi
    elif [ -n "$XML_PATH" ]; then
        mtime=$(stat -c %Y "$XML_PATH" 2>/dev/null || echo 0)
        if [ "$mtime" -le "$last_mtime" ] && [ "$behind" = 0 ]; then
            return
        fi
        [ "$mtime" -gt "$last_mtime" ] && last_mtime=$mtime
        label="${label} (mtime=${mtime})"
    fi
    # Until a Slurm batch job writes its results, the copy above brings nothing back. The stage's
    # folder then holds no results, or does not exist yet, so there is nothing to take a checkpoint
    # of.
    if [ "$final" != final ] && [ -n "$SNAPSHOT_REQUIRES" ] && [ ! -e "$SNAPSHOT_REQUIRES" ]; then
        return
    fi

    LABEL=$label FINAL_SNAPSHOT=${final:+1} PROGRESS_HASH_FILE=$hash_file \
        bash "${scripts_dir}/progress_snapshot.sh"
    prepared=$?
    case $prepared in
        0) upload_checkpoint "$label" ;;
        3) ;;
        *) log "${label} not prepared (progress_snapshot.sh exited ${prepared})" ;;
    esac
}

# Returns 1 once PROGRESS_STOP_FILE exists. Otherwise marks a checkpoint as under way. The mark
# comes before the check, so that the step either sees the mark or the watcher sees the stop file.
begin_checkpoint() {
    [ -z "$PROGRESS_STOP_FILE" ] && return 0
    : > "${PROGRESS_STOP_FILE}.busy"
    if [ -e "$PROGRESS_STOP_FILE" ]; then
        rm -f "${PROGRESS_STOP_FILE}.busy"
        return 1
    fi
    return 0
}

end_checkpoint() {
    if [ -n "$PROGRESS_STOP_FILE" ]; then
        rm -f "${PROGRESS_STOP_FILE}.busy"
    fi
}

while ! tests_done; do
    # Check for the end of the tests every 5 seconds, so that the step does not wait out a whole
    # interval once they end.
    remaining=$interval
    while [ "$remaining" -gt 0 ] && ! tests_done; do
        nap=5
        [ "$remaining" -lt "$nap" ] && nap=$remaining
        sleep "$nap"
        remaining=$((remaining - nap))
    done
    tests_done && break
    if begin_checkpoint; then
        take_checkpoint
        end_checkpoint
    fi
done
if [ ! -e "$PROGRESS_DONE_FILE" ]; then
    log "the step signalled the end of the tests without the done file"
fi
take_checkpoint final
log "watcher exiting"
exit 0
