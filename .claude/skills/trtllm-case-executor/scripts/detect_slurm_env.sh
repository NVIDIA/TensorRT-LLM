#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Detect a SLURM cluster's environment for the current user in a single pass:
#   - account associations and a default account
#   - visible partitions (sinfo) and a default partition validated via
#     `sbatch --test-only` for the user's account
#   - per-partition hardware: Arch (scontrol show node), GRES (sinfo %G),
#     gpus_per_node, requires_gres
#   - the cluster's compatible PMIx plugin (srun --mpi=list)
#
# Run locally for local_slurm; stream over SSH for remote_slurm.
#
# Usage:
#   bash detect_slurm_env.sh                          # JSON, probe all up partitions
#   bash detect_slurm_env.sh --format=text            # human-readable
#   bash detect_slurm_env.sh --partitions=batch,gpu   # restrict hw probe to these
#
# Output (JSON, default):
#   {
#     "user": "<whoami>",
#     "default_account": "<first-account-or-null>",
#     "accounts": ["<acct1>", ...],
#     "default_partition": "<partition-or-null>",
#     "partitions": [
#       {
#         "name": "<part>",
#         "state": "<up|down|...>",
#         "time_limit": "<HH:MM:SS|infinite>",
#         "nodes": "<count>",
#         "arch": "x86_64|aarch64|null",
#         "gres": "gpu:8|(null)",
#         "gpus_per_node": <int|null>,
#         "requires_gres": true|false
#       },
#       ...
#     ],
#     "pmix": {
#       "available": ["pmix", "pmix_v3", "pmix_v4", "pmix_v5"],
#       "preferred": "pmix_v5|null"
#     },
#     "errors": ["<diagnostic>", ...]
#   }
#
# Exit codes:
#   0  success (account and at least one partition resolved, no errors)
#   1  SLURM tools not available
#   2  partial success (one or more errors collected; see "errors")

set -u
set -o pipefail

FORMAT="json"
PARTITIONS_ARG=""
for arg in "$@"; do
    case "$arg" in
        --format=json) FORMAT="json" ;;
        --format=text) FORMAT="text" ;;
        --partitions=*) PARTITIONS_ARG="${arg#--partitions=}" ;;
        --partitions)
            echo "--partitions requires a value (use --partitions=p1,p2)" >&2
            exit 1
            ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

ERRORS=()

if ! command -v sacctmgr >/dev/null 2>&1 || ! command -v sinfo >/dev/null 2>&1; then
    ERRORS+=("SLURM client tools (sacctmgr/sinfo) not found in PATH")
    if [ "$FORMAT" = "json" ]; then
        printf '{"user":"%s","default_account":null,"accounts":[],"default_partition":null,"partitions":[],"pmix":{"available":[],"preferred":null},"errors":["SLURM client tools not found"]}\n' "$(whoami)"
    else
        echo "ERROR: SLURM client tools (sacctmgr / sinfo) not found in PATH" >&2
    fi
    exit 1
fi

USER_NAME="$(whoami)"

# --- Accounts -----------------------------------------------------------------
# `sacctmgr -nP show assoc where user=$USER format=account` returns one
# account per line. Some sites duplicate entries (per-cluster, per-partition);
# de-duplicate while preserving order.
ACCOUNTS_RAW="$(sacctmgr -nP show assoc where user="$USER_NAME" format=account 2>/dev/null || true)"
ACCOUNTS=()
if [ -n "$ACCOUNTS_RAW" ]; then
    while IFS= read -r line; do
        line="${line//[[:space:]]/}"
        [ -z "$line" ] && continue
        # de-dupe
        local_seen=0
        for a in "${ACCOUNTS[@]:-}"; do
            [ "$a" = "$line" ] && local_seen=1 && break
        done
        [ "$local_seen" -eq 0 ] && ACCOUNTS+=("$line")
    done <<< "$ACCOUNTS_RAW"
fi
DEFAULT_ACCOUNT="${ACCOUNTS[0]:-}"
if [ -z "$DEFAULT_ACCOUNT" ]; then
    ERRORS+=("no SLURM account associations for user $USER_NAME")
fi

# --- Partitions (basic sinfo fields) -----------------------------------------
# `sinfo -h -o "%P|%a|%l|%D"` — partition (default has a trailing '*'),
# avail (up/down), time limit, node count.

PART_LINES=()   # "name|state|time_limit|nodes" in sinfo order, de-duped by name
UP_PARTS=()     # ordered list of partition names with state=up
PART_RAW="$(sinfo -h -o '%P|%a|%l|%D' 2>/dev/null || true)"
if [ -n "$PART_RAW" ]; then
    declare -A SEEN_PART
    while IFS='|' read -r pname pavail ptime pnodes; do
        pname_trim="${pname%[*]}"   # strip default '*'
        [ -z "$pname_trim" ] && continue
        if [ -z "${SEEN_PART[$pname_trim]:-}" ]; then
            SEEN_PART["$pname_trim"]=1
            PART_LINES+=("$pname_trim|$pavail|$ptime|$pnodes")
            [ "$pavail" = "up" ] && UP_PARTS+=("$pname_trim")
        fi
    done <<< "$PART_RAW"
fi
if [ "${#PART_LINES[@]}" -eq 0 ]; then
    ERRORS+=("sinfo returned no partitions")
fi

# --- Resolve the hardware-probe target list ----------------------------------
# Default: every up partition. If --partitions=... was provided, restrict to
# those (intersection with up partitions only; unknown names are warned).
HW_TARGETS=()
if [ -n "$PARTITIONS_ARG" ]; then
    IFS=',' read -r -a HW_TARGETS_REQUESTED <<< "$PARTITIONS_ARG"
    declare -A UP_SET
    for u in "${UP_PARTS[@]:-}"; do UP_SET["$u"]=1; done
    for r in "${HW_TARGETS_REQUESTED[@]:-}"; do
        [ -z "$r" ] && continue
        if [ -n "${UP_SET[$r]:-}" ]; then
            HW_TARGETS+=("$r")
        else
            ERRORS+=("--partitions=$r: not an up partition (skipping hw probe)")
        fi
    done
else
    HW_TARGETS=("${UP_PARTS[@]:-}")
fi

# --- Per-partition arch + gres -----------------------------------------------
# PART_HW[name] = "arch|gres|gpus_per_node|requires_gres"; absent if not probed.
declare -A PART_HW
for part in "${HW_TARGETS[@]:-}"; do
    [ -z "$part" ] && continue

    NODE="$(sinfo -p "$part" -h -o '%n' 2>/dev/null | head -1 | awk '{print $1}')"
    ARCH=""
    if [ -n "$NODE" ]; then
        ARCH="$(scontrol show node "$NODE" 2>/dev/null | awk -F= '/^ *Arch=/{print $2; exit}' | tr -d '[:space:]')"
    fi
    if [ -z "$ARCH" ]; then
        ERRORS+=("partition $part: could not determine Arch (node=${NODE:-unknown})")
    fi

    GRES_RAW="$(sinfo -p "$part" -h -o '%G' 2>/dev/null | head -1 | sed 's/[[:space:]]*$//')"
    if [ -z "$GRES_RAW" ]; then
        GRES_RAW="(null)"
    fi

    REQUIRES_GRES=false
    GPUS_PER_NODE=""
    case "$GRES_RAW" in
        gpu:*)
            REQUIRES_GRES=true
            # gpu:4, gpu:4(IDX:0-3), gpu:a100:8, gpu:a100:8(...)
            tmp="${GRES_RAW%%(*}"        # strip "(...)" suffix
            tmp="${tmp##*:}"             # last colon-separated token = count
            if [[ "$tmp" =~ ^[0-9]+$ ]]; then
                GPUS_PER_NODE="$tmp"
            fi
            ;;
    esac

    PART_HW["$part"]="$ARCH|$GRES_RAW|$GPUS_PER_NODE|$REQUIRES_GRES"
done

# --- pmix plugin detection ---------------------------------------------------
# `srun --mpi=list` is a client-side query (no allocation required). Output
# format varies across SLURM versions; harvest every token that starts with
# 'pmix' and dedupe.
PMIX_AVAILABLE=()
PMIX_PREFERRED=""
if command -v srun >/dev/null 2>&1; then
    MPI_LIST_RAW="$(srun --mpi=list 2>&1 || true)"
    if [ -n "$MPI_LIST_RAW" ]; then
        declare -A PMIX_SEEN
        while IFS= read -r tok; do
            tok="${tok//[[:space:]]/}"
            tok="${tok#-}"
            tok="${tok#*}"
            case "$tok" in
                pmix|pmix_v*)
                    if [ -z "${PMIX_SEEN[$tok]:-}" ]; then
                        PMIX_SEEN["$tok"]=1
                        PMIX_AVAILABLE+=("$tok")
                    fi
                    ;;
            esac
        done < <(printf '%s\n' "$MPI_LIST_RAW" | tr ',' '\n')
    else
        ERRORS+=("srun --mpi=list returned no output")
    fi

    # Preferred: highest pmix_vN, else base 'pmix'.
    BEST_VER=-1
    for p in "${PMIX_AVAILABLE[@]:-}"; do
        [ -z "$p" ] && continue
        if [[ "$p" =~ ^pmix_v([0-9]+)$ ]]; then
            v="${BASH_REMATCH[1]}"
            if [ "$v" -gt "$BEST_VER" ]; then
                BEST_VER="$v"
                PMIX_PREFERRED="$p"
            fi
        fi
    done
    if [ -z "$PMIX_PREFERRED" ]; then
        for p in "${PMIX_AVAILABLE[@]:-}"; do
            if [ "$p" = "pmix" ]; then
                PMIX_PREFERRED="pmix"
                break
            fi
        done
    fi
    if [ -z "$PMIX_PREFERRED" ] && [ "${#PMIX_AVAILABLE[@]}" -eq 0 ]; then
        ERRORS+=("no pmix MPI plugin reported by 'srun --mpi=list'")
    fi
else
    ERRORS+=("srun not found; cannot enumerate MPI plugins")
fi

# --- Default partition selection ---------------------------------------------
# Verify actual submission access for each "up" partition using sbatch
# --test-only. A submission error indicates the account/QOS combination is
# not permitted. Among accessible partitions pick by priority:
#   1. Standard partitions  (no recognised suffix)
#   2. Backfill partitions  (*-backfill)
#   3. High-priority / admin partitions (*-hp, *-admin)
_part_priority() {
    local name="$1"
    case "$name" in
        *-hp|*-admin) echo 3 ;;
        *-backfill)   echo 2 ;;
        *)            echo 1 ;;
    esac
}

DEFAULT_PARTITION=""
ACCESSIBLE_PARTS=()
if [ -n "$DEFAULT_ACCOUNT" ] && [ "${#UP_PARTS[@]}" -gt 0 ]; then
    for pname in "${UP_PARTS[@]}"; do
        sbatch --test-only \
            --partition="$pname" \
            --account="$DEFAULT_ACCOUNT" \
            --nodes=1 --time=00:01:00 --ntasks=1 \
            --wrap="true" >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            ACCESSIBLE_PARTS+=("$pname")
        fi
    done
fi

if [ "${#ACCESSIBLE_PARTS[@]}" -gt 0 ]; then
    best_name=""
    best_score=99
    for pname in "${ACCESSIBLE_PARTS[@]}"; do
        score=$(_part_priority "$pname")
        if [ "$score" -lt "$best_score" ]; then
            best_score=$score
            best_name="$pname"
        fi
    done
    DEFAULT_PARTITION="$best_name"
fi

# Fallback: if sbatch --test-only is unavailable or all checks failed,
# use the first listed partition (original behaviour).
if [ -z "$DEFAULT_PARTITION" ] && [ "${#PART_LINES[@]}" -gt 0 ]; then
    ERRORS+=("partition access verification failed; falling back to first available partition")
    first="${PART_LINES[0]}"
    DEFAULT_PARTITION="${first%%|*}"
fi

# --- Emit --------------------------------------------------------------------
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    printf '%s' "$s"
}

emit_json() {
    local acc_json="[]"
    if [ "${#ACCOUNTS[@]}" -gt 0 ]; then
        acc_json="["
        local first=1
        for a in "${ACCOUNTS[@]}"; do
            if [ "$first" -eq 1 ]; then first=0; else acc_json+=","; fi
            acc_json+="\"$(json_escape "$a")\""
        done
        acc_json+="]"
    fi

    local part_json="[]"
    if [ "${#PART_LINES[@]}" -gt 0 ]; then
        part_json="["
        local first=1
        for line in "${PART_LINES[@]}"; do
            IFS='|' read -r p_name p_avail p_time p_nodes <<< "$line"
            if [ "$first" -eq 1 ]; then first=0; else part_json+=","; fi

            local arch_json="null"
            local gres_json="null"
            local gpn_json="null"
            local req_json="false"
            local hw="${PART_HW[$p_name]:-}"
            if [ -n "$hw" ]; then
                IFS='|' read -r h_arch h_gres h_gpn h_req <<< "$hw"
                [ -n "$h_arch" ] && arch_json="\"$(json_escape "$h_arch")\""
                gres_json="\"$(json_escape "$h_gres")\""
                [[ "$h_gpn" =~ ^[0-9]+$ ]] && gpn_json="$h_gpn"
                req_json="$h_req"
            fi

            part_json+="{\"name\":\"$(json_escape "$p_name")\","
            part_json+="\"state\":\"$(json_escape "$p_avail")\","
            part_json+="\"time_limit\":\"$(json_escape "$p_time")\","
            part_json+="\"nodes\":\"$(json_escape "$p_nodes")\","
            part_json+="\"arch\":$arch_json,"
            part_json+="\"gres\":$gres_json,"
            part_json+="\"gpus_per_node\":$gpn_json,"
            part_json+="\"requires_gres\":$req_json}"
        done
        part_json+="]"
    fi

    local pmix_avail_json="[]"
    if [ "${#PMIX_AVAILABLE[@]}" -gt 0 ]; then
        pmix_avail_json="["
        local first=1
        for p in "${PMIX_AVAILABLE[@]}"; do
            if [ "$first" -eq 1 ]; then first=0; else pmix_avail_json+=","; fi
            pmix_avail_json+="\"$(json_escape "$p")\""
        done
        pmix_avail_json+="]"
    fi
    local pmix_pref_json="null"
    [ -n "$PMIX_PREFERRED" ] && pmix_pref_json="\"$(json_escape "$PMIX_PREFERRED")\""

    local err_json="[]"
    if [ "${#ERRORS[@]}" -gt 0 ]; then
        err_json="["
        local first=1
        for e in "${ERRORS[@]}"; do
            if [ "$first" -eq 1 ]; then first=0; else err_json+=","; fi
            err_json+="\"$(json_escape "$e")\""
        done
        err_json+="]"
    fi

    local def_acc_json="null"
    [ -n "$DEFAULT_ACCOUNT" ] && def_acc_json="\"$(json_escape "$DEFAULT_ACCOUNT")\""
    local def_part_json="null"
    [ -n "$DEFAULT_PARTITION" ] && def_part_json="\"$(json_escape "$DEFAULT_PARTITION")\""

    printf '{"user":"%s","default_account":%s,"accounts":%s,"default_partition":%s,"partitions":%s,"pmix":{"available":%s,"preferred":%s},"errors":%s}\n' \
        "$(json_escape "$USER_NAME")" \
        "$def_acc_json" "$acc_json" \
        "$def_part_json" "$part_json" \
        "$pmix_avail_json" "$pmix_pref_json" \
        "$err_json"
}

emit_text() {
    echo "user: $USER_NAME"
    echo "default_account: ${DEFAULT_ACCOUNT:-<none>}"
    echo "accounts: ${ACCOUNTS[*]:-<none>}"
    echo "default_partition: ${DEFAULT_PARTITION:-<none>}"
    echo "partitions:"
    if [ "${#PART_LINES[@]}" -gt 0 ]; then
        printf '  %-20s %-6s %-12s %-6s %-10s %-20s %-12s %s\n' \
            NAME STATE TIME_LIMIT NODES ARCH GRES GPUS_PER_NODE REQUIRES_GRES
        for line in "${PART_LINES[@]}"; do
            IFS='|' read -r p_name p_avail p_time p_nodes <<< "$line"
            arch="-"; gres="-"; gpn="-"; req="-"
            hw="${PART_HW[$p_name]:-}"
            if [ -n "$hw" ]; then
                IFS='|' read -r h_arch h_gres h_gpn h_req <<< "$hw"
                arch="${h_arch:-<unknown>}"
                gres="$h_gres"
                gpn="${h_gpn:-<n/a>}"
                req="$h_req"
            fi
            printf '  %-20s %-6s %-12s %-6s %-10s %-20s %-12s %s\n' \
                "$p_name" "$p_avail" "$p_time" "$p_nodes" "$arch" "$gres" "$gpn" "$req"
        done
    else
        echo "  <none>"
    fi
    echo "pmix:"
    echo "  available: ${PMIX_AVAILABLE[*]:-<none>}"
    echo "  preferred: ${PMIX_PREFERRED:-<none>}"
    if [ "${#ERRORS[@]}" -gt 0 ]; then
        echo "errors:"
        for e in "${ERRORS[@]}"; do echo "  - $e"; done
    fi
}

if [ "$FORMAT" = "json" ]; then
    emit_json
else
    emit_text
fi

# Exit code
if [ -z "$DEFAULT_ACCOUNT" ] || [ -z "$DEFAULT_PARTITION" ]; then
    exit 2
fi
if [ "${#ERRORS[@]}" -gt 0 ]; then
    exit 2
fi
exit 0
