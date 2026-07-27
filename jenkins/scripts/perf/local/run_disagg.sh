#!/usr/bin/env bash
# Generate slurm_launch.sh for a local disaggregated perf-sanity run
# and submit it via sbatch. Run this on a SLURM login node.
#
# Usage:
#   bash run_disagg.sh -c configs/<cluster>.conf
#   bash run_disagg.sh --config configs/h100.conf
#
# All user-tunable variables live in the config file. See configs/example.conf
# for the full list and documentation. Copy it, edit it, point -c at it.
#
# Watch the job with: squeue -u $USER ; logs are under $work_dir.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------- Parse args ----------------
config_file=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            config_file="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            echo
            echo "Available configs:"
            ls -1 "$script_dir/configs/"*.conf 2>/dev/null || echo "  (none — copy configs/example.conf to get started)"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Usage: $0 -c <config-file>" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$config_file" ]]; then
    echo "ERROR: no config file given. Use -c <path-to-conf>." >&2
    echo "       See $script_dir/configs/example.conf for a template." >&2
    exit 1
fi
if [[ ! -f "$config_file" ]]; then
    echo "ERROR: config file not found: $config_file" >&2
    exit 1
fi

# ---------------- Load config ----------------
# shellcheck disable=SC1090
source "$config_file"

# Apply defaults for anything still unset
: "${trtllm:=CHANGE_ME}"
: "${work_dir:=$trtllm/perf_runs/disagg_$(date +%Y%m%d_%H%M%S)}"
: "${partition:=CHANGE_ME}"
: "${account:=coreai_comparch_trtllm}"
: "${job_name:=disagg_test}"
: "${image:=}"
: "${image_var:=LLM_DOCKER_IMAGE}"
: "${mounts:=CHANGE_ME}"
: "${llm_models_path:=/path/to/models}"
: "${install_mode:=wheel}"
: "${wheel_path:=CHANGE_ME}"
: "${build_wheel_flag:=}"
: "${capture_nsys_flag:=}"
: "${time_limit:=02:00:00}"

# Normalize test list: prefer 'test_ids' bash array if set, else fall back to
# legacy single 'test_id'. Either declares a non-empty list at this point.
if declare -p test_ids >/dev/null 2>&1 && [[ "$(declare -p test_ids)" == "declare -a"* ]]; then
    :  # test_ids already an array
elif [[ -n "${test_id:-}" ]]; then
    test_ids=("$test_id")
else
    test_ids=("perf/test_perf_sanity.py::test_e2e[disagg-e2e-CHANGE_ME]")
fi

# ---------------- Sanity checks ----------------
if [[ "$partition" == "CHANGE_ME" ]]; then
    echo "ERROR: please set 'partition' in $config_file." >&2
    exit 1
fi
if [[ ${#test_ids[@]} -eq 0 ]]; then
    echo "ERROR: 'test_ids' is empty. Set 'test_id' or 'test_ids' in $config_file." >&2
    exit 1
fi
for _tid in "${test_ids[@]}"; do
    if [[ "$_tid" == *"CHANGE_ME"* ]]; then
        echo "ERROR: please set 'test_id'/'test_ids' in $config_file (got placeholder: $_tid)." >&2
        exit 1
    fi
done
if [[ "$trtllm" == "CHANGE_ME" ]]; then
    echo "ERROR: please set 'trtllm' (path to TensorRT-LLM source tree) in $config_file." >&2
    exit 1
fi
if [[ "$mounts" == "CHANGE_ME" ]]; then
    echo "ERROR: please set 'mounts' (cluster-specific enroot bind mounts) in $config_file." >&2
    exit 1
fi
if [[ ! -d "$trtllm" ]]; then
    echo "ERROR: trtllm directory not found: $trtllm" >&2
    exit 1
fi
if [[ "$install_mode" == "wheel" ]]; then
    if [[ "$wheel_path" == "CHANGE_ME" || -z "$wheel_path" ]]; then
        echo "ERROR: install_mode=wheel but 'wheel_path' is not set." >&2
        echo "       Either set wheel_path to a local .whl file, or use install_mode=source." >&2
        exit 1
    fi
    if [[ ! -f "$wheel_path" ]]; then
        echo "ERROR: wheel_path file not found: $wheel_path" >&2
        exit 1
    fi
fi

mkdir -p "$work_dir"

# ---------------- Resolve docker image ----------------
# Two mutually-exclusive modes (image takes precedence if both set):
#   (a) 'image' non-empty  → used verbatim. 'image_var' is ignored.
#   (b) 'image' empty/unset → look up 'image_var' as a key in
#       $trtllm/jenkins/current_image_tags.properties (CI-updated file).
if [[ -n "$image" ]]; then
    image_source="pinned in conf"
else
    image_file="$trtllm/jenkins/current_image_tags.properties"
    if [[ ! -f "$image_file" ]]; then
        echo "ERROR: image properties file not found: $image_file" >&2
        exit 1
    fi
    # NOTE: grep returning no match would exit 1 and set -o pipefail+set -e would
    # silently kill the script before our error message — append '|| true' to absorb.
    image=$(grep "^${image_var}=" "$image_file" | head -1 | awk -F"=" '{print $2}' || true)
    if [[ -z "$image" ]]; then
        echo "ERROR: failed to parse '${image_var}' from $image_file" >&2
        echo "       (Either fix image_var to a key in that file, or set 'image' directly in the conf.)" >&2
        exit 1
    fi
    image_source="resolved from \$image_var=${image_var} via current_image_tags.properties"
fi
# urm.nvidia.com/ → urm.nvidia.com# (enroot URI form). No-op if user already passed enroot form.
image=${image//urm.nvidia.com\//urm.nvidia.com#}

echo "=== Configuration ==="
echo "config_file      : $config_file"
echo "trtllm           : $trtllm"
echo "work_dir         : $work_dir"
echo "partition        : $partition"
echo "account          : $account"
echo "job_name         : $job_name"
echo "image            : $image"
echo "image_source     : $image_source"
echo "mounts           : $mounts"
echo "llm_models_path  : $llm_models_path"
echo "install_mode     : $install_mode"
echo "wheel_path       : $wheel_path"
echo "time_limit       : $time_limit"
echo "num_tests        : ${#test_ids[@]}"
for _i in "${!test_ids[@]}"; do
    printf "  test[%02d]       : %s\n" "$_i" "${test_ids[$_i]}"
done
echo "======================"

# Build install-mode / wheel-path args for submit.py
install_args=(--install-mode "$install_mode")
if [[ "$install_mode" == "wheel" ]]; then
    install_args+=(--wheel-path "$wheel_path")
fi

# Default strip list — see note inside the loop.
: "${strip_sbatch_opts:=--segment}"

# Per-test loop: each test gets its own subdir (named by the test-id bracket),
# its own slurm_launch.sh, and its own sbatch submission. Failures are collected,
# not fatal, so a bad test_id doesn't stop the rest of the batch.
num_tests=${#test_ids[@]}
submitted_count=0
failed_tests=()

# CI machine-readable bookkeeping (consumed by trt_jenkins gen_disagg_junit.py).
# expected_tests.txt : every test_id we intend to run (for expected-vs-produced diff)
# failed_submit.txt  : test_id|reason for submit-time failures (no job / no xml)
# slurm_jobs.txt     : <jobid>|<test_id> for jobs that submitted (Jenkins polls these)
failed_submit_file="$work_dir/failed_submit.txt"
slurm_jobs_file="$work_dir/slurm_jobs.txt"
expected_file="$work_dir/expected_tests.txt"
: > "$failed_submit_file"; : > "$slurm_jobs_file"; : > "$expected_file"
printf '%s\n' "${test_ids[@]}" > "$expected_file"

for idx in "${!test_ids[@]}"; do
    tid="${test_ids[$idx]}"

    # Each test gets its own subdir named by the test-id bracket content, e.g.
    # 'disagg-e2e-<stem>' / 'aggr-ctx_only-<stem>'. This matches the trtllm-ci
    # multinode layout so downstream perf parsing (parse_perf_logs.py discover_cases,
    # which keys off 'disagg-*' / 'aggr-*' dir names + test_list.txt) and JUnit
    # generation find cases uniformly. submit.py writes test_list.txt + report.xml
    # into --work-dir (this subdir), so we don't create test_list.txt ourselves.
    case_name="${tid#*[}"       # strip up to and including '['
    case_name="${case_name%]*}" # strip trailing ']' and beyond
    test_work_dir="$work_dir/$case_name"
    test_job_name="${job_name}_${idx}"
    mkdir -p "$test_work_dir"

    echo
    echo "=== [$((idx+1))/$num_tests] $tid ==="
    echo "    work_dir : $test_work_dir"
    echo "    job_name : $test_job_name"

    # 1. Generate slurm_launch.sh for this test.
    # NOTE: we deliberately do NOT pass --draft-launch-sh — submit.py picks the
    # right draft (disaggregated/ vs aggregated/) based on the runtime_mode it
    # derives from the test id. Hard-coding the disagg draft here breaks
    # 'aggr-ctx_only-*' tests, which run in aggregated mode but would otherwise
    # be launched with the disagg orchestration (and produce no report.xml).
    if ! python3 "$trtllm/jenkins/scripts/perf/local/submit.py" \
        --test-list "$tid" \
        --launch-sh "$test_work_dir/slurm_launch.sh" \
        --install-sh "$trtllm/jenkins/scripts/perf/local/slurm_install.sh" \
        --run-sh "$trtllm/jenkins/scripts/perf/local/slurm_run.sh" \
        --llm-src "$trtllm" \
        --work-dir "$test_work_dir" \
        --partition "$partition" \
        --account "$account" \
        --job-name "$test_job_name" \
        --image "$image" \
        --mounts "$mounts" \
        --llm-models-root "$llm_models_path" \
        --time "$time_limit" \
        "${install_args[@]}" \
        $build_wheel_flag \
        $capture_nsys_flag; then
        echo "ERROR: submit.py failed for $tid — skipping." >&2
        failed_tests+=("$tid (submit.py failed)")
        echo "$tid|submit_py_failed" >> "$failed_submit_file"
        continue
    fi

    # Strip SBATCH directives unsupported by this cluster's SLURM.
    # Default '--segment': newer SLURM topology feature, missing on older clusters (e.g. EOS).
    if [[ -n "$strip_sbatch_opts" ]]; then
        IFS=',' read -ra _strip_arr <<< "$strip_sbatch_opts"
        for opt in "${_strip_arr[@]}"; do
            opt_trimmed="${opt#"${opt%%[![:space:]]*}"}"
            opt_trimmed="${opt_trimmed%"${opt_trimmed##*[![:space:]]}"}"
            [[ -z "$opt_trimmed" ]] && continue
            sed -i.bak "s|^#SBATCH \(${opt_trimmed}\)|# (stripped) #SBATCH \1|" "$test_work_dir/slurm_launch.sh"
        done
        rm -f "$test_work_dir/slurm_launch.sh.bak"
    fi

    # 2. Submit — capture job id, do NOT block (Jenkins polls squeue/sacct later, so a
    #    long queue wait survives SSH drops; --wait would hang one ssh for hours).
    jid_raw=$( cd "$test_work_dir" && sbatch --parsable slurm_launch.sh 2>>"$work_dir/sbatch.err" || true )
    jid="${jid_raw%%;*}"   # 'jobid' or 'jobid;cluster' -> jobid
    if [[ "$jid" =~ ^[0-9]+$ ]]; then
        echo "$jid|$tid" >> "$slurm_jobs_file"
        submitted_count=$((submitted_count + 1))
    else
        echo "ERROR: sbatch failed for $tid" >&2
        failed_tests+=("$tid (sbatch failed)")
        echo "$tid|sbatch_failed" >> "$failed_submit_file"
    fi
done

echo
echo "=== Submission summary ==="
echo "Submitted: $submitted_count / $num_tests"
if [[ ${#failed_tests[@]} -gt 0 ]]; then
    echo "Failed:"
    for ft in "${failed_tests[@]}"; do
        echo "  - $ft"
    done
fi
echo "Check status with:  squeue -u \$USER"
echo "Logs and outputs under: $work_dir"
[[ ${#failed_tests[@]} -gt 0 ]] && exit 1
exit 0
