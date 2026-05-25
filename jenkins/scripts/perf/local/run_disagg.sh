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
: "${trtllm:=/localhome/swqa/fzhu/TensorRT-LLM}"
: "${work_dir:=$HOME/perf_runs/disagg_$(date +%Y%m%d_%H%M%S)}"
: "${partition:=CHANGE_ME}"
: "${account:=coreai_comparch_trtllm}"
: "${job_name:=disagg_test}"
: "${image_var:=LLM_DOCKER_IMAGE}"
: "${mounts:=/lustre:/lustre,/home:/home}"
: "${llm_models_path:=/path/to/models}"
: "${test_id:=perf/test_perf_sanity.py::test_e2e[disagg-e2e-CHANGE_ME]}"
: "${install_mode:=wheel}"
: "${wheel_path:=CHANGE_ME}"
: "${build_wheel_flag:=}"
: "${capture_nsys_flag:=}"
: "${time_limit:=02:00:00}"

# ---------------- Sanity checks ----------------
if [[ "$partition" == "CHANGE_ME" ]]; then
    echo "ERROR: please set 'partition' in $config_file." >&2
    exit 1
fi
if [[ "$test_id" == *"CHANGE_ME"* ]]; then
    echo "ERROR: please set 'test_id' in $config_file." >&2
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
image_file="$trtllm/jenkins/current_image_tags.properties"
if [[ ! -f "$image_file" ]]; then
    echo "ERROR: image properties file not found: $image_file" >&2
    exit 1
fi
image=$(grep "^${image_var}=" "$image_file" | head -1 | awk -F"=" '{print $2}')
image=${image//urm.nvidia.com\//urm.nvidia.com#}
if [[ -z "$image" ]]; then
    echo "ERROR: failed to parse ${image_var} from $image_file" >&2
    exit 1
fi

echo "=== Configuration ==="
echo "config_file      : $config_file"
echo "trtllm           : $trtllm"
echo "work_dir         : $work_dir"
echo "partition        : $partition"
echo "account          : $account"
echo "job_name         : $job_name"
echo "image_var        : $image_var"
echo "image            : $image"
echo "mounts           : $mounts"
echo "llm_models_path  : $llm_models_path"
echo "test_id          : $test_id"
echo "install_mode     : $install_mode"
echo "wheel_path       : $wheel_path"
echo "time_limit       : $time_limit"
echo "======================"

# Build install-mode / wheel-path args for submit.py
install_args=(--install-mode "$install_mode")
if [[ "$install_mode" == "wheel" ]]; then
    install_args+=(--wheel-path "$wheel_path")
fi

# 1. Generate slurm_launch.sh
python3 "$trtllm/jenkins/scripts/perf/local/submit.py" \
    --test-list "$test_id" \
    --draft-launch-sh "$trtllm/jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh" \
    --launch-sh "$work_dir/slurm_launch.sh" \
    --install-sh "$trtllm/jenkins/scripts/perf/local/slurm_install.sh" \
    --run-sh "$trtllm/jenkins/scripts/perf/local/slurm_run.sh" \
    --llm-src "$trtllm" \
    --work-dir "$work_dir" \
    --partition "$partition" \
    --account "$account" \
    --job-name "$job_name" \
    --image "$image" \
    --mounts "$mounts" \
    --llm-models-root "$llm_models_path" \
    --time "$time_limit" \
    "${install_args[@]}" \
    $build_wheel_flag \
    $capture_nsys_flag

echo
echo "Generated: $work_dir/slurm_launch.sh"
echo "Submitting to SLURM..."

# 2. Submit
cd "$work_dir"
sbatch slurm_launch.sh

echo
echo "Done. Check status with:  squeue -u \$USER"
echo "Logs and outputs will appear under: $work_dir"
