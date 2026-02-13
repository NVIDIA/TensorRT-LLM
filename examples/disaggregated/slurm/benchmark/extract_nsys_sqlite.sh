#!/bin/bash
set -euo pipefail

# Script to launch a SLURM job that extracts SQLite files from nsys profiles
# Usage: ./extract_nsys_sqlite.sh [OPTIONS] --paths <path1> [<path2> ...]
#
# The script will find all .nsys-rep files in the specified paths (directly or one level nested)
# and extract SQLite files to the same location as the nsys profiles.

# Hardcoded values from config.yaml
container_image="/lustre/fsw/coreai_comparch_trtllm/bbuddharaju/sqsh/tensorrt_llm_pytorch_25.12_py3_aarch64_ubuntu24.04_trt10.14.1.48_skip_tritondevel_202601230553_10896.sqsh"
container_mount="/lustre/fsw/coreai_comparch_trtllm:/lustre/fsw/coreai_comparch_trtllm"
partition="gb300"
account="coreai_comparch_trtllm"
time_limit="01:00:00"
job_name="nsys_sqlite_extract"
dry_run=false
paths=()

# Filtering options (defaults)
server_filter="gen_only"  # gen_only or gen_ctx
gpu_filter="gpu0_only"    # gpu0_only or all_gpu

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --container-image) container_image="$2"; shift 2 ;;
        --container-mount) container_mount="$2"; shift 2 ;;
        --partition) partition="$2"; shift 2 ;;
        --account) account="$2"; shift 2 ;;
        --time) time_limit="$2"; shift 2 ;;
        --job-name) job_name="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --gen-only) server_filter="gen_only"; shift ;;
        --gen-ctx) server_filter="gen_ctx"; shift ;;
        --gpu0-only) gpu_filter="gpu0_only"; shift ;;
        --all-gpu) gpu_filter="all_gpu"; shift ;;
        --paths)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                paths+=("$1")
                shift
            done
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] --paths <path1> [<path2> ...]"
            echo ""
            echo "Server filter options (mutually exclusive):"
            echo "  --gen-only          Extract GEN server profiles only (default)"
            echo "  --gen-ctx           Extract both GEN and CTX server profiles"
            echo ""
            echo "GPU filter options (mutually exclusive):"
            echo "  --gpu0-only         Extract GPU 0 profiles only (default)"
            echo "  --all-gpu           Extract all GPU profiles"
            echo ""
            echo "Other options:"
            echo "  --container-image   Container image to use"
            echo "  --container-mount   Container mount paths"
            echo "  --partition         SLURM partition (default: gb300)"
            echo "  --account           SLURM account (default: coreai_comparch_trtllm)"
            echo "  --time              Time limit for the job (default: 01:00:00)"
            echo "  --job-name          SLURM job name (default: nsys_sqlite_extract)"
            echo "  --dry-run           Print commands without executing"
            echo "  --paths             One or more paths to search for nsys profiles"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ ${#paths[@]} -eq 0 ]; then
    echo "Error: At least one path must be specified with --paths"
    echo "Use -h or --help for usage information"
    exit 1
fi

# Validate paths exist and convert to absolute paths
abs_paths=()
for path in "${paths[@]}"; do
    if [ ! -d "$path" ]; then
        echo "Error: Path does not exist: $path"
        exit 1
    fi
    # Convert to absolute path
    abs_paths+=("$(cd "$path" && pwd)")
done
paths=("${abs_paths[@]}")

echo "=============================================="
echo "NSYS SQLite Extraction Configuration"
echo "=============================================="
echo "Paths to search:"
for path in "${paths[@]}"; do
    echo "  - $path"
done
echo ""
echo "Filters:"
echo "  Server filter: ${server_filter}"
echo "  GPU filter: ${gpu_filter}"
echo ""
echo "Container image: ${container_image}"
echo "Container mount: ${container_mount:-<none>}"
echo "Partition: ${partition}"
echo "Account: ${account:-<none>}"
echo "Time limit: ${time_limit}"
echo "Job name: ${job_name}"
echo "Dry run: ${dry_run}"
echo "=============================================="

# Build file name patterns based on filters
# File pattern: nsys_worker_proc_{GEN|CTX}_{instance}_{gpu}.nsys-rep
build_find_patterns() {
    local patterns=()
    
    # Server filter patterns
    local server_types=()
    if [ "$server_filter" = "gen_only" ]; then
        server_types=("GEN")
    else
        server_types=("GEN" "CTX")
    fi
    
    # GPU filter patterns
    local gpu_suffix=""
    if [ "$gpu_filter" = "gpu0_only" ]; then
        gpu_suffix="_0.nsys-rep"
    else
        gpu_suffix="_*.nsys-rep"
    fi
    
    for server_type in "${server_types[@]}"; do
        patterns+=("nsys_worker_proc_${server_type}_*${gpu_suffix}")
    done
    
    echo "${patterns[@]}"
}

# Find all nsys-rep files (directly in path or one level nested) with filters
nsys_files=()
read -ra find_patterns <<< "$(build_find_patterns)"

for path in "${paths[@]}"; do
    for pattern in "${find_patterns[@]}"; do
        # Direct files in the path
        while IFS= read -r -d '' file; do
            nsys_files+=("$file")
        done < <(find "$path" -maxdepth 1 -name "$pattern" -print0 2>/dev/null)
        
        # Files one level nested
        while IFS= read -r -d '' file; do
            nsys_files+=("$file")
        done < <(find "$path" -mindepth 2 -maxdepth 2 -name "$pattern" -print0 2>/dev/null)
    done
done

# For gpu0_only filter, we need to ensure we only get files ending exactly with _0.nsys-rep
if [ "$gpu_filter" = "gpu0_only" ] && [ ${#nsys_files[@]} -gt 0 ]; then
    filtered_files=()
    for f in "${nsys_files[@]}"; do
        if [[ "$f" =~ _0\.nsys-rep$ ]]; then
            filtered_files+=("$f")
        fi
    done
    nsys_files=("${filtered_files[@]+"${filtered_files[@]}"}")
fi

if [ ${#nsys_files[@]} -eq 0 ]; then
    echo "No .nsys-rep files found in the specified paths"
    exit 0
fi

echo ""
echo "Found ${#nsys_files[@]} nsys profile(s) to process:"
for f in "${nsys_files[@]}"; do
    echo "  - $f"
done
echo ""

# Create work directory on shared filesystem (not /tmp which is node-local)
# Use the first search path as the base for the work directory
work_dir="${paths[0]}/.nsys_extract_$$_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${work_dir}"
extraction_script="${work_dir}/extract_sqlite.sh"

cat > "${extraction_script}" << 'EXTRACTION_SCRIPT_EOF'
#!/bin/bash
set -euo pipefail

# This script extracts SQLite files from nsys profiles
# It receives the list of files via a file passed as argument

files_list="$1"
total_files=$(wc -l < "$files_list")
current=0
failed=0
skipped=0

echo "Starting SQLite extraction for ${total_files} file(s)..."
echo ""

while IFS= read -r nsys_file; do
    current=$((current + 1))
    
    if [ ! -f "$nsys_file" ]; then
        echo "[${current}/${total_files}] SKIP: File not found: $nsys_file"
        skipped=$((skipped + 1))
        continue
    fi
    
    # Generate output sqlite filename (same location, .sqlite extension)
    sqlite_file="${nsys_file%.nsys-rep}.sqlite"
    
    # Check if sqlite already exists
    if [ -f "$sqlite_file" ]; then
        echo "[${current}/${total_files}] SKIP: SQLite already exists: $sqlite_file"
        skipped=$((skipped + 1))
        continue
    fi
    
    echo "[${current}/${total_files}] Processing: $nsys_file"
    echo "             Output: $sqlite_file"
    
    # Extract sqlite using nsys export
    if nsys export --type sqlite --output "$sqlite_file" "$nsys_file" 2>&1; then
        echo "             SUCCESS"
    else
        echo "             FAILED"
        failed=$((failed + 1))
    fi
    echo ""
done < "$files_list"

echo "=============================================="
echo "Extraction Summary"
echo "=============================================="
echo "Total files: ${total_files}"
echo "Processed: $((total_files - skipped))"
echo "Skipped: ${skipped}"
echo "Failed: ${failed}"
echo "=============================================="

if [ $failed -gt 0 ]; then
    exit 1
fi
EXTRACTION_SCRIPT_EOF

chmod +x "${extraction_script}"

# Create a file with the list of nsys files
files_list="${work_dir}/nsys_files.txt"
printf '%s\n' "${nsys_files[@]}" > "${files_list}"

# Build SLURM command
slurm_args=()
slurm_args+=(--job-name="${job_name}")
slurm_args+=(--time="${time_limit}")
slurm_args+=(--nodes=1)
slurm_args+=(--ntasks=1)
slurm_args+=(--partition="${partition}")
slurm_args+=(--output="${work_dir}/slurm_%j.out")
slurm_args+=(--error="${work_dir}/slurm_%j.err")

if [ -n "${account}" ]; then
    slurm_args+=(--account="${account}")
fi

# Build srun command
srun_args=()
if [ -n "${container_image}" ]; then
    srun_args+=(--container-image="${container_image}")
fi
if [ -n "${container_mount}" ]; then
    srun_args+=(--container-mounts="${container_mount}")
fi

# Build the full command
if [ -n "${container_image}" ]; then
    run_cmd="srun ${srun_args[*]} bash ${extraction_script} ${files_list}"
else
    run_cmd="bash ${extraction_script} ${files_list}"
fi

echo "Work directory: ${work_dir}"
echo "Extraction script: ${extraction_script}"
echo "Files list: ${files_list}"
echo ""

if [ "$dry_run" = true ]; then
    echo "DRY RUN - Would execute:"
    echo ""
    echo "sbatch ${slurm_args[*]} --wrap=\"${run_cmd}\""
    echo ""
    echo "Contents of extraction script:"
    echo "------------------------------"
    cat "${extraction_script}"
    echo "------------------------------"
    echo ""
    echo "Contents of files list:"
    echo "------------------------------"
    cat "${files_list}"
    echo "------------------------------"
else
    echo "Submitting SLURM job..."
    sbatch "${slurm_args[@]}" --wrap="${run_cmd}"
    echo ""
    echo "Job submitted. Use 'squeue -u \$USER' to check status."
    echo ""
    echo "Work directory: ${work_dir}"
    echo "  - Extraction script: ${extraction_script}"
    echo "  - Files list: ${files_list}"
    echo "  - SLURM logs will be at: ${work_dir}/slurm_<jobid>.out"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f ${work_dir}/slurm_*.out"
fi
