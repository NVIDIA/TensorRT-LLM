#!/bin/bash
set -x
set -e

# Function to check if server is running on port 12478
check_server() {
    if curl -s http://localhost:12478 > /dev/null; then
        echo "Server is running on http://localhost:12478"
        return 0
    fi
    return 1
}

# Function to display usage instructions
usage() {
    echo "Usage: $0 [-e <engine_path>] [-t <tokenizer_path>] [-d <evaltool_dir>] [-m <max_output_len>] [-c <device_count>] [-l <lookahead_config>]"
    exit 1
}

# Parse command line arguments
device_count=1
while getopts "e:t:d:m:c:l:" opt; do
    case "$opt" in
        e) engine_path=$OPTARG ;;
        t) tokenizer_path=$OPTARG ;;
        d) evaltool_dir=$OPTARG ;;
        m) max_output_len=$OPTARG ;;
        c)
            if [ -z "$OPTARG" ]; then
                :
            else
                device_count=$OPTARG
            fi
            ;;
        l) lookahead_config=$OPTARG ;;
        *) usage ;;
    esac
done

# Check if all arguments are provided
if [ -z "$engine_path" ] || [ -z "$tokenizer_path" ] || [ -z "$evaltool_dir" ]; then
    usage
fi

# Check if the paths are valid
if [ ! -e "$engine_path" ]; then
    echo "Error: engine_path '$engine_path' does not exist."
    exit 1
fi

if [ ! -e "$tokenizer_path" ]; then
    echo "Error: tokenizer_path '$tokenizer_path' does not exist."
    exit 1
fi

if [ ! -d "$evaltool_dir" ]; then
    echo "Error: evaltool_dir '$evaltool_dir' is not a valid directory."
    exit 1
fi

# Check if the server is already running
if check_server; then
    echo "Server is already running, skipping the rest of the script"
    exit 0
fi

# Install dependencies
trtllm_script_dir="evaltool/inference_server/trtllm"
cd ${evaltool_dir}/${trtllm_script_dir} || exit
pip install -r requirements.txt

file_path=$(dirname "$0")
cp -f ${file_path}/patch_for_server/server.py ${evaltool_dir}/${trtllm_script_dir}/
cp -f ${file_path}/patch_for_server/trt_llm_model.py ${evaltool_dir}/${trtllm_script_dir}/

# Run the server in the background and redirect logs to a file
export PYTHONUNBUFFERED=1
current_datetime=$(date '+%Y-%m-%d_%H-%M-%S')

if [ -z "$lookahead_config" ]; then
    nohup mpirun --allow-run-as-root --oversubscribe -np $device_count python3 server.py --engine_path "$engine_path" --tokenizer_path "$tokenizer_path" --max_output_len $max_output_len > server_$current_datetime.log 2>&1 &
else
    nohup mpirun --allow-run-as-root --oversubscribe -np $device_count python3 server.py --engine_path "$engine_path" --tokenizer_path "$tokenizer_path" --max_output_len $max_output_len --lookahead_config "$lookahead_config" > server_$current_datetime.log 2>&1 &
fi

# Wait for server to start and check status every 30 seconds for 60 attempts
for i in {1..60}; do
    sleep 30
    if check_server; then
        echo "Server started."
        exit 0
    fi
done

echo "Error: Server did not start on http://localhost:12478 within the expected time."
exit 1
