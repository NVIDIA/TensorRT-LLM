#! /bin/bash

echo "commit id: $TRT_LLM_GIT_COMMIT"
echo "ucx info: $(ucx_info -v)"
echo "hostname: $(hostname)"

hostname=$(hostname)
short_hostname=$(echo "$hostname" | awk -F'.' '{print $1}')
echo "short_hostname: ${short_hostname}"

config_file=$1

# Check and replace hostname settings in config_file
if [ -f "$config_file" ]; then
    # Use sed to find hostname line and check if replacement is needed
    if grep -q "hostname:" "$config_file"; then
        # Extract current hostname value from config
        current_hostname=$(grep "hostname:" "$config_file" | sed 's/.*hostname:[ ]*//' | awk '{print $1}')

        if [ "$current_hostname" != "$short_hostname" ]; then
            echo "Replacing hostname '$current_hostname' with '$short_hostname' in $config_file"
            # Use sed to replace hostname value
            sed -i "s/hostname:[ ]*[^ ]*/hostname: $short_hostname/" "$config_file"
        else
            echo "Hostname '$current_hostname' already matches '$short_hostname', no change needed"
        fi
    else
        echo "No hostname setting found in $config_file"
    fi
else
    echo "Config file $config_file not found"
fi

trtllm-serve disaggregated -c ${config_file} -t 1800 -r 7200
