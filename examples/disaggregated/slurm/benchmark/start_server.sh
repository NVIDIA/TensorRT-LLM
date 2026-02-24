#!/bin/bash
set -u
set -e
set -x

config_file=$1

trtllm-serve disaggregated -c ${config_file} -t 7200 -r 7200
