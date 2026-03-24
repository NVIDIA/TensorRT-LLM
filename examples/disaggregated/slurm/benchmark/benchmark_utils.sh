#!/bin/bash
# Shared utility functions for disaggregated benchmark scripts.
#
# Globals set by setup_start_logs (used by do_process_all_logs "clean" mode):
#   tmp_start_logs  — path to temporary start log directory

do_get_logs(){
    local input_file="$1"
    local output_file="$2"
    local mode="$3"
    local start_line="$4"
    # check mode is ctx or gen
    if [ "${mode}" = "ctx" ]; then
        sed -n "${start_line},\$p" "${input_file}" | grep -a "'num_generation_tokens': 0" > "${output_file}" || true
    elif [ "${mode}" = "gen" ]; then
        sed -n "${start_line},\$p" "${input_file}" | grep -a "'num_ctx_requests': 0, 'num_ctx_tokens': 0" > "${output_file}" || true
    else
        echo "Invalid mode: ${mode}"
        return 1
    fi
    return 0
}

do_process_all_logs(){
    local input_folder="$1"
    local output_folder="$2"
    local mode="$3"
    local log_path="$4"
    if [ "${mode}" != "line" ] && [ "${mode}" != "log" ] && [ "${mode}" != "clean" ]; then
        echo "Invalid mode: ${mode}"
        exit 1
    fi
    local ctx_log
    local ctx_num
    local gen_log
    local gen_num
    local line_count
    local start_line
    for ctx_log in "${input_folder}"/3_output_CTX_*.log; do
        if [ -f "${ctx_log}" ]; then
            ctx_num=$(basename "${ctx_log}" | sed 's/3_output_CTX_\([0-9]*\)\.log/\1/')
            if [ "${mode}" = "line" ]; then
                line_count=$(wc -l < "${ctx_log}")
                echo "${line_count}" > "${output_folder}/ctx_only_line_${ctx_num}.txt"
            elif [ "${mode}" = "log" ]; then
                if [ ! -f "${output_folder}/ctx_only_line_${ctx_num}.txt" ]; then
                    start_line=0
                else
                    start_line=$(cat "${output_folder}/ctx_only_line_${ctx_num}.txt")
                    rm -f "${output_folder}/ctx_only_line_${ctx_num}.txt"
                fi
                do_get_logs "${ctx_log}" "${output_folder}/ctx_only_${ctx_num}.txt" "ctx" "${start_line}"
            elif [ "${mode}" = "clean" ]; then
                rm -f "${ctx_log}"
            fi
        fi
    done
    # process all the gen log files in the input folder
    for gen_log in "${input_folder}"/3_output_GEN_*.log; do
        if [ -f "${gen_log}" ]; then
            gen_num=$(basename "${gen_log}" | sed 's/3_output_GEN_\([0-9]*\)\.log/\1/')
            if [ "${mode}" = "line" ]; then
                line_count=$(wc -l < "${gen_log}")
                echo "${line_count}" > "${output_folder}/gen_only_line_${gen_num}.txt"
            elif [ "${mode}" = "log" ]; then
                if [ ! -f "${output_folder}/gen_only_line_${gen_num}.txt" ]; then
                    start_line=0
                else
                    start_line=$(cat "${output_folder}/gen_only_line_${gen_num}.txt")
                    rm -f "${output_folder}/gen_only_line_${gen_num}.txt"
                fi
                do_get_logs "${gen_log}" "${output_folder}/gen_only_${gen_num}.txt" "gen" "${start_line}"
            elif [ "${mode}" = "clean" ]; then
                rm -f "${gen_log}"
            fi
        fi
    done
    if [ "${mode}" = "clean" ]; then
        if [ -n "${tmp_start_logs:-}" ] && [ -d "${tmp_start_logs}" ]; then
            mkdir -p "${log_path}/start_logs"
            cp "${tmp_start_logs}"/3_output_CTX_*.log "${log_path}/start_logs/" 2>/dev/null || true
            cp "${tmp_start_logs}"/3_output_GEN_*.log "${log_path}/start_logs/" 2>/dev/null || true
            rm -rf "${tmp_start_logs}"
        fi
    fi
}

setup_start_logs(){
    local job_id="$1"
    local log_path="$2"
    tmp_start_logs="/tmp/${job_id}/start_logs"
    mkdir -p "${tmp_start_logs}"
    cp "${log_path}"/3_output_CTX_*.log "${tmp_start_logs}/" 2>/dev/null || true
    cp "${log_path}"/3_output_GEN_*.log "${tmp_start_logs}/" 2>/dev/null || true
}
