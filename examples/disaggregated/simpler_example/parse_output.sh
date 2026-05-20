#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# extract text of each choice from output.json and output to output.txt
perl -0ne 'while (/"text"[[:space:]]*:[[:space:]]*"((?:\\.|[^"\\])*)"/gs) { print "\"$1\"\n"; }' \
    "${SCRIPT_DIR}/output.json" \
    | jq -r 'gsub("[\r\n]+"; " ")' > "${SCRIPT_DIR}/output.txt"