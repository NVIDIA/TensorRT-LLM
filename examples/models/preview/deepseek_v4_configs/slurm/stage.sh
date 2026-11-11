#!/bin/bash
# Copies the checkpoint to node-local storage, once per node: a later job on the node reuses the
# copy. Servers load far faster from local NVMe than from NFS, where the context servers' scattered
# reads took hours. slurm/launch.sbatch runs it on every server node, outside the container.
#   usage: stage.sh <checkpoint dir> <node-local destination dir>
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

set -euo pipefail
src=$1 dst=$2
mkdir -p "$dst"
exec 9> "$dst.lock"
flock 9   # one copy per node, even with two jobs on it

list() { (cd "$src" && find . -type f ! -path './.cache/*' -printf '%P %s\n' | sort); }
want=$(list | md5sum | cut -d' ' -f1)
if [ "$(cat "$dst/.staged" 2> /dev/null)" = "$want" ]; then
    echo "$(hostname): $dst already staged"
    exit 0
fi

copy() {   # <relative path>: skip a file already copied whole, otherwise copy it atomically
    local s=$src/$1 d=$dst/$1
    if [ -f "$d" ] && [ "$(stat -c %s "$s")" = "$(stat -c %s "$d")" ]; then return 0; fi
    mkdir -p "$(dirname "$d")"
    cp "$s" "$d.part" && mv "$d.part" "$d"
}
export -f copy
export src dst
t0=$(date +%s)
list | cut -d' ' -f1 | xargs -P 16 -I{} bash -c 'copy "$1"' _ {}
[ "$(cd "$dst" && find . -type f ! -name '.staged' ! -name '*.part' -printf '%P %s\n' | sort | md5sum | cut -d' ' -f1)" = "$want" ] ||
    { echo "$(hostname): staged copy does not match $src" >&2; exit 1; }
echo "$want" > "$dst/.staged"
t=$(($(date +%s) - t0))
echo "$(hostname): staged $(du -sh "$dst" | cut -f1) in ${t}s"
