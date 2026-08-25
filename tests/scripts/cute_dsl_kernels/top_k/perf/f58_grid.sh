#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Full-grid re-run on the current tree (post P4 fold work), fresh f58 prefix.
BR=${GVR_BENCH_OUT:-./bench_results}
cd "$(dirname "$0")"
# PYTHONPATH must carry cutlass-dsl 4.5.x and the repo root; see README.md
export TMPDIR=/tmp/nsys_g$1; mkdir -p $TMPDIR
for UNIT in $2; do
  TAG=$(echo $UNIT | tr ':' '_')
  for ARM in ${ARMS:-pr st va vb}; do
    rm -f $BR/f58_${TAG}_${ARM}*.csv $BR/f58_${TAG}_${ARM}.sqlite
    CUDA_VISIBLE_DEVICES=$1 env ARM=$ARM UNITS=$UNIT \
      GVR_BSTAR=8192 GVR_KC=8192 GVR_CAPC=16384 OUT=$BR/f58_${TAG}_${ARM}.jsonl \
      nsys profile --trace=cuda,nvtx --force-overwrite true -o $BR/f58_${TAG}_${ARM} \
        python3 ab_steps.py >> $BR/f58_${TAG}.log 2>&1
    nsys stats -r nvtx_kern_sum --format csv -o $BR/f58_${TAG}_${ARM} \
      $BR/f58_${TAG}_${ARM}.nsys-rep >> $BR/f58_${TAG}.log 2>&1
    echo "f58 done ${TAG}_${ARM}" >> $BR/f58.log
  done
  # list tier (wf) under the C-only contract, same dataset
  rm -f $BR/f58w_${TAG}*.csv $BR/f58w_${TAG}.sqlite
  CUDA_VISIBLE_DEVICES=$1 env ARM=wf UNITS=$UNIT CONLY=1 CONLY_RANK=2 \  # codespell:ignore
    GVR_BSTAR=8192 GVR_KC=8192 GVR_CAPC=16384 OUT=$BR/f58w_${TAG}.jsonl \
    nsys profile --trace=cuda,nvtx --force-overwrite true -o $BR/f58w_${TAG} \
      python3 ab_steps.py >> $BR/f58_${TAG}.log 2>&1
  nsys stats -r nvtx_kern_sum --format csv -o $BR/f58w_${TAG} \
    $BR/f58w_${TAG}.nsys-rep >> $BR/f58_${TAG}.log 2>&1
  echo "f58 done ${TAG}_wf" >> $BR/f58.log
done
echo "f58 GPU$1 ALL DONE" >> $BR/f58.log
