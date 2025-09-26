#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np


def save_raw(data: np.ndarray, filename: str):
    with open(filename, 'wb') as file:
        file.write(data.tobytes())


headElems = 256
nbKHeads = 1
headGrpSize = 32

beamWidth = 1

nbVHeads = nbKHeads
nbQHeads = nbKHeads * headGrpSize
inputElem = np.float16
inputElemSize = np.dtype(inputElem).itemsize
cacheElem = np.int8
cacheElemSize = np.dtype(cacheElem).itemsize

batchSize = 1
seqLen = 256

kScale = 1 if cacheElemSize == 2 else 1 / 4
vScale = kScale
qkScale = (headElems**-0.5) * kScale

dataBuf = open("data.bin", 'rb').read()
cache_data = np.frombuffer(dataBuf[0:cacheElemSize * headElems * seqLen *
                                   (nbKHeads + nbVHeads) * batchSize],
                           dtype=cacheElem)
offset = 0

k_shape = (batchSize, nbKHeads, seqLen, headElems)
offset_next = offset + np.prod(k_shape)
k = np.reshape(cache_data[offset:offset_next], k_shape)
offset = offset_next

v_shape = (batchSize, nbKHeads, seqLen, headElems)
offset_next = offset + np.prod(v_shape)
v = np.reshape(cache_data[offset:offset_next], v_shape)
offset = offset_next

io_data = np.frombuffer(dataBuf[cacheElemSize * offset:], dtype=inputElem)
offset = 0

input_shape = (batchSize, beamWidth, (nbQHeads + nbKHeads + nbVHeads),
               headElems)
offset_next = offset + np.prod(input_shape)
input = np.reshape(io_data[offset:offset_next], input_shape)
offset = offset_next

q = np.reshape(input[:, :, :nbQHeads, :],
               (batchSize, beamWidth, nbKHeads, headGrpSize, headElems))
q = np.transpose(q, axes=[0, 2, 1, 3, 4])
assert q.shape == (batchSize, nbKHeads, beamWidth, headGrpSize, headElems)
q = np.reshape(q, (batchSize, nbKHeads, beamWidth * headGrpSize, headElems))

ref = np.zeros((batchSize, nbKHeads, beamWidth, headGrpSize, headElems),
               dtype=np.float16)
for req in range(batchSize):
    for g in range(nbKHeads):
        qk = np.mat(q[req, g]).astype(np.float32) * np.mat(k[req, g]).astype(
            np.float32).T * qkScale
        row_max = np.max(qk, axis=1)
        qk = np.exp(qk - row_max).astype(np.float16).astype(np.float32)
        row_sum = np.sum(qk, axis=1)
        qk = qk / row_sum
        qkv = (qk.astype(np.float32) * np.mat(v).astype(np.float32)).astype(
            np.float16) * vScale
        ref[req, g] = np.reshape(np.array(qkv),
                                 (beamWidth, headGrpSize, headElems))

out_shape = (batchSize, beamWidth, nbQHeads, headElems)
offset_next = offset + np.prod(out_shape)
out = np.reshape(io_data[offset:offset_next], out_shape)
offset = offset_next
assert offset == io_data.shape[0]

ref_cpp = np.reshape(
    np.frombuffer(open("ref_cpp.bin", 'rb').read(), dtype=np.float32),
    ref.shape).astype(np.float16)


def is_close(a, b):
    return np.max(np.abs(a - b)) < 0.01


print("maxDiff: %f\n" % np.max(np.abs(ref - ref_cpp)))
assert is_close(ref, ref_cpp)

debug_refcheck = False  # only for batchSize 1 and seqLen 256
if debug_refcheck:
    #tiled to emulate kernel implementation (for no ctaRowMax update)
    q = np.reshape(q, (32, headElems))
    save_raw(np.transpose(np.reshape(q, (32, 8, 32)), axes=[1, 0, 2]),
             'q_8x32x32_f16.bin')
    k = np.reshape(k, (seqLen, headElems))
    save_raw(np.transpose(np.reshape(k, (4, 64, 8, 32)), axes=[2, 0, 1, 3]),
             'k_8x4x64x32_f16.bin')
    qk = np.mat(q.astype(np.float32)) * np.mat(k.astype(np.float32)).T
    qk_tiles = np.transpose(np.reshape(np.array(qk), (32, 4, 64)),
                            axes=[1, 0, 2])
    assert qk_tiles.shape == (4, 32, 64)
    save_raw(qk_tiles, 'qk_4x32x64_f32.bin')
    tile_row_max = np.max(qk_tiles, axis=2, keepdims=True)
    save_raw(tile_row_max, 'tileRowMax_4x32_f32.bin')
    x = np.exp(qk_tiles * qkScale - tile_row_max).astype(np.float16)
    save_raw(x, 'x_4x32x64_f16.bin')
    tile_row_sum = np.sum(x.astype(np.float32), axis=2, keepdims=True)
    save_raw(tile_row_sum, 'tileRowSum_4x32_f32.bin')

    cta_row_max = np.full((32, 1), fill_value=-np.inf)
    cta_row_sum = np.zeros((32, 1))
    acc1 = np.zeros((4, 32, 256), dtype=np.float32)  # first dim is for steps
    v = np.reshape(v, (seqLen, headElems))
    save_raw(np.transpose(np.reshape(v, (8, 32, 4, 64)), axes=[0, 2, 1, 3]),
             'v_8x4x32x64_f16.bin')
    for i in range(4):
        cta_row_max_old = cta_row_max
        cta_row_max = np.maximum(cta_row_max, tile_row_max[i])
        xScale = np.exp(tile_row_max[i] - cta_row_max)
        x[i] = x[i] * xScale
        tile_row_sum[i] = tile_row_sum[i] * xScale
        acc1Scale = np.exp(cta_row_max_old - cta_row_max)
        acc1[i] = acc1Scale * (acc1[i - 1] if i > 0 else np.zeros(
            (32, 256), dtype=np.float32))
        cta_row_sum = cta_row_sum * acc1Scale + tile_row_sum[i]
        acc1[i] = acc1[i] + np.mat(x[i]).astype(np.float32) * np.mat(
            v[64 * i:(64 * i + 64), :]).astype(np.float32)

    save_raw(acc1, 'acc1PerStep_4x32x256_f32.bin')
    ref_tiled = (acc1[3] / cta_row_sum * vScale).astype(np.float16)
    save_raw(ref_tiled, 'out_32x256_f16.bin')

    assert is_close(ref_tiled, np.reshape(ref, (32, headElems)))


def compute(q, k, v, kvScale, headElems):
    qkScale = (headElems**-0.5) * kvScale
    qk = q @ k.T * qkScale
    row_max = np.max(qk, axis=1).reshape(-1, 1)
    x = np.exp(qk - row_max)
    row_sum = np.sum(x, axis=1).reshape(-1, 1)
    x @ v * (kvScale / row_sum)
    return x, row_max, row_sum
