# FMHA_v2

## Introduction

FMHA_v2 is just a bunch of Multi-head Attention kernels that we've enabled for known cases. It's not built as a library (cuBLAS, cuDNN, HazyResearch's MHA, etc) that is supposed to deliver good perf for all cases. End users will get access to FMHA through products or libraries, not directly through FMHA_v2.

## Launch a container to build the code

We recommend that you use a container to build and run the code. For example:
```
docker run -it --rm --gpus all --user `id -u`:`id -g` -v $PWD:/workspace nvcr.io/nvidia/pytorch:24.05-py3 /bin/bash
```

## Build the code

Some kernels are disabled by default (for example, SM70 codes, HMMA/HGMMA FP32 accumulation codes, etc.)
To enable them you have to export the following environment variables before calling
the `setup.py` code:
```
export TORCH_CUDA_ARCH_LIST=9.0 ENABLE_SM89_QMMA=1 ENABLE_HMMA_FP32=1 SCHEDULING_MODE=1 ENABLE_SM100=1 ENABLE_SM120=1
```

To generate subset of kernels, you can add conditions in setup.py.

To generate the files and compile the kernels:
```
python3 setup.py && make -j
```

`ccache` allows caching previous compilations to speed up recompilation. To leverage `ccache`:
```
apt install ccache
export USE_CCACHE=1
python3 setup.py && make -j
```

## Running tests

### Command-line arguments

The fmha executable has been compiled as `bin/fmha.exe`. All the detailed parameters related to the attention kernels can be specified by providing the appropriate [command-line arguments](src/fused_multihead_attention.cpp#L679) when running
the executable.

For example,

```bash

# run causal-mask multi-head attention kernels with
# batch_size (b) = 4, num_heads (h) = 32, head_size (d) = 128, sequence_length (s) = 1024, data_type = fp16.
# verbose (v) = false
bin/fmha.exe -d 128 -b 4 -h 32 -s 1024 -min-s 1024 -fp16 -runs 10 -warm-up-runs 100 -causal-mask -v 0

# console output
v1=0 il=0 s_q=1024, s=1024 b=4 h=32/32 d=128 dtype=FP16, flash_attn=true, warp_spec=true, mask=causal, alibi=false, attn=mha, paged_kv=false, wm=4 wn=1
Checks........: SUCCESS
Elapsed ......: 1172.329590 us (57.49x), 468.94 Tflop/s, 457.95 GB/s
```

### Pytest

FMHA_v2 uses pytest to aggregate test results from GTest-based unit tests and prompt-based perf
test that doubles as all-purpose test.

To install dependencies:

```bash
make deps
```

To run test:

```bash
pytest fmha_test.py
```

## Frequently Asked Questions

Why is the FMHA_v2 slower than public implementation in several cases?

```
Usually, adding new launch configurations suffices. The heuristics of FMHA_v2 are designed to work optimally for known cases. If you encounter an unknown case, first check if FMHA_v2 has a suitable kernel. If there isn't one, feel free to approach us and we'll enable a new configuration
```

What's the difference between cubins and cu files?

'''
Cubins are precompiled (from the internal fmha_v2 repo) binary files and take a lot of space, cu files are generated directly from this repo. Now we replace most of the kernels with cu files and delete unused cubins.
You can modify code in this repo to change or create your own kernels and run.
Now there are some kernels still running in cubins. See use_cubin_header(setup.py#L3055) and modify_cubin_header(setup.py#L3413) for details.
'''
