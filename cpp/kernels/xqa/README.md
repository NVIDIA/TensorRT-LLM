<div align="left">

# XQA - A set of optimized kernels for generation-phase MQA/GQA

## Dependency

If you want to build & run unit tests, you need libgtest-dev and libeigen3-dev.

## Options

Kernel compile-time options can be found in defines.h. See code comments for details. Runtime options of unit tests can be modified in test.cpp.

## Build & run unit tests

You need to install libgtest-dev and libeigen3-dev before building. To build, use the normal cmake build steps:

- ```mkdir build```
- ```cd build```
- ```cmake .. -DCMAKE_BUILD_TYPE=Release```
- ```cmake --build . -j```

To run unit tests, run `./unitTests`. There are a few runtime options that can be controlled with environment variables:

- XQA_ZERO_FILL: Set this to 1 to initialize input data with zeros (instead of random numbers). This is useful if you want to run perf tests quickly and skip the slow random data generation step. Note there is an impact on measure perf.
- XQA_USE_QGMMA: On Hopper, we try to use TMA+QGMMA kernel (mha_sm90.cu) by default if possible. To force using mha.cu, set this to 0.
- XQA_NB_SUB_SEQ: The number of CUDA thread blocks used to handle one K/V head. We have reasonable default but if you want to change it manually, use this variable.

## Generation cubins used in TensorRT-LLM

Run `gen_cubin.py` in the repo workspace.
