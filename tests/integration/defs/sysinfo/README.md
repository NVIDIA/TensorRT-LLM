# How use this folder

## Overview

This folder is designed to retrieve Mako information for the current node. To execute the script, run the following command:

```bash
python get_sysinfo.py --test-prefix ${stageName} --mako-opt stage=post_merge
```

## Functionality

The script generates information about the following system properties:
- **Operating System**: Includes OS version, OS code name, hostname, and memory.
- **CPU**: Provides details about the CPU and system name.
- **GPU**: Includes GPU count, compute capability, GPU type, chip model, GPU memory, CUDA version, cuBLAS version, and more.
- **Other Properties**: Includes support for FP8/int8/tf32 and indicates whether the system is Android.
All the information got will output to console

## Command Line Arguments

- `--mako-opt`: This argument allows you to add Mako options. Currently, stage=post_merge is used in the test database to differentiate between cases run during merge requests and those run post-merge.
Please note that more arguments may be added as needed.

## Notes

In addition to compute capability and chip details, we utilize the `nvidia-ml-py` package to gather most information. We use `cuda-python` to retrieve compute capability, and we read from a gpu_chip_mapping.json file, which comes from an internal package called `nrsu`, to obtain chip properties.
