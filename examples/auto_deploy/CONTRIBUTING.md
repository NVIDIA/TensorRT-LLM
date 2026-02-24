# Contributing to AutoDeploy

## Setting Up Developer Environment

### 0. Clone the repo

Clone the TensorRT LLM repo and `cd` into it:

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
```

### 1. Create virtual env

We recommend setting up a virtual environment via `conda` to simplify dependency management of
non-pip dependencies like `openmpi`. Follow the instructions [here](https://docs.anaconda.com/miniconda/install/) to install `miniconda` if you have not set up `conda` yet.

Then you can set up a virtual environment:

```bash
AUTO_ENV=auto
conda create -y -n $AUTO_ENV pip python=3.12
conda activate $AUTO_ENV
```

### 2. Setup mpi

Next, you can install mpi-related dependencies using conda's package manager:

```bash
conda install -y -c conda-forge mpi4py openmpi
```

Since `openmpi` gets installed in a non-standard location make sure to set up the following
environment variables when running/importing the `tensorrt_llm` library.

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMPI_MCA_opal_cuda_support=true
```

You can add those lines to your `.bashrc` or `.zshrc` as well.

### 3. Install `tensorrt_llm`

We will use pre-compiled wheels from the latest CI builds. You can inspect available builds from the
[PyPI artifactory](https://pypi.nvidia.com/tensorrt-llm/)
and set `TRTLLM_PRECOMPILED_LOCATION` according to the desired wheel URL.

*Note: We suggest using the latest pre-built wheel for compatibility and recommend repeating these steps regularly to keep the pre-built portion up-to-date.*

For example, on a Linux x86 system running Python 3.12, the most recent pre-compiled wheel is available at:

```bash
export TRTLLM_PRECOMPILED_LOCATION=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.18.0.dev2025021800-cp312-cp312-linux_x86_64.whl
```

Now can you proceed to do an editable pip install of `tensorrt_llm`:

```bash
pip install -e ".[devel]"
```

### 4. Verify install

You can try running the quickstart demo script to verify your installation. Don't forget to
correctly set the mpi-related environment variables in your shell.

```bash
cd examples/auto_deploy
python build_and_run_ad.py --config '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}'
```

## Committing code

### Linting and Pre-commit hooks

TensorRT LLM uses pre-commit hooks to lint code.

#### Set up pre-commit hooks

To setup the pre-commit hooks run

```
pip install pre-commit
pre-commit install
```

#### Use pre-commit hooks

Pre-commit hooks are run during every commit to ensure the diff is correctly linted. If you want to
disable linting for a commit you can use the `-n` flag:

```bash
git commit -n ...
```

To run all linters on the whole repo use

```bash
pre-commit run --all-files
```

If you want to run linters on the diff between `$HEAD` and `main` run

```bash
pre-commit run --from-ref origin/main --to-ref HEAD
```

### VSCode Integration

We provide recommended workspace settings for VSCode at [\`examples/auto_deploy/.vscode](.vscode). Feel free to adopt and use them if developing with VSCode.

## Testing and Debugging

### Run tests

We use `pytest` to run a suite of standardized tests. You can invoke the test suite via `pytest` and pointing to the desired subfolder.

To invoke the full test suite, run

```bash
pytest tests/_torch/autodeploy

```

### Debugging

Our example script is compatible with off-the-shelf python debuggers. We recommend setting `world_size=0` to avoid spawning subprocesses and simplify debugging.
