#!/bin/bash
# set these variables according to your cluster setup

export CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/containers/tllm_pyt2508_py3_aarch64_trt10.13.2.6_202509112230_7568.sqsh"   # /path/to/image.sqsh
#export CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/containers/tllm_pyt2508_py3_aarch64_trt10.13.2.6_202509112230_7568_buildbb9db33344d24319c559f12459f20c242a5d1356.sqsh"
export CONTAINER_MOUNTS="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations:/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations"  # /path:/path
export WORK_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/$USER/TensorRT-LLM/examples/disaggregated/slurm/benchmark/"          # Path to this directory
export MODEL_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/data/models/DeepSeek-R1/DeepSeek-R1-FP4/"         # Path to the model checkpoint
export REPO_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/$USER/TensorRT-LLM"          # Path to the repo to install TensorRT-LLM, if this is empty, the pre-installed version will be used
export DATA_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/$USER/data"          # Path to the data directory
export SLURM_PARTITION="batch"   # slurm partition
export SLURM_ACCOUNT="coreai_horizon_dilations"     # slurm account
export SLURM_JOB_NAME="helix_benchmark_gen_only"    # slurm job name
#unset REPO_DIR
