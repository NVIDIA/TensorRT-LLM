# Helix Parallelism

For more details, see the following paper describing Helix Parallelism:
[Helix Parallelism link TODO](https://todo).

Helix parallelism is a type of context / KV cache parallelism.
Unlike most other types of context parallelism (e.g. star attention or ring attention),
Helix is used during the decode/generation phase.

Helix parallelism is most useful in scenarios where all of the following
conditions apply:

1. Disaggregated serving: Helix parallelism will be applied to the generation
   server(s) only.
2. High input sequence length / context size: depending on the model, Helix
   likely only provides performance advantages with input sequence lengths >64K,
   possibly more.
3. Low batch sizes: Helix is most useful in low-latency / high tokens/s/user
   scenarios. On the typical Pareto curve, these are found at the highest point
   on the x-axis / towards the right of the plot.

## Testing / benchmarking Helix with TensorRT-LLM

There are currently three main ways of testing and benchmarking Helix parallelism
in TensorRT-LLM, as described below.

### Benchmarking end-to-end

End-to-end benchmarking should be done using a SLURM cluster, with disaggregated
serving.

The scripts to help with this can be found in the
[slurm/benchmark](../../../../examples/disaggregated/slurm/benchmark) folder,
found under the disaggregated examples folder.

TODO: note the following section should be updated with more generic scripts
(not just for OCI HSG cluster).

The main entry script for benchmarking is
[submit_mjoux.sh](../../../../examples/disaggregated/slurm/benchmark/submit_mjoux.sh).

This script requires the right container image to run, which can be obtained in
two ways:

1. Set the `repo_dir` variable in the script as expected, as well as `build_wheel=true`.
   This will build the right image before starting the remaining job.
2. Edit the install job in
   [disaggr_torch.slurm](../../../../examples/disaggregated/slurm/benchmark/disaggr_torch.slurm)
   as follows (the `srun` command after `echo "Installing TensorRT-LLM..."`):

   - Update the `srun` command to use the `--container-save=<output_path.sqsh>` argument.
   - Use `pip install .` instead of `pip install -e .` in the bash command.
   - Submit the job using [submit_mjoux.sh](../../../../examples/disaggregated/slurm/benchmark/submit_mjoux.sh).
   - Point the `container_image` variable to `<output_path.sqsh>` as provided above.

Finally, the following variables in
[submit_mjoux.sh](../../../../examples/disaggregated/slurm/benchmark/submit_mjoux.sh)
can be used to control the parameters of the benchmark:

- `gen_tp_size`: TP size used for generation servers
- `gen_pp_size`: PP size used for generation servers
- `gen_cp_size`: CP (or KVP) size used for generation servers
- `gen_ep_size`: EP size used for generation servers.
- `batch`: The batch size
- `isl`: the input sequence length
- `osl`: the output sequence length
- `model_dir`: the path to the (TensorRT-LLM) model weights used for the benchmark.
   Note that this was only tested with DeepSeek-style models.

The total number of GPUs required for the generation servers is at least
`gen_tp_size * gen_pp_size * gen_cp_size`.  
Note that for the FFN part of the module, all CP/KVP GPUs are automatically
re-purposed to TP or EP GPUs. Thus, for dense FFN modules, the effective TP size is
`gen_tp_size * gen_cp_size`. For Mixture-of-Experts FFN modules, the effective
TP size is `(gen_tp_size * gen_cp_size) / gen_ep_size`.

The benchmark outputs statistics about the generation given the input parameters,
including throughput and latency numbers, such as time per output token (TPOT).

### Benchmarking a single layer

Benchmarking a single layer using MLA can be done using the
[test_helix_deepseek.py](../../../../tests/unittest/_torch/modeling/test_helix_deepseek.py) script.

As the name implies, this benchmark is limited to DeepSeek-style layers, as
DeepSeek is the only model type supported by TensorRT-LLM using MLA,
as of time of writing this documentation.

The script should be launched using `python` (note that it is not part of the
standard test suite and should not be called through `pytest`).

The parameters for the benchmark can be controlled through the following
command-line arguments:

- `--type`: model type (DeepSeek V3/R1 or DeepSeek-V3-Lite)
- `--dense`: if set, uses a dense layer for FFN, instead of Mixture-of-Experts (default)
- `--ctx_len_start` / `--ctx_len_end`: range of context length to benchmark (increased in multiples of 2)
- `--batch`: batch size
- `--tp`: TP size
- `--kvp`: KVP / CP size
- `--ep`: EP size

Note that for this test, the required number of GPUs is given by the multiplication
of the value for `--tp` and `--kvp`: KVP GPUs are automatically re-purposed to
TP GPUs for the attention output, and they are re-purposed to TP or EP for FFN.

This benchmark outputs the total time taken as well as the expected Time per output
token (TPOT) for a full DeepSeek-V3 model, if all layers were of the given type.

### Correctness test for the MLA module

The simplest test can be found in
[test_mla_helix.py](../../../../tests/unittest/_torch/modules/test_mla_helix.py).

This is a unit test of just the
[MLA attention module](../../../../tensorrt_llm/_torch/modules/attention.py),
which has been updated to support Helix parallelism.

This unit test can be launched using `pytest` and is part of the TensorRT-LLM test suite.

### End-to-end integration tests

End-to-end integration tests are run in disaggregated mode with cache transmission
using `pytest`. These tests validate the complete Helix parallelism pipeline:

- `tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_deepseek_v3_lite_bf16_tllm_gen_helix`
- `tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_deepseek_v3_lite_fp8_tllm_gen_helix`
- `tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_deepseek_r1_fp4_tllm_gen_helix`
