# CPU Affinity configuration in TensorRT-LLM

## NUMA-aware affinity in TensorRT-LLM

TensorRT-LLM is frequently deployed on
[NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access) systems. In
order to ensure consistent and optimal performance on these systems, it is
critical to set the CPU affinity of the workers/tasks launched as part of a
particular TRT-LLM instance so as to minimize latency and maximize bandwidth of
CPU&harr;GPU and CPU&harr;DRAM communication.

Because TensorRT-LLM does the work of allocating GPU/CUDA devices to ranks, it
is the ideal place for the CPU affinity to be determined and set. For this
reason, TensorRT-LLM provides a mechanism to automatically set CPU affinity
according to NUMA topology. In some niche situations/deployments, the user may
wish to configure CPU affinity manually (i.e. using numactl or [wrappers around
the same](https://github.com/NVIDIA/mlperf-common/blob/main/client/bindpcie)).
For this reason, this feature is only activated if it is explicitly enabled or
if CPU affinity is not already constrained by the user or environment. It is
controlled by the TLLM_NUMA_AWARE_WORKER_AFFINITY environment variable as
follows:


| TLLM_NUMA_AWARE_WORKER_AFFINITY | Behavior                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| <unset>                         | Affinity is auto-configured if it is unconstrained, and cleared if it is constrained by the user and/or environment          |
| 1                               | Affinity is unconditionally auto-configured.                                                                                 |
| 0 or any other value            | Affinity remains as configured by the user and/or environment                                                                |


## Other environmental considerations

Whether or not the user chooses to manually configure CPU affinity or have
TensorRT-LLM configure it automatically, the environment can also constrain the
CPU affinity in a way that subverts the user's intent. Both OpenMPI and Slurm
may configure CPU affinity, so the following additional configuration is
recommended to avoid this.

### OpenMPI

By default, OpenMPI chooses a rank-wise CPU affinity that is not sensitized to
the NUMA-topology of the system. Because it does not know which GPU a
particular rank will be communicating with (this is determined by TRT-LLM at
runtime), it cannot set the CPU affinity accordingly. For this reason, it is
recommended that OpenMPI's default binding policy be disabled as follows:

```
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_inherit=1
```

The first environment variable ensures that OpenMPI will not attempt to bind or
set the affinity of the ranks that are created at launch. The second ensures
that this policy will propagate to MPI workers that are dynamically created
when using mpirun.

### Slurm

If Slurm is configured to use a affinity or cgroup task plugin, then Slurm may
also configure CPU affinity by default in a way that is not sensitized to NUMA
topology. To prevent this, Slurm jobs should be launched accordingly:

#### srun

The srun parameters should include `--cpu-bind=none` and exclude `--exclusive`:

```
srun --cpu-bind=none ...
```

#### sbatch

The sbatch script should set `SLURM_CPU_BIND` environment variable to "none":

```
export SLURM_CPU_BIND=none
```

Note: if this environment variable is set, it is not necessary to supply the
`--cpu-bind=none` to each job step (srun invocation)
