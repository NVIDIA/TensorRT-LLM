# CPU Affinity configuration in TensorRT-LLM

## NUMA-aware affinity in TensorRT-LLM

TensorRT-LLM is frequently deployed on
[NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access) systems. In
order to ensure consistent and optimal performance on these systems, it is
critical to set the CPU affinity of the workers/tasks launched as part of a
particular TRT-LLM instance so as to minimize latency and maximize bandwidth of
CPU&harr;GPU and CPU&harr;DRAM communication.

Because TensorRT-LLM does the work of allocating GPU/CUDA devices to ranks, it
is logically the ideal place for the CPU affinity to be determined and set. For
this reason, TensorRT-LLM provides a mechanism to automatically set CPU
affinity according to NUMA topology. In some situations/deployments, the user
may wish to configure CPU affinity manually (i.e. using
[numactl](https://github.com/numactl/numactl), [wrappers around the
same](https://github.com/NVIDIA/mlperf-common/blob/main/client/bindpcie), or
mpirun). For this reason, this feature is only activated if it is explicitly
enabled or if CPU affinity is not already constrained by the user or
environment. It is controlled by the TLLM_NUMA_AWARE_WORKER_AFFINITY
environment variable as follows:


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

```bash
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_inherit=1
```

The first environment variable ensures that OpenMPI will not attempt to bind or
set the affinity of the ranks that are created at launch.

The second ensures that OpenMPI's binding policy will propagate to MPI workers
that are spawned by `mpi4py`'s `MPIPoolExecutor` class within TensorRT-LLM
(when using mpirun).

### Slurm

If Slurm is configured to use a affinity or cgroup task plugin, then Slurm may
also configure CPU affinity by default in a way that is not sensitized to NUMA
topology. To prevent this, Slurm jobs should be launched accordingly:

#### srun

The srun parameters should include `--cpu-bind=none` and exclude `--exclusive`:

```bash
srun --cpu-bind=none ...
```

#### sbatch

The sbatch script should set `SLURM_CPU_BIND` environment variable to "none":

```bash
export SLURM_CPU_BIND=none
```

Note: if this environment variable is set, it is not necessary to supply the
`--cpu-bind=none` to each job step (srun invocation)

## CPU affinity configuration examples

### Using NUMA-aware autoconfiguration

To explicitly enable the NUMA-aware autoconfiguration feature in TensorRT-LLM,
simply set `TLLM_NUMA_AWARE_WORKER_AFFINITY` in the launch script (prior to
`trtllm-bench` or `trtllm-serve`) as follows:

```bash
export TLLM_NUMA_AWARE_WORKER_AFFINITY=1
```

Because autoconfiguration happens within TensorRT-LLM itself, it will override
any CPU affinity or binding that has been previously set by OpenMPI or Slurm.

### NUMA-aware CPU affinity using [bindpcie](https://github.com/NVIDIA/mlperf-common/blob/main/client/bindpcie)

The bindpcie script is designed to set a per-rank CPU affinity that is ideal
for NUMA topology. While setting `TLLM_NUMA_AWARE_WORKER_AFFINITY=1` usually
achieves the same result in terms of the CPU affinity that is set, this
approach has the distinct advantage that the optimal CPU affinity gets set
_upon launching_ TensorRT-LLM, guaranteeing that each worker/rank executes on
the optimal NUMA node from inception. The NUMA-aware CPU affinity
autoconfiguration mechanism in TensorRT-LLM, on the other hand, is triggered by
each worker/rank upon its own PID _after_ it has already launched. If the
worker/rank executes on a NUMA node other than the optimal NUMA node at some
point between the launch of the process and the NUMA-aware autoconfiguration,
it is possible that some CPU memory may have been allocated/touched on what
will become a remote NUMA node after the point of autoconfiguration,
potentially negatively impacting performance. In practice, this effect has been
observed to have minimal performance impact, but some degradation of
performance due to remote NUMA node access is still theoretically possible.

The `bindpcie` script can only be applied to deployments that make use of
`trtllm-llmapi-launch` within an sbatch script. One example of how to apply
bindpcie to `trtllm-serve` in an sbatch script is as follows:

```bash
# Prevent TensorRT-LLM from autoconfiguring or clearing CPU affinity
export TLLM_NUMA_AWARE_WORKER_AFFINITY=0

# Prevent OpenMPI from overriding affinity set by bindpcie
export OMPI_MCA_hwloc_base_binding_policy=none

# Ensure that MPI binding policy propagates to any MPI workers dynamically
# spawned by MPIPoolExecutor
export OMPI_MCA_rmaps_base_inherit=1

# Prevent Slurm from assigning a default CPU affinity
export SLURM_CPU_BIND=none

srun -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST} \
    --container-workdir=${WORKDIR} \
    --export=ALL,PYTHONPATH=${SOURCE_ROOT} \
    --mpi=pmix \
    bash -c "
        set -ex
        $PROLOGUE
        export PATH=$PATH:~/.local/bin

        bindpcie trtllm-llmapi-launch \
         trtllm-serve $LOCAL_MODEL \
            ${ADDITIONAL_OPTIONS}
```

> [!NOTE]
> This is not a complete or exhaustive example of an sbatch script to launch
> trtllm-serve and is only intended to highlight the application of bindpcie
> within an existing sbatch script.

### Using [numactl](https://github.com/numactl/numactl)

```bash
# Prevent TensorRT-LLM from autoconfiguring or clearing CPU affinity
export TLLM_NUMA_AWARE_WORKER_AFFINITY=0

# Prevent OpenMPI from overriding affinity set by numactl
export OMPI_MCA_hwloc_base_binding_policy=none

# Ensure that MPI binding policy propagates to any MPI workers dynamically
# spawned by MPIPoolExecutor
export OMPI_MCA_rmaps_base_inherit=1

# Use numactl to specify CPU and memory binding for all ranks (not per-rank)
numactl --physcpubind=0,1,16,17 --membind=0 mpirun --report-bindings --oversubscribe --allow-run-as-root \
<trtllm-serve | trtllm-bench> <arguments>
```

### Using mpirun

If a manually-specified per-rank CPU affinity is desired when running on a
single node with mpirun, this can be achieved most easily using an OpenMPI
rankfile. The following is an example of how a rankfile can be used to
arbitrarily map each of 4 MPI ranks to a distinct set of 4 cores:

```bash
# Prevent TensorRT-LLM from autoconfiguring or clearing CPU affinity
export TLLM_NUMA_AWARE_WORKER_AFFINITY=0

# Not strictly needed here, since we are overriding with explicit bindings from
# a rankfile
# export OMPI_MCA_hwloc_base_binding_policy=none

# Ensure that MPI binding policy propagates to any MPI workers dynamically
# spawned by MPIPoolExecutor
export OMPI_MCA_rmaps_base_inherit=1

# Create a rankfile to enumerate a set of 4 cores to which each rank is bound
cat > ./rankfile <<EOF
rank 0=localhost slot=0,1,2,3
rank 1=localhost slot=4,5,6,7
rank 2=localhost slot=8,9,10,11
rank 3=localhost slot=12,13,14,15
EOF

# Run with (and report to verify) the bindings from the rankfile
mpirun --rankfile ./rankfile -n 1 --report-bindings --oversubscribe --allow-run-as-root \
<trtllm-serve | trtllm-bench> <arguments>
```

See the official [OpenMPI Documentation](https://www.open-mpi.org/doc/) for
more details on mapping and binding of MPI ranks.
