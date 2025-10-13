# Layer-wise Benchmarks

## Generate profiles

### Run with MPI

Step 1: Start a container using Docker, Enroot or others. Please refer to `../../jenkins/current_image_tags.properties` for the Docker image URI.

Step 2: In the container, Install `tensorrt_llm`:

```bash
pip install -e ../..
```

Step 3: In the container, run benchmarks and generate profiles:

```bash
NP=4 ./mpi_launch.sh ./run_single.sh --test-case GEN
```

### Run with Slurm

Step 1: On the controller node, alloc one or multiple nodes, and record the `SLURM_JOB_ID`:

```bash
SLURM_JOB_ID=$(NODES=2 ./slurm_alloc.sh)
```

Please fill the variables in `./slurm_alloc.sh`.

Step 2: Start a container and install `tensorrt_llm`. Run the following command on the controller node:

```bash
SLURM_JOB_ID=$SLURM_JOB_ID ./slurm_init_containers.sh
```

Step 3: Run benchmarks to generate profiles, where `NODES` &le; the number of allocated nodes. Run the following command on the controller node:

```bash
SLURM_JOB_ID=$SLURM_JOB_ID NODES=2 NP=8 ./slurm_launch.sh ./run_single.sh --test-case GEN
```

## Parse profiles

Coming soon.
