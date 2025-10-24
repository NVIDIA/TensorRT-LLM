# Layer-wise Benchmarks

## Generate profiles

### Run with MPI

**Step 1:** Start a container using Docker, Enroot or others. Please refer to `../../jenkins/current_image_tags.properties` for the Docker image URI.

**Step 2:** In the container, install `tensorrt_llm`:

```bash
pip install -e ../..
```

**Step 3:** In the container, run benchmarks and generate profiles:

```bash
NP=4 ./mpi_launch.sh ./run_single.sh DeepSeek-R1.yaml --test-case GEN
```

### Run with Slurm

> Tips: If you have a running job with environment installed, please skip step 1 and 2 and go straight to step 3. In this case, your job must be run with `--container-name aaa`, and if the container name is not "layer_wise_benchmarks" please `export CONTAINER_NAME=aaa`.

**Step 1:** On the controller node, allocate one or multiple nodes, and record the `SLURM_JOB_ID`:

```bash
SLURM_JOB_ID=$(NODES=2 TIME=01:00:00 ./slurm_alloc.sh)
```

Please fill the variables in `./slurm_alloc.sh`.

**Step 2:** Start a container and install `tensorrt_llm`. Run the following command on the controller node:

```bash
SLURM_JOB_ID=$SLURM_JOB_ID ./slurm_init_containers.sh
```

It uses the image recorded in `../../jenkins/current_image_tags.properties`. The image will be downloaded to `../../enroot/` for once.

**Step 3:** Run benchmarks to generate profiles. Run the following command on the controller node, where `NODES` &le; the number of allocated nodes:

```bash
SLURM_JOB_ID=$SLURM_JOB_ID NODES=2 NP=8 ./slurm_launch.sh ./run_single.sh DeepSeek-R1.yaml --test-case GEN
```

## Parse profiles

Coming soon.
