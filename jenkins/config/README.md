# Test stage resources

`stage_resources.json` configures the CPU and memory requests and limits of the `trt-llm`
container in Kubernetes test stages. Each top-level key is an exact stage name.
CBTS can run a narrowed stage as `<stage>-cbts`; that stage uses the entry for
`<stage>` unless the file also has an entry for `<stage>-cbts`.
Values are positive integers: `cpu_m` is CPU millicores, and `memory_mi` is
memory in MiB. Requests and limits use the same values.

Stages without an entry keep their existing requests and limits.
SLURM dispatcher pods and jnlp containers are not supported in `stage_resources.json`.

## Before you raise a value

Requests and limits use the same value, so Kubernetes reserves the full amount on
the node while the stage runs. Check these points:

- The value covers only the `trt-llm` container. The Pod also requests 2 CPUs and
  10 GiB for the JNLP container. ModelExpress stages also run `redis` (1 CPU,
  4 GiB) and `model-express-server` (4 CPUs, 8 GiB). The total must fit on one
  node of the stage's GPU type.
- Stay within the stage's share of the node. Several stages can run on the same
  multi-GPU node. By default, a single-GPU stage gets 12 CPUs and 96 GiB, and a
  multi-GPU stage gets 12 CPUs and 150 GiB per GPU. A stage that asks for more
  than its share can leave other GPUs on the node idle, because no other Pod
  fits next to it.
- DGX Spark (`gb10x`) has 128 GB of memory that the CPU and the GPU share. The
  default gives 64 GiB to the CPU side and leaves the rest to the GPU. A larger
  value leaves less memory for the GPU.
