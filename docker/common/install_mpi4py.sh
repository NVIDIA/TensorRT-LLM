#!/bin/bash

set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

MPI4PY_VERSION="3.1.5"
RELEASE_URL="${GITHUB_URL}/mpi4py/mpi4py/archive/refs/tags/${MPI4PY_VERSION}.tar.gz"

# Create and use a temporary directory
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Download and extract in one step
curl -L ${RELEASE_URL} | tar -zx -C "$TMP_DIR"

# Bypassing compatibility issues with higher versions (>= 69) of setuptools.
sed -i 's/>= 40\.9\.0/>= 40.9.0, < 69/g' "$TMP_DIR/mpi4py-${MPI4PY_VERSION}/pyproject.toml"

# Apply the patch
cd "$TMP_DIR/mpi4py-${MPI4PY_VERSION}"
git apply <<EOF
diff --git a/src/mpi4py/futures/_lib.py b/src/mpi4py/futures/_lib.py
index f14934d1..eebfb8fc 100644
--- a/src/mpi4py/futures/_lib.py
+++ b/src/mpi4py/futures/_lib.py
@@ -278,6 +278,40 @@ def _manager_comm(pool, options, comm, full=True):


 def _manager_split(pool, options, comm, root):
+    if(os.getenv("TRTLLM_USE_MPI_KVCACHE")=="1"):
+        from cuda import cudart
+        has_slurm_rank=False
+        has_ompi_rank=False
+        slurm_rank=0
+        ompi_rank=0
+        if(os.getenv("SLURM_PROCID")):
+            slurm_rank = int(os.environ["SLURM_PROCID"])
+            has_slurm_rank=True
+        elif(os.getenv("OMPI_COMM_WORLD_RANK")):
+            ompi_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
+            has_ompi_rank=True
+        else:
+            raise RuntimeError("No SLURM_PROCID or OMPI_COMM_WORLD_RANK environment variable found When TRTLLM_USE_MPI_KVCACHE is set to 1")
+        if(has_slurm_rank and has_ompi_rank):
+            if(slurm_rank>0 and ompi_rank>0):
+                raise RuntimeError("Only one of SLURM_PROCID or OMPI_COMM_WORLD_RANK should >0 when TRTLLM_USE_MPI_KVCACHE is set to 1")
+            else:
+                rank=slurm_rank if slurm_rank>0 else ompi_rank
+        else:
+            rank = ompi_rank if has_ompi_rank else slurm_rank
+
+        def CUASSERT(cuda_ret):
+            err = cuda_ret[0]
+            if err != cudart.cudaError_t.cudaSuccess:
+                raise RuntimeError(
+                    f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
+                )
+            if len(cuda_ret) > 1:
+                return cuda_ret[1:]
+            return None
+        device_count = CUASSERT(cudart.cudaGetDeviceCount())[0]
+        CUASSERT(cudart.cudaSetDevice(rank%device_count))
+        print(f"rank: {rank},set  device: {CUASSERT(cudart.cudaGetDevice())[0]} in mpi4py _manager_split")
     comm = serialized(comm_split)(comm, root)
     _manager_comm(pool, options, comm, full=False)

EOF

# Install with pip and clean up cache
pip3 install --no-cache-dir "$TMP_DIR/mpi4py-${MPI4PY_VERSION}"

# Clean up
rm -rf "$TMP_DIR"
rm -rf ~/.cache/pip
