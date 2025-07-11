How to generate `nvshmem_fast_build.patch`?

1. Build the project without applying the `nvshmem_fast_build.patch`.
2. Link NVSHMEM to DeepEP with one NVSHMEM object file omitted.
3. Repeat step 2 until no more object files can be omitted.
4. Remove the unused files from NVSHMEM's `CMakelists.txt`, and save the differences as `nvshmem_fast_build.patch`.

The script `strip_nvshmem_helper.py` automatically performs steps 2 and 3.
