How to generate `nvshmem_fast_build.patch`?

1. Build without `nvshmem_fast_build.patch`.
2. Try linking DeepEP to NVSHMEM while omitting one object file.
3. Repeat step 2 until no more object files can be omitted.
4. Remove the unused files from NVSHMEM's `CMakelists.txt`, and save the differences as `nvshmem_fast_build.patch`.

The script `strip_nvshmem_helper.py` automatically performs steps 2 and 3.
