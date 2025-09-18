We have recently open-sourced a set of Cutlass kernels that were previously known as "internal_cutlass_kernels". Due to internal dependencies, these kernels were previously only available to users as static libraries. We have now decoupled these internal dependencies, making the kernels available as source code.

The open-sourced Cutlass kernels are on the path `cpp/tensorrt_llm/kernels/cutlass_kernels`, including:
- `low_latency_gemm`
- `moe_gemm`
- `fp4_gemm`
- `allreduce_gemm`

To ensure stability and provide an optimized performance experience, we have maintained the previous method of calling these kernels via static libraries as an alternative option. You can switch between open-sourced Cutlass kernels and static library Cutlass kernels through the `USING_OSS_CUTLASS_*` macro (where * represents the specific kernel name), enabling kernel-level control. By default, the open-source Cutlass kernels are used.
Note that support for these static libraries will be gradually deprioritized in the future and may eventually be deprecated.

**Default Configuration (Using open-sourced Cutlass Kernels)**

To build using the open-source Cutlass kernels (default setting), run:

```bash
python3 ./scripts/build_wheel.py --cuda_architectures "90-real;100-real"
```

**Using Static Library Cutlass Kernels**

If you prefer to use the Cutlass kernels from the static library, you can control this during compilation by setting the `USING_OSS_CUTLASS_*` macro to `OFF`. For example, to use the static library implementation specifically for `low_latency_gemm` and `moe_gemm` while keeping other kernels as OSS, use the following compilation command:

```bash
python3 ./scripts/build_wheel.py --cuda_architectures "90-real;100-real" -D "USING_OSS_CUTLASS_MOE_GEMM=OFF;USING_OSS_CUTLASS_LOW_LATENCY_GEMM=OFF"
```
