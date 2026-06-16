# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

<!-- Add lessons below using the strict template. -->

## Lesson: deepgemm-jit-includeparser-init
Lesson ID: BL-20260615-deepgemm-jit-includeparser-init
Scope: cpp/tensorrt_llm/thop/fp8BlockScalingGemm.cpp (DeepGEMM direct path); any standalone harness that loads libth_common.so
Problem Description: Calling the C++ DeepGEMM direct path (e.g. torch.ops.trtllm.fp8_block_scaling_gemm_deep_gemm, or fp8_block_scaling_gemm_impl with env=direct_deep_gemm) from a minimal harness that loads libth_common.so but does NOT `import deep_gemm` fails at JIT time with: Assertion error include_parser.hpp: "Failed to open: deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh".
Root Cause: DeepGEMM's JIT has a separate `IncludeParser` whose kernel-source include root is set by `IncludeParser::prepare_init(root)` → root/include. The thop `init_deep_gemm_runtime_once` initializes `Compiler::prepare_init` and `KernelRuntime::prepare_init` but NOT `IncludeParser::prepare_init`; the IncludeParser is instead initialized as a side effect of `import deep_gemm` (python). Without that import, the include root is unset and the parser cannot open kernel .cuh files.
Solution: Add `deep_gemm::IncludeParser::prepare_init(deep_gemm_root)` to `init_deep_gemm_runtime_once` (mirror DeepGEMM `csrc/apis/runtime.hpp`, which inits Compiler+KernelRuntime+IncludeParser) and `#include "jit/include_parser.hpp"`. IMPORTANT: `import deep_gemm` does NOT fix this — it initializes a SEPARATE C++ runtime's static IncludeParser (the standalone deep_gemm_cpp), not the DeepGEMM compiled into libth_common.so. The C++ path must self-init.
Constraints: Observed on SM90/GH200 with vendored DeepGEMM 2.5.0 and the rc17 release container.
Validation Evidence: round0_new_ops_sanity.py failed with "Failed to open: deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh" until the prepare_init call was added; after the fix the deep_gemm-forced op matched the env `direct_deep_gemm` path exactly.
Source Rounds: 0

## Lesson: deepgemm-jit-root-needs-cutlass
Lesson ID: BL-20260615-deepgemm-jit-root-needs-cutlass
Scope: DeepGEMM C++ JIT compilation invoked from the thop (TRTLLM_DEEP_GEMM_ROOT); validation/benchmark harnesses
Problem Description: After the IncludeParser fix, the DeepGEMM JIT got past header lookup but NVCC failed: "cutlass/arch/barrier.h: No such file or directory", because the DeepGEMM kernel includes CUTLASS headers.
Root Cause: DeepGEMM's Compiler builds nvcc include flags as only `-I{library_include_path} -I{cuda_home}/include` (compiler.hpp). It relies on `library_include_path` (= deep_gemm_root/include) ALSO containing `cutlass/`. The raw `DeepGEMM/deep_gemm/include` does NOT bundle cutlass; only the PACKAGED deep_gemm (build output / installed wheel) copies cutlass under `include/`.
Solution: Set `TRTLLM_DEEP_GEMM_ROOT` to a packaged deep_gemm whose `include/` bundles cutlass — e.g. the build venv `/code/TensorRT-LLM/.venv-3.12/lib/python3.12/site-packages/deep_gemm`, or `DeepGEMM/build/lib.linux-aarch64-cpython-312/deep_gemm`, or the C++ build's packaged `tensorrt_llm/deep_gemm`. Do NOT use raw `DeepGEMM/deep_gemm`.
Constraints: SM90/GH200, DeepGEMM 2.5.0, aarch64 build. The cutlass-bearing include dir must match the build arch.
Validation Evidence: sanity PASS only after switching TRTLLM_DEEP_GEMM_ROOT from `DeepGEMM/deep_gemm` (include/{deep_gemm,-}) to the venv site-packages deep_gemm (include/{deep_gemm,cutlass}).
Source Rounds: 0
