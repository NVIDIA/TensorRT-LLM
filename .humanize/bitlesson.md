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
Solution: In any validation/benchmark harness, `import deep_gemm` before invoking the op (as bench_trtllm_deepgemm_three_way.py does), or run with the build's `.venv-3.12` python where deep_gemm is importable. (Potential code follow-up: have `init_deep_gemm_runtime_once` also call `IncludeParser::prepare_init` so the C++ path is self-sufficient.)
Constraints: Observed on SM90/GH200 with vendored DeepGEMM 2.5.0 and the rc17 release container (which does not ship an importable `deep_gemm` at system python).
Validation Evidence: round0_new_ops_sanity.py failed with the include_parser error until import was addressed; the trtllm-forced op (no DeepGEMM JIT) passed in the same harness.
Source Rounds: 0
