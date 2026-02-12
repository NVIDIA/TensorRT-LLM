import os
import sys
import time

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.suffix_automaton import is_native_available
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, NGramDecodingConfig


def _check_native_kernel_thorough():
    """Check native suffix automaton kernel availability more thoroughly.

    Beyond just checking the import, this verifies the binding actually exposes
    the required functions.  This catches the case where a stale local
    bindings.cpython-*.so (from an older CMake build) shadows the installed
    one that was built *with* suffix_automaton support.
    """
    if not is_native_available():
        return False
    try:
        from tensorrt_llm.bindings.internal import suffix_automaton as _sa
        # Smoke-test a cheap function to make sure the binding is real
        _sa.get_state_size(1024)
        return True
    except Exception:
        return False


def benchmark_ngram(use_cuda_graph: bool, num_runs: int = 10):
    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[1, 2, 4, 8, 16]) if use_cuda_graph else None

    spec_config = NGramDecodingConfig(max_draft_len=4, max_matching_ngram_size=3)
    llm = LLM(
        model="meta-llama/Llama-3.1-8B",
        speculative_config=spec_config,
        cuda_graph_config=cuda_graph_config,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                      max_tokens=8192),
        max_batch_size=16,
        max_num_tokens=2048,
    )

    prompts = ["Explain quantum computing"] * 10

    # Warmup
    llm.generate(prompts[:1], SamplingParams(max_tokens=50))

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        llm.generate(prompts, SamplingParams(max_tokens=100))
    elapsed = time.perf_counter() - start

    llm.shutdown()
    return elapsed / num_runs


if __name__ == '__main__':
    if not _check_native_kernel_thorough():
        # Diagnose the problem – look for a local bindings .so in the source tree
        import glob as _glob
        _so_pattern = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tensorrt_llm",
            "bindings.cpython-*.so",
        )
        _local_sos = _glob.glob(_so_pattern)
        local_so = _local_sos[0] if _local_sos else None
        msg = (
            "Native suffix automaton kernel is not available.\n"
            "The NGram benchmark requires the native C++/CUDA kernel.\n\n"
        )
        if local_so:
            msg += (
                "NOTE: A local bindings .so was found at:\n"
                f"  {local_so}\n"
                "This may be a stale build artifact (compiled before suffix_automaton\n"
                "was added to the C++ source). Rebuild from the current source:\n"
                "  python scripts/build_wheel.py --cuda_architectures=<arch> "
                "--fast_build --use_ccache -j --trt_root /usr/local/tensorrt\n"
                "  pip install build/*.whl --no-deps --force-reinstall\n"
                "Then copy the updated .so into the source tree:\n"
                f"  cp /usr/local/lib/python3.12/dist-packages/tensorrt_llm/"
                f"bindings.cpython-*.so {os.path.dirname(local_so)}/\n"
            )
        else:
            msg += (
                "Please rebuild TensorRT-LLM with suffix_automaton support:\n"
                "  python scripts/build_wheel.py --cuda_architectures=<arch> "
                "--fast_build --use_ccache -j --trt_root /usr/local/tensorrt\n"
                "  pip install build/*.whl --no-deps --force-reinstall\n"
            )
        print(msg, file=sys.stderr)
        raise SystemExit(1)

    # Compare
    try:
        time_with_graph = benchmark_ngram(use_cuda_graph=True)
    except RuntimeError as e:
        if "Native suffix automaton kernel" in str(e) or "Executor worker" in str(e):
            print(
                "ERROR: The native suffix automaton kernel is available in the "
                "main process but not in the executor worker process.\n"
                "This usually means a stale local bindings .so is shadowing "
                "the installed package.\n"
                "Try: pip install build/*.whl --no-deps --force-reinstall\n"
                "Or remove tensorrt_llm/bindings.cpython-*.so from the source tree.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        raise

    time_without_graph = benchmark_ngram(use_cuda_graph=False)

    print(f"With CUDA graph: {time_with_graph:.3f}s")
    print(f"Without CUDA graph: {time_without_graph:.3f}s")
    print(f"Speedup: {time_without_graph / time_with_graph:.2f}x")
