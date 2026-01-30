import re
import subprocess
import sys

import numpy as np


def lazy_convert_sqlite(nsys_rep_file_path, sqlite_file_path):
    if (
        not sqlite_file_path.is_file()
        or nsys_rep_file_path.stat().st_mtime > sqlite_file_path.stat().st_mtime
    ):
        subprocess.check_call(
            [
                "nsys",
                "export",
                "--type",
                "sqlite",
                "-o",
                sqlite_file_path,
                "--force-overwrite=true",
                nsys_rep_file_path,
            ]
        )


parser_keywords = [
    ("cuBLASGemm", "nvjet"),
    ("cutlassGroupGemm", "cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal"),
    ("cutlassGemm", "GemmUniversal"),
    ("CuteDSLMoePermute", "cute_dsl::moePermuteKernel"),
    (
        "CuteDSLGemm",
        ["cute_dsl_kernels", "blockscaled_gemm_persistent"],
    ),
    (
        "CuteDSLGroupedGemmSwiglu",
        ["cute_dsl_kernels", "blockscaled_contiguous_grouped_gemm_swiglu_fusion"],
    ),
    (
        "CuteDSLGroupedGemmFinalize",
        ["cute_dsl_kernels", "blockscaled_contiguous_grouped_gemm_finalize_fusion"],
    ),
    ("torchAdd", "at::native::CUDAFunctorOnSelf_add"),
    ("torchAdd", "CUDAFunctor_add"),
    ("torchClamp", "at::native::<unnamed>::launch_clamp_scalar("),
    ("torchCompare", "at::native::<unnamed>::CompareFunctor<"),
    ("torchCopy", "at::native::bfloat16_copy_kernel_cuda"),
    ("torchCopy", "at::native::direct_copy_kernel_cuda("),
    ("torchDiv", "at::native::binary_internal::DivFunctor<"),
    ("torchFill", "at::native::FillFunctor"),
    ("torchIndexPut", "at::native::index_put_kernel_impl<"),
    ("torchMul", "at::native::binary_internal::MulFunctor<"),
    ("torchPow", "at::native::<unnamed>::pow_tensor_scalar_kernel_impl<"),
    ("torchReduceSum", ["at::native::reduce_kernel<", "at::native::sum_functor<"]),
    ("torchScatterGather", "void at::native::_scatter_gather_elementwise_kernel<"),
    ("torchSigmoid", "at::native::sigmoid_kernel_cuda"),
    ("torchWhere", "at::native::<unnamed>::where_kernel_impl("),
]
warned_names = set()


def kernel_short_name(name):
    for dst, src in parser_keywords:
        if not isinstance(src, (tuple, list)):
            src = [src]
        if all(keyword in name for keyword in src):
            return dst
    if re.search(r"at::native::.*elementwise_kernel<", name):
        if name not in warned_names:
            print(f"Not parsed torch kernel name: {name}", file=sys.stderr)
            warned_names.add(name)
    assert "!unnamed!" not in name
    name = name.replace("<unnamed>", "!unnamed!")
    if "<" in name:
        name = name[: name.index("<")]
    if "(" in name:
        name = name[: name.index("(")]
    if "::" in name:
        name = name[name.rindex("::") + 2 :]
    name = name.replace("!unnamed!", "<unnamed>")
    return name


def shortest_common_supersequence(a, b):
    # Merge two lists into their shortest common supersequence,
    # so that both `a` and `b` are subsequences of the result.
    # Uses dynamic programming to compute the shortest common supersequence, then reconstructs it.
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    # Backtrack to build the merged sequence
    res = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            res.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] < dp[i][j - 1]:
            res.append(a[i - 1])
            i -= 1
        else:
            res.append(b[j - 1])
            j -= 1
    while i > 0:
        res.append(a[i - 1])
        i -= 1
    while j > 0:
        res.append(b[j - 1])
        j -= 1
    res.reverse()
    return res


try:
    import numba

    numba_installed = True
except ImportError:
    numba_installed = False

if numba_installed:
    # The core computation function: compiled to machine code by Numba.
    # 'nopython=True' ensures it runs entirely without the Python interpreter for max speed.
    @numba.jit(nopython=True)
    def _core_scs(a_ids, b_ids):
        m = len(a_ids)
        n = len(b_ids)

        # Use a NumPy array instead of a Python list of lists.
        # This creates a continuous memory block, similar to int dp[m+1][n+1] in C.
        dp = np.zeros((m + 1, n + 1), dtype=np.int32)

        # 1. Initialize boundaries
        # Corresponds to: dp[i][0] = i
        for i in range(m + 1):
            dp[i, 0] = i
        # Corresponds to: dp[0][j] = j
        for j in range(n + 1):
            dp[0, j] = j

        # 2. Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a_ids[i - 1] == b_ids[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    val1 = dp[i - 1, j] + 1
                    val2 = dp[i, j - 1] + 1
                    if val1 < val2:
                        dp[i, j] = val1
                    else:
                        dp[i, j] = val2

        # 3. Backtrack to reconstruct the result
        # dp[m, n] holds the total length of the shortest common supersequence.
        res_len = dp[m, n]

        # Pre-allocate the result array.
        # Filling a pre-allocated array is much faster than appending to a list.
        res_ids = np.empty(res_len, dtype=np.int32)
        k = res_len - 1  # Index for writing into res_ids

        i, j = m, n
        while i > 0 and j > 0:
            if a_ids[i - 1] == b_ids[j - 1]:
                res_ids[k] = a_ids[i - 1]
                i -= 1
                j -= 1
            elif dp[i - 1, j] < dp[i, j - 1]:
                res_ids[k] = a_ids[i - 1]
                i -= 1
            else:
                res_ids[k] = b_ids[j - 1]
                j -= 1
            k -= 1

        while i > 0:
            res_ids[k] = a_ids[i - 1]
            i -= 1
            k -= 1

        while j > 0:
            res_ids[k] = b_ids[j - 1]
            j -= 1
            k -= 1

        return res_ids

    def shortest_common_supersequence(a, b):
        # 1. Build a mapping table (String -> Int)
        # Extract unique tokens from both lists
        unique_tokens = list(set(a) | set(b))
        token_to_id = {token: i for i, token in enumerate(unique_tokens)}
        id_to_token = {i: token for i, token in enumerate(unique_tokens)}

        # 2. Convert input lists to NumPy integer arrays
        a_ids = np.array([token_to_id[x] for x in a], dtype=np.int32)
        b_ids = np.array([token_to_id[x] for x in b], dtype=np.int32)

        # 3. Call the JIT-compiled core function
        # The first time this runs, it will compile (takes ~200ms). Subsequent runs are instant.
        res_ids = _core_scs(a_ids, b_ids)

        # 4. Convert the result back to strings (Int -> String)
        return [id_to_token[idx] for idx in res_ids]
