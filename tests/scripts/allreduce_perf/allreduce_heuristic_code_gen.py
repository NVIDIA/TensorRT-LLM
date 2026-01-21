import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tensorrt_llm._utils import get_sm_version


@dataclass
class Constants:
    # 16384
    num_tokens_bits = 15
    hidden_size_bits = 14
    max_num_tokens_considered = 2**num_tokens_bits
    max_hidden_size_considered = 2**hidden_size_bits
    oneshot_num_tokens_threshold: int = 1
    oneshot_hidden_size_threshold = 128
    num_tokens_list = [2**i for i in range(num_tokens_bits)]
    hidden_size_list = [2**i for i in range(7, hidden_size_bits)]
    fusion_op_list = [
        'NONE', 'RESIDUAL_RMS_NORM', 'RESIDUAL_RMS_NORM_QUANT_FP8',
        'RESIDUAL_RMS_NORM_QUANT_NVFP4'
    ]
    tp_size_list = [2, 4, 8]
    strategy_name_to_enum = {
        'NCCL': 0,
        'NCCL_SYMMETRIC': 8,
        'ONESHOT': 4,
        'TWOSHOT': 5,
    }


def find_best_strategy(df: pd.DataFrame):
    """Find the best strategy for each combination of parameters."""
    return df.groupby([
        'world_size', 'fusion', 'hidden_size', 'num_tokens'
    ]).apply(lambda group: group.loc[group['time (us)'].idxmin(), 'strategy'])


def filter_df(df: pd.DataFrame):
    df = df[(df['num_tokens'] >= Constants.oneshot_num_tokens_threshold)
            & (df['num_tokens'] <= Constants.max_num_tokens_considered) &
            (df['hidden_size'] >= Constants.oneshot_hidden_size_threshold) &
            (df['hidden_size'] <= Constants.max_hidden_size_considered)]
    return df


def generate_heuristic_look_up_table(df: pd.DataFrame) -> str:
    """
    Generate a heuristic lookup table from benchmark data and output as C++ array.

    Args:
        df: DataFrame with columns: world_size, dtype, size, num_tokens, hidden_size,
            strategy, fusion, time (us)

    Returns:
        String containing C++ array definition for the lookup table
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return ""

    print(f"Input DataFrame shape: {df.shape}")
    print(f"Available strategies: {df['strategy'].unique()}")
    print(f"Available fusions: {df['fusion'].unique()}")
    print(f"Available tp_sizes: {sorted(df['world_size'].unique())}")

    # Filter out AUTO strategy as it's not a concrete implementation
    df_filtered = df[df['strategy'] != 'AUTO'].copy()
    print(f"After filtering AUTO strategy: {df_filtered.shape}")

    # Apply range filters
    df_filtered = filter_df(df_filtered)

    # Find best strategy for each combination
    best_strategies = find_best_strategy(df_filtered)

    # Create lookup table dimensions
    tp_size_count = len(Constants.tp_size_list)
    fusion_count = len(Constants.fusion_op_list)
    hidden_size_count = len(Constants.hidden_size_list)
    num_tokens_count = len(Constants.num_tokens_list)

    # Initialize lookup table with default values (NCCL_SYMMETRIC = 8)
    strategy_table = np.full(
        (tp_size_count, fusion_count, hidden_size_count, num_tokens_count),
        Constants.strategy_name_to_enum['NCCL_SYMMETRIC'],
        dtype=int)

    # Fill the lookup table with best strategies
    filled_entries = 0
    for (tp_size, fusion, hidden_size,
         num_tokens), best_strategy in best_strategies.items():
        try:
            tp_idx = Constants.tp_size_list.index(tp_size)
            fusion_idx = Constants.fusion_op_list.index(fusion)
            hidden_size_idx = Constants.hidden_size_list.index(hidden_size)
            num_tokens_idx = Constants.num_tokens_list.index(num_tokens)

            if best_strategy in Constants.strategy_name_to_enum:
                strategy_value = Constants.strategy_name_to_enum[best_strategy]
                strategy_table[tp_idx, fusion_idx, hidden_size_idx,
                               num_tokens_idx] = strategy_value
                filled_entries += 1
        except ValueError:
            # Skip entries that don't match our defined lists
            continue

    print(f"Filled {filled_entries} entries in the lookup table")

    return strategy_table


def generate_cpp_strategy_lut_code(
    strategy_table: np.ndarray,
    sm_version: int,
) -> str:
    """Generate formatted C++ array code from numpy lookup table."""
    tp_size_count, fusion_count, hidden_size_count, num_tokens_count = strategy_table.shape

    # Header with compact comments
    cpp_code = f"// AllReduce lookup: [tp][fusion][hidden][tokens] = strategy\n"
    cpp_code += f"// TP:{Constants.tp_size_list}\n"
    cpp_code += f"// Fusion:{Constants.fusion_op_list}\n"
    cpp_code += f"// Hidden:{Constants.hidden_size_list}\n"
    cpp_code += f"// Tokens:{Constants.num_tokens_list}\n"
    cpp_code += f"inline AllReduceBestStrategyTableType AllReduceBestStrategyTableSM{sm_version} = {{\n"

    # Generate formatted array notation
    for tp_idx in range(tp_size_count):
        cpp_code += "    {\n"
        cpp_code += f"        // TP={Constants.tp_size_list[tp_idx]}\n"

        for fusion_idx in range(fusion_count):
            cpp_code += f"        {{ // Fusion={Constants.fusion_op_list[fusion_idx]}\n"

            for hidden_idx in range(hidden_size_count):
                cpp_code += "            {"
                # Put all token values on one line
                token_values = []
                for token_idx in range(num_tokens_count):
                    value = strategy_table[tp_idx, fusion_idx, hidden_idx,
                                           token_idx]
                    token_values.append(str(value))
                cpp_code += ",".join(token_values)
                cpp_code += "}"
                if hidden_idx < hidden_size_count - 1:
                    cpp_code += ","
                cpp_code += "\n"

            cpp_code += "        }"
            if fusion_idx < fusion_count - 1:
                cpp_code += ","
            cpp_code += "\n"

        cpp_code += "    }"
        if tp_idx < tp_size_count - 1:
            cpp_code += ","
        cpp_code += "\n"

    cpp_code += "};\n"
    return cpp_code


def main():
    # add args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--sm_version", type=int, default=None)
    parser.add_argument("--save_csv_dir", type=str, default=None)
    parser.add_argument("--enable_auto", action="store_true", default=False)

    args = parser.parse_args()
    tp_size_list = [2, 4, 8]

    # Process the benchmark data
    # combine all the data into one dataframe
    data_dir = args.data_dir
    sm_version = args.sm_version

    if sm_version is None:
        sm_version = get_sm_version()
        print(f"Using SM version: {sm_version}")

    df = pd.DataFrame()

    if data_dir is None:
        if args.save_csv_dir is not None:
            data_dir = args.save_csv_dir
            os.makedirs(data_dir, exist_ok=True)
        else:
            tmpdir = tempfile.TemporaryDirectory()
            data_dir = tmpdir.name
        for tp_size in tp_size_list:
            # use mpi to run all_reduce.py to benchmark the performance if data_dir is not provided
            script_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../microbenchmarks/all_reduce.py")
            save_csv = f"{data_dir}/benchmark.tp{tp_size}.sm{sm_version}.csv"

            print("enable_auto", args.enable_auto)
            cmd = [
                "mpirun",
                "-n",
                str(tp_size),
                "python",
                script_path,
                "--explore_2d",
                "--save_csv",
                save_csv,
            ]
            if args.enable_auto:
                cmd.append("--enable_auto")
            subprocess.run(
                cmd,
                env=os.environ,
            )

    for tp_size in tp_size_list:
        data_file = f"{data_dir}/benchmark.tp{tp_size}.sm{sm_version}.csv"
        if not (Path(data_file)).exists():
            print(f"File {data_file} does not exist")
            return

        df_tp = pd.read_csv(Path(data_file))
        df = pd.concat([df, df_tp])

    assert df.empty == False, "Benchmark data is empty"

    if not os.path.exists(f"{data_dir}/gen_heuristic_code"):
        os.makedirs(f"{data_dir}/gen_heuristic_code")

    if df is not None:
        # Generate the C++ lookup table code
        strategy_table = generate_heuristic_look_up_table(df)
        cpp_code = generate_cpp_strategy_lut_code(strategy_table, sm_version)

        # Write the generated code to a file
        output_file = f"{data_dir}/gen_heuristic_code/generated_lookup_table.cpp"
        with open(output_file, 'w') as f:
            f.write(cpp_code)

        print(f"\nGenerated C++ lookup table, written to: {output_file}")
        print("\nFirst 20 lines of generated code:")
        print(cpp_code)
    else:
        print("Failed to load benchmark data")


if __name__ == "__main__":
    main()
