# AllReduce Performance Offline Autotuning and Visualization Tools

This directory contains tools for benchmarking, analyzing, and visualizing AllReduce performance in TensorRT-LLM. The toolkit consists of two main components:

1. **`allreduce_heuristic_code_gen.py`** - Generates optimized C++ lookup tables for AllReduce strategy selection
2. **`allreduce_perf_viz.py`** - Creates comprehensive visualizations of AllReduce performance data

## Overview

The AllReduce performance analysis workflow involves:
1. **Benchmarking**: Run performance tests across different configurations
2. **Analysis**: Generate optimal strategy lookup tables
3. **Visualization**: Create heatmaps and performance comparison charts

## Prerequisites

- TensorRT-LLM environment with MPI support
- Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`
- CUDA-capable GPU(s) for benchmarking

## Tool 1: allreduce_heuristic_code_gen.py

### Purpose
Generates C++ lookup tables that contain the optimal AllReduce strategy for different parameter combinations (tensor parallel size, fusion operations, hidden sizes, and token counts).

### Usage

#### Basic Usage (Auto-benchmark and generate)
```bash
python allreduce_heuristic_code_gen.py
```

#### Advanced Usage
```bash
python allreduce_heuristic_code_gen.py \
    --data_dir /path/to/benchmark/data \
    --sm_version 89 \
    --save_csv_dir /path/to/save/csv \
    --enable_auto
```

### Parameters

- `--data_dir`: Directory containing existing benchmark CSV files (optional)
- `--sm_version`: CUDA SM version (auto-detected if not specified)
- `--save_csv_dir`: Directory to save benchmark CSV files
- `--enable_auto`: Enable AUTO strategy in benchmarking

### Workflow

1. **Benchmark Generation** (if `--data_dir` not provided):
   - Automatically runs `all_reduce.py` microbenchmark using MPI
   - Tests multiple tensor parallel sizes (currently TP=2)
   - Generates CSV files: `benchmark.tp{size}.sm{version}.csv`

2. **Strategy Analysis**:
   - Loads benchmark data from CSV files
   - Filters data based on predefined thresholds
   - Finds optimal strategy for each parameter combination
   - Creates 4D lookup table: `[tp_size][fusion][hidden_size][num_tokens]`

3. **Code Generation**:
   - Converts lookup table to C++ array format
   - Outputs to `gen_heuristic_code/generated_lookup_table.cpp`
   - Ready for integration into TensorRT-LLM codebase

### Output Example
```cpp
// AllReduce lookup: [tp][fusion][hidden][tokens] = strategy
// TP:[2, 4, 8] Fusion:['NONE', 'RESIDUAL_RMS_NORM', ...]
inline AllReduceBestStrategyTableType AllReduceBestStrategyTableSM89 = {
    {
        // TP=2
        { // Fusion=NONE
            {0,0,4,4,5,5,5,5,5,5,5,5,5,5,5}, // hidden_size=128
            {0,4,4,4,5,5,5,5,5,5,5,5,5,5,5}, // hidden_size=256
            // ... more rows
        },
        // ... more fusion types
    },
    // ... more TP sizes
};
```

## Tool 2: allreduce_perf_viz.py

### Purpose
Creates comprehensive visualizations of AllReduce performance data including performance heatmaps, strategy comparison charts, and difference analysis.

### Usage

#### Basic Usage
```bash
python allreduce_perf_viz.py --data_dir /path/to/benchmark/data
```

### Parameters

- `--data_dir`: Directory containing benchmark CSV files (default: 'data')

### Generated Visualizations

The tool generates three types of visualizations for each configuration:

#### 1. Performance Heatmaps (`*_heatmap.png`)
- **Purpose**: Show raw performance times for each strategy
- **Layout**: Side-by-side heatmaps for each AllReduce strategy
- **Axes**: X=num_tokens, Y=hidden_size
- **Colors**: Performance time in microseconds (μs)
- **Features**: Shared colorbar, logarithmic scaling for better visualization

#### 2. Best Strategy Maps (`*_best_strategy.png`)
- **Purpose**: Show optimal strategy for each parameter combination
- **Layout**: Single heatmap with categorical colors
- **Axes**: X=num_tokens, Y=hidden_size
- **Colors**: Different strategies (NCCL, ONESHOT, TWOSHOT, etc.)
- **Features**: Custom colorbar with strategy labels, distribution statistics

#### 3. Strategy Difference Heatmaps (`*_strategy_difference_heatmap.png`)
- **Purpose**: Show performance difference from optimal strategy
- **Layout**: Side-by-side heatmaps for each strategy
- **Axes**: X=num_tokens, Y=hidden_size
- **Colors**: Percentage difference from best strategy (white=optimal, red=slower)
- **Features**: Annotated cells with exact difference values

### Visualization Functions

The script provides three main visualization functions that can be used programmatically:

```python
# 1. Performance heatmaps
visualize_2d_heatmap(df, fusion_op='NONE', save_path='heatmap.png')

# 2. Best strategy visualization
visualize_2d_best_strategy(df, fusion_op='NONE', save_path='best_strategy.png')

# 3. Strategy difference analysis
visualize_strategy_difference_heatmaps(df, fusion_op='NONE', save_path='diff.png')
```

### Output Structure
```
data/
├── viz/
│   ├── NONE/
│   │   ├── benchmark.tp2.sm89_heatmap.png
│   │   ├── benchmark.tp2.sm89_best_strategy.png
│   │   └── benchmark.tp2.sm89_strategy_difference_heatmap.png
│   ├── RESIDUAL_RMS_NORM/
│   │   └── ... (similar files)
│   └── ... (other fusion operations)
└── benchmark.tp2.sm89.csv
```

## Configuration Details

### Supported Strategies
- **NCCL** (0): Standard NCCL AllReduce
- **ONESHOT** (4): Custom single-phase AllReduce
- **TWOSHOT** (5): Custom two-phase AllReduce

### Supported Fusion Operations
- `NONE`: No fusion
- `RESIDUAL_RMS_NORM`: Residual + RMS normalization
- `RESIDUAL_RMS_NORM_QUANT_FP8`: RESIDUAL_RMS_NORM + FP8 quantization
- `RESIDUAL_RMS_NORM_QUANT_NVFP4`: RESIDUAL_RMS_NORM + NVFP4 quantization

### Parameter Ranges
- **Tensor Parallel Sizes**: 2, 4, 8
- **Hidden Sizes**: 128 to 8192 (powers of 2)
- **Token Counts**: 1 to 16384 (powers of 2)

## Performance Tips

- Run benchmarks on target hardware for accurate results
- Use multiple runs and average results for stability
- Consider different fusion operations based on your use case
- Monitor GPU memory usage during benchmarking

## Integration with TensorRT-LLM

The generated lookup tables can be integrated into TensorRT-LLM's AllReduce implementation to automatically select optimal strategies based on runtime parameters. The C++ arrays follow the format expected by the TensorRT-LLM AllReduce subsystem.

## Contributing

When adding new strategies or fusion operations:

1. **Update Configuration**: Modify the `Constants` class in `allreduce_heuristic_code_gen.py`
2. **Add Strategy Mapping**: Update `strategy_name_to_enum` dictionary with new strategy entries
3. **Generate New Lookup Tables**: Run `allreduce_heuristic_code_gen.py` to create updated lookup tables for optimal AllReduce strategies
4. **Integrate into Codebase**: Copy the generated C++ array into the appropriate lookup table in `cpp/tensorrt_llm/common/customAllReduceUtils.h`
5. **Update Visualizations**: Modify color schemes in `allreduce_perf_viz.py` if needed for new strategies
6. **Validate**: Test with representative workloads to ensure performance improvements
