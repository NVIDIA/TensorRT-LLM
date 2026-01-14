import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from tensorrt_llm._utils import get_sm_version


def visualize_2d_heatmap(df, fusion_op='NONE', save_path=None):
    """Visualize the allreduce dataframe as a 2D heatmap using seaborn.

        Args:
        df: DataFrame with columns: world_size, dtype, size, strategy, fusion, version, time (us)
        save_path: Optional path to save the plot. If None, displays the plot.

    Creates a 2D heatmap where:
    - x-axis: num_tokens
    - y-axis: hidden_size
    - colors: time (us) values using seaborn heatmap
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return

    fusion_col = 'fusion' if 'fusion' in df.columns else 'fusion_op'
    df_filtered = df[df[fusion_col] == fusion_op].copy()

    if df_filtered.empty:
        print(f"No data found for fusion == '{fusion_op}'")
        return

    # Determine column names (adapt to different data formats)
    num_tokens_col = 'num_tokens'
    time_col = 'time (us)' if 'time (us)' in df_filtered.columns else 'time_ms'

    # Get unique strategies
    strategies = df_filtered['strategy'].unique()

    # Set seaborn style for better aesthetics
    sns.set_style("whitegrid")
    plt.style.use('default')  # Reset to default style

    # Create subplots for each strategy
    n_strategies = len(strategies)
    fig, axes = plt.subplots(1, n_strategies, figsize=(8 * n_strategies, 6))

    # Handle single strategy case
    if n_strategies == 1:
        axes = [axes]

    fig.suptitle(f'AllReduce Performance Heatmaps (Fusion: {fusion_op})',
                 fontsize=16,
                 fontweight='bold',
                 y=1.02)

    # Calculate global min and max for consistent colorbar across all strategies
    global_time_min = df_filtered[time_col].min()
    global_time_max = df_filtered[time_col].max()

    for i, strategy in enumerate(strategies):
        ax = axes[i]
        strategy_data = df_filtered[df_filtered['strategy'] == strategy]

        if strategy_data.empty:
            ax.set_title(f'{strategy} - No Data')
            continue

        # Create pivot table for heatmap
        # Use num_tokens as index (y-axis) and hidden_size as columns (x-axis)
        pivot_data = strategy_data.pivot_table(
            index='hidden_size',
            columns='num_tokens',
            values=time_col,
            aggfunc='mean'  # Use mean if there are duplicate entries
        )

        # Sort indices and columns for better visualization
        pivot_data = pivot_data.sort_index(
            ascending=False)  # Larger hidden_size at top
        pivot_data = pivot_data.reindex(sorted(pivot_data.columns),
                                        axis=1)  # Sort columns

        # Create seaborn heatmap with linear scaling
        sns.heatmap(
            pivot_data,
            ax=ax,
            cmap='Spectral_r',
            cbar=False,  # We'll add a shared colorbar later
            fmt='.1f',
            square=True,  # Make cells square instead of rectangular
            linewidths=0.5,
            linecolor='white',
            vmin=global_time_min,
            vmax=global_time_max)

        # Customize labels and title
        ax.set_xlabel(f'{num_tokens_col.title()}',
                      fontweight='bold',
                      fontsize=12)
        ax.set_ylabel('Hidden Size', fontweight='bold', fontsize=12)
        ax.set_title(f'{strategy}', fontweight='bold', fontsize=14)

        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

        # Format tick labels to remove decimal points for integers
        x_labels = [
            f'{int(float(label.get_text()))}' for label in ax.get_xticklabels()
        ]
        y_labels = [
            f'{int(float(label.get_text()))}' for label in ax.get_yticklabels()
        ]
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

    # Add a single shared colorbar for all subplots
    fig.subplots_adjust(right=0.87)  # Make room for colorbar

    # Create a dummy plot for colorbar with linear scaling
    sm = plt.cm.ScalarMappable(cmap='Spectral_r',
                               norm=plt.Normalize(vmin=global_time_min,
                                                  vmax=global_time_max))
    sm.set_array([])

    cbar_ax = fig.add_axes([0.89, 0.15, 0.01,
                            0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Time (μs)', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(right=0.87)  # Ensure colorbar space is maintained

    # Save or show the plot
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D heatmap saved to: {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

    # Print some statistics
    print(f"\n2D Heatmap Statistics:")
    print(f"Strategies: {list(strategies)}")
    print(
        f"{num_tokens_col.title()} values: {sorted(df_filtered['num_tokens'].unique())}"
    )
    print(f"Hidden sizes: {sorted(df_filtered['hidden_size'].unique())}")
    print(
        f"Global time range (colorbar): {global_time_min:.4f} - {global_time_max:.4f} μs"
    )
    print(f"Total data points: {len(df_filtered)}")


def visualize_2d_best_strategy(df, fusion_op='NONE', save_path=None):
    """Visualize the best strategy for each mesh grid position as a 2D heatmap.

        Args:
        df: DataFrame with columns: world_size, dtype, size, strategy, fusion, version, time (us)
        save_path: Optional path to save the plot. If None, displays the plot.

    Creates a heat map where:
    - x-axis: num_tokens
    - y-axis: hidden_size
    - colors: best strategy for each (num_tokens, hidden_size) combination
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return

    fusion_col = 'fusion' if 'fusion' in df.columns else 'fusion_op'
    df_filtered = df[df[fusion_col] == fusion_op].copy()

    # Filter out AUTO strategy
    df_filtered = df_filtered[df_filtered['strategy'] != 'AUTO'].copy()

    if df_filtered.empty:
        print(
            f"No data found for fusion == '{fusion_op}' after filtering out AUTO strategy"
        )
        return

    # Determine column names (adapt to different data formats)
    num_tokens_col = 'num_tokens'
    time_col = 'time (us)' if 'time (us)' in df_filtered.columns else 'time_ms'

    # Find the best strategy for each (num_tokens, hidden_size) combination
    best_strategy_data = []

    for (num_tokens, hidden_size), group in df_filtered.groupby(
        ['num_tokens', 'hidden_size']):
        # Find the strategy with minimum time for this combination
        best_row = group.loc[group[time_col].idxmin()]
        best_strategy_data.append({
            'num_tokens': num_tokens,
            'hidden_size': hidden_size,
            'best_strategy': best_row['strategy'],
            'best_time': best_row[time_col]
        })

    best_df = pd.DataFrame(best_strategy_data)

    # Get unique strategies and create a color mapping
    strategies = sorted(df_filtered['strategy'].unique())

    # Create a categorical color map with distinct colors
    strategy_colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    strategy_to_num = {strategy: i for i, strategy in enumerate(strategies)}

    # Create pivot table for heatmap with strategy numbers
    pivot_data = best_df.pivot_table(
        index='hidden_size',
        columns='num_tokens',
        values='best_strategy',
        aggfunc='first'  # Should be unique anyway
    )

    # Convert strategy names to numbers for heatmap
    pivot_numeric = pivot_data.applymap(lambda x: strategy_to_num[x]
                                        if pd.notna(x) else np.nan)

    # Sort indices and columns for better visualization
    pivot_numeric = pivot_numeric.sort_index(
        ascending=False)  # Larger hidden_size at top
    pivot_numeric = pivot_numeric.reindex(sorted(pivot_numeric.columns),
                                          axis=1)  # Sort columns

    # Set seaborn style for better aesthetics
    sns.set_style("whitegrid")
    plt.style.use('default')  # Reset to default style

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create custom colormap
    cmap = ListedColormap(strategy_colors)

    # Create seaborn heatmap with categorical data
    sns.heatmap(
        pivot_numeric,
        ax=ax,
        cmap=cmap,
        cbar=False,  # We'll create a custom colorbar
        fmt='',
        square=True,  # Make cells square instead of rectangular
        linewidths=1,
        linecolor='white',
        vmin=-0.5,
        vmax=len(strategies) - 0.5)

    # Create custom colorbar with strategy labels
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                              norm=plt.Normalize(
                                                  vmin=-0.5,
                                                  vmax=len(strategies) - 0.5)),
                        ax=ax,
                        ticks=range(len(strategies)))
    cbar.ax.set_yticklabels(strategies)
    cbar.set_label('Best Strategy', fontweight='bold', fontsize=12)

    # Customize labels and title
    ax.set_xlabel(f'{num_tokens_col.title()}', fontweight='bold', fontsize=14)
    ax.set_ylabel('Hidden Size', fontweight='bold', fontsize=14)
    ax.set_title(
        f'Best AllReduce Strategy for Each Mesh Grid Position\n(Fusion: {fusion_op})',
        fontweight='bold',
        fontsize=16,
        pad=20)

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    # Format tick labels to remove decimal points for integers
    x_labels = [
        f'{int(float(label.get_text()))}' for label in ax.get_xticklabels()
    ]
    y_labels = [
        f'{int(float(label.get_text()))}' for label in ax.get_yticklabels()
    ]
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Best strategy heatmap saved to: {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

    # Print some statistics
    print(f"\nBest Strategy Heatmap Statistics:")
    print(f"Strategies found: {strategies}")
    print(
        f"{num_tokens_col.title()} values: {sorted(best_df['num_tokens'].unique())}"
    )
    print(f"Hidden sizes: {sorted(best_df['hidden_size'].unique())}")
    print(f"Total grid positions: {len(best_df)}")

    # Show strategy distribution
    strategy_counts = best_df['best_strategy'].value_counts()
    print(f"\nStrategy distribution:")
    for strategy, count in strategy_counts.items():
        percentage = (count / len(best_df)) * 100
        print(f"  {strategy}: {count} positions ({percentage:.1f}%)")

    return best_df


def visualize_strategy_difference_heatmaps(df,
                                           fusion_op='NONE',
                                           save_path=None):
    """Generate 2D heatmaps showing the difference between each strategy and the best strategy.

        Args:
        df: DataFrame with columns: world_size, dtype, size, strategy, fusion, version, time (us)
        save_path: Optional file path to save the plot. If None, displays the plots.

    Creates difference heatmaps where:
    - x-axis: num_tokens
    - y-axis: hidden_size
    - colors: time difference (current_strategy_time - best_time) in μs
    - Generates one heatmap per strategy
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return

    fusion_col = 'fusion' if 'fusion' in df.columns else 'fusion_op'
    df_filtered = df[df[fusion_col] == fusion_op].copy()

    if df_filtered.empty:
        print(f"No data found for fusion == '{fusion_op}'")
        return

    # Determine column names
    num_tokens_col = 'num_tokens'
    time_col = 'time (us)' if 'time (us)' in df_filtered.columns else 'time_ms'

    # Find the best strategy and time for each (num_tokens, hidden_size) combination
    best_times = {}
    for (num_tokens, hidden_size), group in df_filtered.groupby(
        ['num_tokens', 'hidden_size']):
        best_row = group.loc[group[time_col].idxmin()]
        best_times[(num_tokens, hidden_size)] = best_row[time_col]

    # Get unique strategies
    strategies = sorted(df_filtered['strategy'].unique())

    # Set seaborn style for better aesthetics
    sns.set_style("whitegrid")
    plt.style.use('default')

    # Calculate difference data for all strategies
    all_diff_data = []
    max_diff = 0
    min_diff = 0

    for strategy in strategies:
        strategy_data = df_filtered[df_filtered['strategy'] == strategy]

        if strategy_data.empty:
            continue

        # Calculate differences for this strategy
        diff_data = []
        for _, row in strategy_data.iterrows():
            num_tokens = row['num_tokens']
            hidden_size = row['hidden_size']
            current_time = row[time_col]

            best_time = best_times.get((num_tokens, hidden_size))
            if best_time is not None:
                diff = (current_time - best_time) / best_time * 100
                diff_data.append({
                    'num_tokens': num_tokens,
                    'hidden_size': hidden_size,
                    'diff (%)': diff,
                    'strategy': strategy
                })
                max_diff = max(max_diff, diff)
                min_diff = min(min_diff, diff)

        all_diff_data.extend(diff_data)

    # Ensure we have data to visualize
    if not all_diff_data:
        print("No valid data found for difference calculation")
        return

    # Check if we have multiple strategies for meaningful comparison
    if len(strategies) == 1:
        print(
            f"Warning: Only one strategy ({strategies[0]}) found. All differences will be zero."
        )

    print(
        f"Generating difference heatmaps for {len(strategies)} strategies with {len(all_diff_data)} data points..."
    )

    # Create subplots for each strategy
    n_strategies = len(strategies)
    fig, axes = plt.subplots(1, n_strategies, figsize=(8 * n_strategies, 6))

    # Handle single strategy case
    if n_strategies == 1:
        axes = [axes]

    fig.suptitle(
        f'Strategy Performance Difference from Best Strategy (Fusion: {fusion_op})',
        fontsize=16,
        fontweight='bold',
        y=1.02)

    # Use a colormap where 0 (best strategy) appears white
    cmap = 'Reds'  # White at 0 -> Red for worse performance

    for i, strategy in enumerate(strategies):
        ax = axes[i]

        # Filter data for this strategy
        strategy_diff_data = [
            d for d in all_diff_data if d['strategy'] == strategy
        ]

        if not strategy_diff_data:
            ax.set_title(f'{strategy} - No Data')
            continue

        # Convert to DataFrame and create pivot table
        strategy_df = pd.DataFrame(strategy_diff_data)
        pivot_data = strategy_df.pivot_table(index='hidden_size',
                                             columns='num_tokens',
                                             values='diff (%)',
                                             aggfunc='mean')

        # Sort indices and columns for better visualization
        pivot_data = pivot_data.sort_index(
            ascending=False)  # Larger hidden_size at top
        pivot_data = pivot_data.reindex(sorted(pivot_data.columns),
                                        axis=1)  # Sort columns

        # Determine annotation font size based on grid size
        total_cells = pivot_data.shape[0] * pivot_data.shape[1]
        annot_fontsize = 8 if total_cells <= 25 else 6 if total_cells <= 64 else 4

        # Create seaborn heatmap with 0 values as white
        sns.heatmap(
            pivot_data,
            ax=ax,
            cmap=cmap,
            cbar=False,  # We'll add a shared colorbar later
            fmt='.1f',
            square=True,
            linewidths=0.5,
            linecolor='white',
            vmin=0,  # Force 0 (best strategy) to map to white
            vmax=max(
                max_diff,
                0.1),  # Ensure we have at least some range for visualization
            annot=True,  # Show values in cells
            annot_kws={
                'fontsize': annot_fontsize,
                'weight': 'bold'
            })

        # Customize labels and title
        ax.set_xlabel(f'{num_tokens_col.title()}',
                      fontweight='bold',
                      fontsize=12)
        ax.set_ylabel('Hidden Size', fontweight='bold', fontsize=12)
        ax.set_title(f'{strategy}\n(Difference from Best %)',
                     fontweight='bold',
                     fontsize=12)

        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

        # Format tick labels to remove decimal points for integers
        x_labels = [
            f'{int(float(label.get_text()))}' for label in ax.get_xticklabels()
        ]
        y_labels = [
            f'{int(float(label.get_text()))}' for label in ax.get_yticklabels()
        ]
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

    # Add a single shared colorbar for all subplots
    fig.subplots_adjust(right=0.87)  # Make room for colorbar

    # Create a dummy plot for colorbar with 0 values as white
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=max(max_diff,
                                            0.1))  # Force 0 to map to white
    )
    sm.set_array([])

    cbar_ax = fig.add_axes([0.89, 0.15, 0.01,
                            0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Time Difference (%)\n', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(right=0.87)  # Ensure colorbar space is maintained

    # Save or show the plot
    if save_path:
        # Create parent directory if it doesn't exist
        save_file_path = Path(save_path)
        if not save_file_path.suffix:
            # If no extension provided, add .png
            save_file_path = save_file_path.with_suffix('.png')
        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
        print(f"Strategy difference heatmaps saved to: {save_file_path}")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

    # Print statistics
    print(f"\nStrategy Difference Heatmap Statistics:")
    print(f"Strategies analyzed: {strategies}")
    print(f"Difference range: {min_diff:.2f} to {max_diff:.2f} μs")
    print(f"Note: Positive values indicate slower than best strategy")

    # Show per-strategy statistics
    for strategy in strategies:
        strategy_diffs = [
            d['diff (%)'] for d in all_diff_data if d['strategy'] == strategy
        ]
        if strategy_diffs:
            avg_diff = np.mean(strategy_diffs)
            max_diff_strategy = max(strategy_diffs)
            print(
                f"  {strategy}: avg diff = {avg_diff:.2f} μs, max diff = {max_diff_strategy:.2f} μs"
            )


def main():
    # add args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    fusion_op_list = [
        "NONE",
        "RESIDUAL_RMS_NORM",
        "RESIDUAL_RMS_NORM_QUANT_FP8",
        "RESIDUAL_RMS_NORM_QUANT_NVFP4",
    ]
    tp_size_list = [2, 4, 8]

    if args.data_dir is None:
        print("Please provide a data directory")
        return

    if not os.path.exists(os.path.join(args.data_dir, "viz")):
        os.makedirs(os.path.join(args.data_dir, "viz"))

    for tp_size in tp_size_list:
        case_name = f"benchmark.tp{tp_size}.sm{get_sm_version()}"
        fname = os.path.join(args.data_dir, case_name + ".csv")
        if not (Path(fname)).exists():
            print(f"File {fname} does not exist")
            continue

        df = pd.read_csv(Path(fname))
        for fusion_op in fusion_op_list:
            # if not exists, create the directory
            if not os.path.exists(os.path.join(args.data_dir, "viz",
                                               fusion_op)):
                os.makedirs(os.path.join(args.data_dir, "viz", fusion_op))

            if df is not None:
                print(f"\n=== TP Size: {tp_size} ===")
                print(df.head())
                print(f"Data shape: {df.shape}")

                # Create 2D heatmap visualization and save to data/viz directory
                viz_path_heatmap = f"{args.data_dir}/viz/{fusion_op}/{case_name}_heatmap.png"
                visualize_2d_heatmap(df, fusion_op, save_path=viz_path_heatmap)

                # Create best strategy heatmap visualization and save to data/viz directory
                viz_path_best_strategy = f"{args.data_dir}/viz/{fusion_op}/{case_name}_best_strategy.png"
                visualize_2d_best_strategy(df,
                                           fusion_op,
                                           save_path=viz_path_best_strategy)

                # Create strategy difference heatmaps and save to data/viz directory
                viz_path_diff = f"{args.data_dir}/viz/{fusion_op}/{case_name}_strategy_difference_heatmap.png"
                visualize_strategy_difference_heatmaps(df,
                                                       fusion_op,
                                                       save_path=viz_path_diff)


if __name__ == "__main__":
    main()
