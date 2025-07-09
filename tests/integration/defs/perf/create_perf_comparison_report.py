import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

TEST_NAME = 'test_name'
METRIC_VALUE = 'perf_metric'
METRIC_TYPE = 'metric_type'
MEAN_COL = 'mean'


def shorten_names(merged: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    name_mapping = {
        k: f'configuration_{i+1}'
        for i, k in enumerate(set(config for config in merged[TEST_NAME]))
    }
    merged[TEST_NAME] = merged[TEST_NAME].apply(lambda name: name_mapping[name])
    return merged, name_mapping


def write_name_mapping_table(name_mapping: dict[str, str],
                             pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(max(len(n)
                                        for n in name_mapping.keys()) * 0.3,
                                    len(name_mapping) *
                                    0.4))  # height depends on number of entries
    ax.axis('off')

    table_data = [["Original Name", "Short Name"]]
    for original, short in name_mapping.items():
        table_data.append([original, short])

    plt.title("Long name to short name mapping")
    table = ax.table(cellText=table_data, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    pdf.savefig(fig)
    plt.close(fig)


def plot_metric(merged: pd.DataFrame, metric: str,
                suffixes: set[str]) -> Figure:
    metric_data = merged[merged[METRIC_TYPE] == metric]
    relevant_metrics = {
        MEAN_COL: metric_data[MEAN_COL]
    } | {
        suffix: metric_data[f"{METRIC_VALUE}_{suffix}"]
        for suffix in suffixes
    }

    # Prepare the data: extract only the needed columns
    plot_data = pd.DataFrame({
        TEST_NAME: metric_data[TEST_NAME],
    } | relevant_metrics)

    plot_data = plot_data.set_index(TEST_NAME)

    x = np.arange(len(plot_data))
    width = 0.8 / len(relevant_metrics.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, suffix in enumerate(relevant_metrics.keys()):
        values = plot_data[suffix]
        bar_positions = x + i * width
        bars = ax.bar(bar_positions, values, width, label=suffix)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8)

    ax.set_title(f"Comparison for {metric}")
    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Model Name")
    ax.set_xticks(x + width * (len(suffixes) - 1) / 2)
    ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
    ax.legend(title='Suffix')

    return fig


def generate_plots(output_path: Path, name_mapping: dict[str, str],
                   merged: pd.DataFrame, suffixes: set[str]) -> None:
    metric_types = merged[METRIC_TYPE].unique()
    with PdfPages(output_path.as_posix()) as pdf:
        write_name_mapping_table(name_mapping, pdf)
        for metric in metric_types:
            fig = plot_metric(merged, metric, suffixes)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def parse_perf_data(
        perf_files: list[str]) -> tuple[dict[str, str], pd.DataFrame, set[str]]:
    perfs = {
        Path(file_path).name: pd.read_csv(file_path)
        for file_path in perf_files
    }

    merged = pd.DataFrame(columns=[TEST_NAME, METRIC_TYPE])
    suffixes: set[str] = set()
    for file_path, df in perfs.items():
        df = df.rename(
            columns={
                column: f'{column}_{file_path}'
                for column in df.columns
                if column not in (TEST_NAME, METRIC_TYPE)
            })
        merged = merged.merge(df, on=[TEST_NAME, METRIC_TYPE], how='outer')
        suffixes.add(file_path)

    merged[MEAN_COL] = merged[[
        f'{METRIC_VALUE}_{suffix}' for suffix in suffixes
    ]].mean(axis=1)
    merged, name_mapping = shorten_names(merged)

    return name_mapping, merged, suffixes


def generate_perf_compare_report(perf_files: list[str],
                                 output_path: str) -> None:
    name_mapping, merged, suffixes = parse_perf_data(perf_files)
    generate_plots(Path(output_path), name_mapping, merged, suffixes)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a report comparing multiple performance csvs")
    parser.add_argument('--files',
                        nargs='*',
                        help="A list of csv files to compare")
    parser.add_argument("--output_path",
                        type=str,
                        help="Output path for report (pdf file)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    generate_perf_compare_report(args.files, args.output_path)


if __name__ == '__main__':
    main()
