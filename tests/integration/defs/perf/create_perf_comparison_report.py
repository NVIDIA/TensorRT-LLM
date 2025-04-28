import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def shorten_names(merged: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:

    def clean_name(name: str) -> str:
        try:
            machine = name.split('-')[0]
            test = name.split('[')[1][:-1]
            return f"{machine}-{test}"
        except:
            return name

    name_mapping = {
        k: f'configuration_{i+1}'
        for i, k in enumerate(
            set(clean_name(config) for config in merged['model_name']))
    }
    merged['model_name'] = merged['model_name'].apply(
        lambda name: name_mapping[clean_name(name)])
    return merged, name_mapping


def write_name_mapping(name_mapping: dict[str, str], pdf: PdfPages) -> None:
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


def generate_plots(output_path: Path, name_mapping: dict[str, str],
                   merged: pd.DataFrame) -> None:
    metric_types = merged['metric_type'].unique()
    with PdfPages(output_path.as_posix()) as pdf:
        write_name_mapping(name_mapping, pdf)
        for metric in metric_types:
            metric_data = merged[merged['metric_type'] == metric]

            plot_data = pd.DataFrame({
                'model_name':
                metric_data['model_name'],
                'base':
                metric_data['metric_value_base'],
                'target':
                metric_data['metric_value_target']
            })

            plot_data = plot_data.set_index('model_name')
            plot_data.plot(kind='bar', figsize=(10, 6))

            plt.title(f"Comparison for {metric}")
            plt.ylabel("Metric Value")
            plt.xlabel("Model Name")
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def generate_perf_compare_report(base_perf: str, target_perf: str,
                                 output_path: str):
    base_df = pd.read_csv(base_perf)
    target_df = pd.read_csv(target_perf)

    merged = pd.merge(base_df,
                      target_df,
                      on=["model_name", "metric_type"],
                      suffixes=('_base', '_target'),
                      how='outer')
    merged, name_mapping = shorten_names(merged)

    generate_plots(Path(output_path), name_mapping, merged)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare two CSVs and generate a PDF report.")
    parser.add_argument("--base_csv_path",
                        type=str,
                        help="Path to the base CSV file")
    parser.add_argument("--target_csv_path",
                        type=str,
                        help="Path to the target CSV file")
    parser.add_argument("--output_path",
                        type=str,
                        help="Output path for report (pdf file)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    generate_perf_compare_report(args.base_csv_path, args.target_csv_path,
                                 args.output_path)


if __name__ == '__main__':
    main()
