import os
import sys
from collections import defaultdict

import yaml


def generate_rst(yaml_path, output_file=None):
    """Generate RST table from YAML config database.

    Args:
        yaml_path: Path to scenario_list.yaml
        output_file: Optional output file path. If None, prints to stdout.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Group by model first, then by key attributes
    # Structure: model -> (gpu, isl, osl) -> list of entries
    model_groups = defaultdict(lambda: defaultdict(list))
    for entry in data:
        model = entry.get("model", "Unknown Model")
        key = (entry.get("gpu"), entry.get("isl"), entry.get("osl"))
        model_groups[model][key].append(entry)

    # Prepare output lines
    lines = []

    # Sort models alphabetically
    sorted_models = sorted(model_groups.keys())

    for model in sorted_models:
        lines.append(f".. start-{model}")
        lines.append("")
        # Section Header for Model
        lines.append(f".. _{model}:")
        lines.append("")
        lines.append(model)
        lines.append("^" * len(model))
        lines.append("")

        # Table Header
        lines.append(".. list-table::")
        lines.append("   :width: 100%")
        lines.append("   :header-rows: 1")
        # Adjusted widths removing 'Model Name' (originally 20)
        # Distributing remaining space roughly to Config/Command which are long
        lines.append("   :widths: 12 15 18 10 20 25")
        lines.append("")
        lines.append("   * - GPU")
        lines.append("     - Performance Profile")
        lines.append("     - ISL / OSL")
        lines.append("     - Concurrency")
        lines.append("     - Config")
        lines.append("     - Command")

        # Process entries for this model
        subgroups = model_groups[model]

        # Sort subgroups by GPU, ISL, OSL
        sorted_keys = sorted(
            subgroups.keys(), key=lambda k: (str(k[0]), int(k[1] or 0), int(k[2] or 0))
        )

        for key in sorted_keys:
            entries = subgroups[key]
            # Sort by concurrency
            entries.sort(key=lambda x: int(x.get("concurrency", 0)))

            # Get concurrency range for this group to determine profile
            min_conc = int(entries[0].get("concurrency", 0))
            max_conc = int(entries[-1].get("concurrency", 0))
            conc_range = max_conc - min_conc

            for entry in entries:
                gpu = entry.get("gpu", "N/A")
                isl = entry.get("isl", "N/A")
                osl = entry.get("osl", "N/A")
                conc = int(entry.get("concurrency", 0))
                config_path = entry.get("config_path", "")

                # Determine profile based on relative position in concurrency range
                if len(entries) == 1:
                    # Single entry: use concurrency value as heuristic
                    if conc <= 16:
                        profile = "Min Latency"
                    elif conc >= 64:
                        profile = "Max Throughput"
                    else:
                        profile = "Balanced"
                else:
                    # Multiple entries: use relative position
                    relative_pos = (conc - min_conc) / conc_range if conc_range > 0 else 0.5

                    if relative_pos < 0.33:
                        profile = "Min Latency"
                    elif relative_pos > 0.67:
                        profile = "Max Throughput"
                    else:
                        profile = "Balanced"

                full_config_path = os.path.join("tensorrt_llm/configure", config_path)
                command = f"trtllm-serve {model} --extra_llm_api_options {full_config_path}"

                config_filename = os.path.basename(full_config_path)
                github_url = f"https://github.com/NVIDIA/TensorRT-LLM/blob/main/{full_config_path}"
                config_link = f"`{config_filename} <{github_url}>`_"

                lines.append(f"   * - {gpu}")
                lines.append(f"     - {profile}")
                lines.append(f"     - {isl} / {osl}")
                lines.append(f"     - {conc}")
                lines.append(f"     - {config_link}")
                lines.append(f"     - ``{command}``")

        lines.append("")  # Space between tables
        lines.append(f".. end-{model}")
        lines.append("")

    # Output to file or stdout
    output_text = "\n".join(lines)
    if output_file:
        with open(output_file, "w") as f:
            f.write(output_text)
        print(f"Generated table written to: {output_file}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    # Assume script is run from repo root or tools dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script is in tools/, repo root is parent
    repo_root = os.path.dirname(script_dir)

    # Just to be safe, look for the file in fixed location relative to this script
    yaml_path = os.path.join(repo_root, "tensorrt_llm/configure/database/scenario_list.yaml")

    if not os.path.exists(yaml_path):
        # Try relative to CWD if script logic above fails (e.g. symlinks or strange structure)
        yaml_path = "tensorrt_llm/configure/database/scenario_list.yaml"

    if not os.path.exists(yaml_path):
        print(f"Error: YAML file not found at {yaml_path}", file=sys.stderr)
        sys.exit(1)

    # Generate to separate file
    output_path = os.path.join(repo_root, "docs/source/deployment-guide/comprehensive_table.rst")
    generate_rst(yaml_path, output_file=output_path)
