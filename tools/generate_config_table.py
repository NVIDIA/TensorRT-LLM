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

    # Group by key attributes to determine min/max concurrency
    # Key: (model, gpu, isl, osl) -> list of entry
    groups = defaultdict(list)
    for entry in data:
        key = (entry.get("model"), entry.get("gpu"), entry.get("isl"), entry.get("osl"))
        groups[key].append(entry)

    # Prepare output lines
    lines = []
    lines.append(".. list-table::")
    lines.append("   :header-rows: 1")
    lines.append("   :widths: 20 12 12 15 10 20 25")
    lines.append("")
    lines.append("   * - Model Name")
    lines.append("     - GPU")
    lines.append("     - Performance Profile")
    lines.append("     - ISL / OSL")
    lines.append("     - Concurrency")
    lines.append("     - Config")
    lines.append("     - Command")

    # Sort keys for consistent output
    # Key elements might be None, so convert to str/int safely
    sorted_keys = sorted(
        groups.keys(), key=lambda k: (str(k[0]), str(k[1]), int(k[2] or 0), int(k[3] or 0))
    )

    for key in sorted_keys:
        entries = groups[key]
        # Sort by concurrency
        entries.sort(key=lambda x: int(x.get("concurrency", 0)))

        # Get concurrency range for this group
        min_conc = int(entries[0].get("concurrency", 0))
        max_conc = int(entries[-1].get("concurrency", 0))
        conc_range = max_conc - min_conc

        for entry in entries:
            model = entry.get("model", "N/A")
            gpu = entry.get("gpu", "N/A")
            isl = entry.get("isl", "N/A")
            osl = entry.get("osl", "N/A")
            conc = int(entry.get("concurrency", 0))
            config_path = entry.get("config_path", "")

            # Determine profile based on relative position in concurrency range
            if len(entries) == 1:
                # Single entry: use concurrency value as heuristic
                # Lower concurrency typically means lower latency
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

            lines.append(f"   * - {model}")
            lines.append(f"     - {gpu}")
            lines.append(f"     - {profile}")
            lines.append(f"     - {isl} / {osl}")
            lines.append(f"     - {conc}")
            lines.append(f"     - {full_config_path}")
            lines.append(f"     - ``{command}``")

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
