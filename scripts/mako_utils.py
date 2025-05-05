import argparse
import json
import os
import subprocess


def get_mako_opts(get_mako_script, mako_args=None, chip_mapping_file=None):
    """
    Get Mako options by running the specified script.
    """
    if mako_args is None:
        mako_args = []

    mako_opts = {}
    list_mako_cmd = ["python3", get_mako_script, "--device", "0"]

    if mako_args:
        mako_opt_args = [f"--mako-opt {arg}" for arg in mako_args]
        list_mako_cmd.extend(mako_opt_args)

    if chip_mapping_file:
        list_mako_cmd.append(f"--chip-mapping-file {chip_mapping_file}")

    print(f"Running Mako command: {' '.join(list_mako_cmd)}")

    try:
        result = subprocess.run(
            list_mako_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30 * 60  # 30 minutes
        )
        result.check_returncode()
        turtle_output = result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run Mako script: {e.stdout}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("Mako script timed out.")

    # Parse the output
    started_mako_opts = False
    for line in turtle_output.splitlines():
        if started_mako_opts:
            if "=" in line:
                param, value = line.split("=", 1)
                param = param.strip()
                value = value.strip()
                if value.lower() == "none":
                    value = None
                elif value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                mako_opts[param] = value
        if line.strip() == "Mako options:":
            started_mako_opts = True

    print(f"Mako options: {json.dumps(mako_opts, indent=2)}")
    return mako_opts


def render_test_db(test_context, llm_src, stage_name, chip_mapping_file=None):
    """
    Render the test database using Mako options.
    """
    script_path = os.path.join(llm_src,
                               "tests/integration/defs/sysinfo/get_sysinfo.py")
    mako_args = []
    is_post_merge = "Post-Merge" in stage_name
    mako_args.append(f"stage={'post_merge' if is_post_merge else 'pre_merge'}")

    if "-PyTorch-" in stage_name:
        mako_args.append("backend=pytorch")
    elif "-TensorRT-" in stage_name:
        mako_args.append("backend=tensorrt")
    elif "-CPP-" in stage_name:
        mako_args.append("backend=cpp")

    if "-DeepSeek-" in stage_name:
        mako_args.append("auto_trigger=deepseek")
    else:
        mako_args.append("auto_trigger=others")

    mako_opts = get_mako_opts(script_path, mako_args, chip_mapping_file)

    test_db_path = os.path.join(llm_src, "tests/integration/test_lists/test-db")
    test_list = os.path.join(llm_src, f"{test_context}.txt")
    test_db_query_cmd = [
        "trt-test-db", "-d", test_db_path, "--context", test_context,
        "--test-names", "--output", test_list, "--match",
        json.dumps(mako_opts)
    ]

    print(f"Running test DB query command: {' '.join(test_db_query_cmd)}")

    try:
        subprocess.run(test_db_query_cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to render test DB: {e.stdout}") from e

    print(f"Test list generated at: {test_list}")
    return test_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mako Utilities")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for render_test_db
    render_test_db_parser = subparsers.add_parser(
        "render_test_db", help="Render the test database")
    render_test_db_parser.add_argument("--test-context",
                                       required=True,
                                       help="Test context")
    render_test_db_parser.add_argument("--llm-src",
                                       required=True,
                                       help="LLM source directory")
    render_test_db_parser.add_argument("--stage-name",
                                       required=True,
                                       help="Stage name")
    render_test_db_parser.add_argument("--chip-mapping-file",
                                       required=False,
                                       help="Path to the GPU chip mapping file")

    args = parser.parse_args()
    print_info(f"Arguments chip_mapping_file: {args.chip_mapping_file}")

    if args.command == "render_test_db":
        render_test_db(args.test_context, args.llm_src, args.stage_name,
                       args.chip_mapping_file)
