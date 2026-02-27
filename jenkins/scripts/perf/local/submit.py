#!/usr/bin/env python3
import argparse
import os
import re
from datetime import datetime

import yaml

AGGR_CONFIG_FOLDER = os.environ.get("AGG_CONFIG_FOLDER", "tests/scripts/perf-sanity/aggregated")
DISAGG_CONFIG_FOLDER = os.environ.get("DISAGG_CONFIG_FOLDER", "tests/scripts/perf-sanity/disaggregated")


def get_llm_src_default():
    """Get default llm_src path by going up 4 directories from this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, "..", "..", "..", ".."))


def detect_config_type(config):
    """Detect if config is disagg (has worker_config) or aggr (has server_configs)."""
    if "worker_config" in config:
        return "disagg"
    elif "server_configs" in config:
        return "aggr"
    else:
        raise ValueError("Cannot detect config type: missing worker_config or server_configs")


def extract_test_case_name(test_string):
    """Extract test case name from test string with brackets.

    Args:
        test_string: Full test string like 'perf/test_perf_sanity.py::test_e2e[aggr-config-test]'

    Returns:
        str: Test case name (content inside brackets), e.g., 'aggr-config-test'
    """
    # Remove TIMEOUT suffix if present
    test_string = re.sub(r"\s+TIMEOUT\s*\(\d+\)\s*$", "", test_string.strip())

    if "[" not in test_string or "]" not in test_string:
        raise ValueError(
            f"Invalid test string format. Expected test name with brackets: {test_string}"
        )
    return test_string.split("[")[-1].split("]")[0]


def parse_test_string(test_case_name: str):
    """Parse test case name to get config base name, select pattern, runtime, and benchmark_mode.

    Test name formats:
    - Disagg e2e: disagg_upload-e2e-{config_base}
    - Disagg gen_only: disagg_upload-gen_only-{config_base}
    - ctx_only: aggr_upload-ctx_only-{config_base} (runs aggr mode but reads disagg config)
    - Regular aggr: aggr_upload-{config}-{server_name}

    Returns:
        tuple: (config_base_name, select_pattern, runtime_mode, benchmark_mode)
            - runtime_mode: "aggregated" or "disaggregated"
            - benchmark_mode: "e2e", "gen_only", "ctx_only", or None (for normal aggr)
    """
    labels = test_case_name.split("-")

    assert len(labels) > 1, "perf_sanity test must have a config file!"

    prefix = labels[0]
    is_disagg_prefix = "disagg" in prefix
    is_aggr_prefix = "aggr" in prefix

    if is_disagg_prefix:
        # Disagg format: disagg_upload-{e2e|gen_only}-{config_base}
        assert len(labels) > 2, "Disagg test must have benchmark_mode and config!"
        benchmark_mode = labels[1]  # e2e or gen_only
        assert benchmark_mode in ("e2e", "gen_only"), (
            f"Invalid benchmark_mode for disagg: {benchmark_mode}"
        )
        runtime_mode = "disaggregated"
        config_base_name = "-".join(labels[2:])
        select_pattern = None
    elif is_aggr_prefix:
        # Check if this is ctx_only (aggr_upload-ctx_only-{config_base})
        if len(labels) > 2 and labels[1] == "ctx_only":
            # ctx_only: aggr_upload-ctx_only-{config_base}
            # Runs in aggregated mode but reads disagg config
            benchmark_mode = "ctx_only"
            runtime_mode = "aggregated"
            config_base_name = "-".join(labels[2:])
            select_pattern = None
        else:
            # Regular aggr: aggr_upload-config_yml or aggr_upload-config_yml-server_config_name
            benchmark_mode = None
            runtime_mode = "aggregated"
            config_base_name = labels[1]
            # select_pattern is server config name (e.g., "r1_fp8_dep8_mtp1_1k1k")
            select_pattern = "-".join(labels[2:]) if len(labels) > 2 else None
    else:
        raise ValueError(f"Invalid test name prefix: {prefix}")

    return config_base_name, select_pattern, runtime_mode, benchmark_mode


def get_config_yaml_path(llm_src, config_base_name, benchmark_mode):
    """Get config yaml path based on benchmark_mode.

    Args:
        llm_src: Path to LLM source code
        config_base_name: Base name of config file (without .yaml extension)
        benchmark_mode: "e2e", "gen_only", "ctx_only", or None (for normal aggr)

    Returns:
        str: Full path to config yaml file
    """
    if benchmark_mode in ("e2e", "gen_only", "ctx_only"):
        config_dir = os.path.join(llm_src, DISAGG_CONFIG_FOLDER)
    else:
        config_dir = os.path.join(llm_src, AGGR_CONFIG_FOLDER)

    config_yaml_path = os.path.join(config_dir, f"{config_base_name}.yaml")

    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Config file not found: {config_yaml_path}")

    return config_yaml_path


def get_hardware_config(config, runtime_mode, benchmark_mode, test_name=None):
    """Get hardware config based on mode."""
    hardware = config.get("hardware", {})
    gpus_per_node = hardware.get("gpus_per_node")

    if gpus_per_node is None:
        raise ValueError("Missing gpus_per_node in hardware configuration")

    # ctx_only mode reads disagg config but runs in aggregated mode
    if benchmark_mode == "ctx_only":
        # Use ctx worker config to determine hardware
        worker_config = config.get("worker_config", {})
        ctx_config = worker_config.get("ctx", {})
        ctx_tp = ctx_config.get("tensor_parallel_size", 1)
        ctx_pp = ctx_config.get("pipeline_parallel_size", 1)
        ctx_cp = ctx_config.get("context_parallel_size", 1)
        gpus_per_server = ctx_tp * ctx_pp * ctx_cp

        nodes_per_server = (gpus_per_server + gpus_per_node - 1) // gpus_per_node
        total_nodes = nodes_per_server
        total_gpus = total_nodes * gpus_per_node
        gpus_per_node_per_server = min(gpus_per_server, gpus_per_node)

        return {
            "gpus_per_node": gpus_per_node,
            "gpus_per_server": gpus_per_server,
            "nodes_per_server": nodes_per_server,
            "gpus_per_node_per_server": gpus_per_node_per_server,
            "total_nodes": total_nodes,
            "total_gpus": total_gpus,
        }
    elif runtime_mode == "aggregated":
        # Normal aggregated mode
        server_configs = config.get("server_configs", [])
        server_config = None
        for sc in server_configs:
            if sc.get("name") == test_name:
                server_config = sc
                break

        if server_config is None:
            raise ValueError(f"Server config not found for test_name: {test_name}")

        tp = server_config.get("tensor_parallel_size", 1)
        pp = server_config.get("pipeline_parallel_size", 1)
        cp = server_config.get("context_parallel_size", 1)
        gpus_per_server = tp * pp * cp

        nodes_per_server = (gpus_per_server + gpus_per_node - 1) // gpus_per_node
        total_nodes = nodes_per_server
        total_gpus = total_nodes * gpus_per_node
        gpus_per_node_per_server = min(gpus_per_server, gpus_per_node)

        return {
            "gpus_per_node": gpus_per_node,
            "gpus_per_server": gpus_per_server,
            "nodes_per_server": nodes_per_server,
            "gpus_per_node_per_server": gpus_per_node_per_server,
            "total_nodes": total_nodes,
            "total_gpus": total_gpus,
        }
    else:
        # Disaggregated mode (e2e or gen_only)
        worker_config = config.get("worker_config", {})

        num_ctx_servers = (
            0
            if benchmark_mode == "gen_only"
            and "gen_only_no_context" in config.get("benchmark", {}).get("mode", "")
            else hardware.get("num_ctx_servers")
        )
        num_gen_servers = hardware.get("num_gen_servers")

        ctx_config = worker_config.get("ctx", {})
        gen_config = worker_config.get("gen", {})
        ctx_tp = ctx_config.get("tensor_parallel_size", 1)
        ctx_pp = ctx_config.get("pipeline_parallel_size", 1)
        ctx_cp = ctx_config.get("context_parallel_size", 1)
        gpus_per_ctx_server = ctx_tp * ctx_pp * ctx_cp
        gen_tp = gen_config.get("tensor_parallel_size", 1)
        gen_pp = gen_config.get("pipeline_parallel_size", 1)
        gen_cp = gen_config.get("context_parallel_size", 1)
        gpus_per_gen_server = gen_tp * gen_pp * gen_cp

        if None in [
            num_ctx_servers,
            num_gen_servers,
            gpus_per_ctx_server,
            gpus_per_gen_server,
        ]:
            raise ValueError("Missing required hardware configuration")

        nodes_per_ctx_server = (gpus_per_ctx_server + gpus_per_node - 1) // gpus_per_node
        nodes_per_gen_server = (gpus_per_gen_server + gpus_per_node - 1) // gpus_per_node

        gpus_per_node_per_ctx_server = min(gpus_per_ctx_server, gpus_per_node)
        gpus_per_node_per_gen_server = min(gpus_per_gen_server, gpus_per_node)

        total_nodes = (
            num_ctx_servers * nodes_per_ctx_server + num_gen_servers * nodes_per_gen_server
        )
        total_gpus = total_nodes * gpus_per_node

        return {
            "num_ctx_servers": num_ctx_servers,
            "num_gen_servers": num_gen_servers,
            "gpus_per_node": gpus_per_node,
            "gpus_per_ctx_server": gpus_per_ctx_server,
            "gpus_per_gen_server": gpus_per_gen_server,
            "nodes_per_ctx_server": nodes_per_ctx_server,
            "nodes_per_gen_server": nodes_per_gen_server,
            "gpus_per_node_per_ctx_server": gpus_per_node_per_ctx_server,
            "gpus_per_node_per_gen_server": gpus_per_node_per_gen_server,
            "total_nodes": total_nodes,
            "total_gpus": total_gpus,
        }


def get_env_config(config, runtime_mode):
    """Get env config based on mode."""
    if runtime_mode == "aggregated":
        return {}
    env = config.get("environment", {})
    return {
        "worker_env_var": env.get("worker_env_var", ""),
        "server_env_var": env.get("server_env_var", ""),
        "benchmark_env_var": env.get("benchmark_env_var", ""),
    }


def get_benchmark_config(config, benchmark_mode):
    """Get benchmark config based on mode."""
    if benchmark_mode is None:
        return {}
    benchmark = config.get("benchmark", {})
    concurrency_str = benchmark.get("concurrency_list", "1")
    concurrency = int(concurrency_str) if isinstance(concurrency_str, str) else concurrency_str

    return {
        "mode": benchmark_mode,
        "concurrency": concurrency,
    }


def generate_sbatch_params(args, hardware_config, work_dir):
    """Generate #SBATCH parameters."""
    total_nodes = hardware_config["total_nodes"]
    gpus_per_node = hardware_config["gpus_per_node"]
    total_gpus = hardware_config["total_gpus"]
    lines = [
        "#!/bin/bash",
        f"#SBATCH --nodes={total_nodes}",
        f"#SBATCH --segment={total_nodes}",
        f"#SBATCH --ntasks={total_gpus}",
        f"#SBATCH --ntasks-per-node={gpus_per_node}",
        f"#SBATCH --gpus-per-node={gpus_per_node}",
        f"#SBATCH --gres=gpu:{gpus_per_node}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --account={args.account}",
        f"#SBATCH -J {args.job_name}",
        f"#SBATCH -o {work_dir}/slurm-%j.out",
    ]
    return lines


def generate_srun_args(args, runtime_mode, timestamp):
    """Generate srun arguments."""
    is_disagg = runtime_mode == "disaggregated"
    container_name = f"{'disagg' if is_disagg else 'aggr'}_test-{timestamp}"

    lines = [
        f"--container-name={container_name}",
        f"--container-image={args.image}",
    ]

    if args.work_dir:
        lines.append(f"--container-workdir={args.work_dir}")

    if args.mounts:
        lines.append(f"--container-mounts={args.mounts}")

    lines.append("--container-env=NVIDIA_IMEX_CHANNELS")

    if is_disagg:
        lines.append("--mpi=pmix")
    else:
        lines.append("--mpi=pmi2")

    return lines


def generate_pytest_command(
    llm_src, work_dir, config_file_base_name, select_pattern, runtime_mode, benchmark_mode
):
    """Generate pytest command and test list."""
    # Generate test list content based on runtime_mode and benchmark_mode
    if runtime_mode == "disaggregated":
        # disagg_upload-{e2e|gen_only}-{config_base}
        test_list_content = (
            f"perf/test_perf_sanity.py::test_e2e[disagg-{benchmark_mode}-{config_file_base_name}]"
        )
    elif benchmark_mode == "ctx_only":
        # aggr_upload-ctx_only-{config_base}
        test_list_content = (
            f"perf/test_perf_sanity.py::test_e2e[aggr-ctx_only-{config_file_base_name}]"
        )
    else:
        # Normal aggr: aggr-{config}-{select_pattern}
        test_list_content = (
            f"perf/test_perf_sanity.py::test_e2e[aggr-{config_file_base_name}-{select_pattern}]"
        )

    test_list_path = os.path.join(work_dir, "test_list.txt")

    pytest_command = (
        f"pytest -v -s "
        f"--test-prefix={llm_src}/tests/integration/defs "
        f"--test-list={test_list_path} "
        f"--output-dir={work_dir} "
        f"-o junit_logging=out-err"
    )

    return pytest_command, test_list_content, test_list_path


def remove_whitespace_lines(lines):
    """Remove empty lines and strip whitespace."""
    return [line for line in lines if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM launch script for local runs (aggregated or disaggregated)"
    )
    parser.add_argument(
        "--test-list",
        default="",
        help="Test string, e.g., 'perf/test_perf_sanity.py::test_e2e[aggr-config-test_name]'. "
        "If both --test-list and --config-file are provided, --test-list takes precedence.",
    )
    parser.add_argument("--config-file", default="", help="Path to config YAML file")
    parser.add_argument(
        "--test-name",
        default="",
        help="Test name (only used for normal aggregated mode when --config-file is provided)",
    )
    parser.add_argument(
        "--benchmark-mode",
        default="",
        choices=["", "e2e", "gen_only", "ctx_only"],
        help="Benchmark mode for disagg config (when --config-file is provided)",
    )
    parser.add_argument("--partition", required=True, help="SLURM partition")
    parser.add_argument("--time", default="02:00:00", help="SLURM time limit")
    parser.add_argument("--account", required=True, help="SLURM account")
    parser.add_argument("--job-name", required=True, help="SLURM job name")
    parser.add_argument("--image", required=True, help="Container image")
    parser.add_argument("--mounts", default="", help="Container mounts")
    parser.add_argument(
        "--work-dir",
        default="",
        help="Work directory (used for both workdir and container-workdir)",
    )
    parser.add_argument("--draft-launch-sh", default="", help="Path to draft-launch.sh script")
    parser.add_argument("--launch-sh", default="", help="Path to output launch.sh script")
    parser.add_argument("--run-sh", default="", help="Path to slurm_run.sh script")
    parser.add_argument("--install-sh", default="", help="Path to slurm_install.sh script")
    parser.add_argument("--llm-src", default="", help="Path to LLM source code")
    parser.add_argument("--llm-models-root", required=True, help="Path to LLM models root")
    parser.add_argument(
        "--build-wheel", action="store_true", help="Build wheel before running tests"
    )
    parser.add_argument(
        "--capture-nsys", action="store_true", help="Capture nsys profile"
    )
    parser.add_argument(
        "--nsys-start-stop",
        default="1-100",
        help="Nsys start-stop range (default: 1-100)",
    )

    args = parser.parse_args()

    # Determine llm_src
    llm_src = args.llm_src if args.llm_src else get_llm_src_default()
    llm_src = os.path.abspath(llm_src)

    # Determine config_yaml, config_file_base_name, select_pattern, runtime_mode, and benchmark_mode
    # --test-list takes precedence over --config-file
    if args.test_list:
        test_case_name = extract_test_case_name(args.test_list)
        config_file_base_name, select_pattern, runtime_mode, benchmark_mode = parse_test_string(
            test_case_name
        )
        config_yaml = get_config_yaml_path(llm_src, config_file_base_name, benchmark_mode)
    elif args.config_file:
        config_yaml = args.config_file
        config_file_base_name = os.path.splitext(os.path.basename(config_yaml))[0]

        # Load config to detect type
        with open(config_yaml, "r") as f:
            config = yaml.safe_load(f)

        config_type = detect_config_type(config)

        if config_type == "disagg":
            # Disagg config - need benchmark_mode
            benchmark_mode = args.benchmark_mode if args.benchmark_mode else "e2e"
            if benchmark_mode == "ctx_only":
                runtime_mode = "aggregated"
            else:
                runtime_mode = "disaggregated"
            select_pattern = None
        else:
            # Aggr config
            runtime_mode = "aggregated"
            benchmark_mode = None
            select_pattern = args.test_name
            if not select_pattern:
                raise ValueError("--test-name is required for aggregated config")
    else:
        raise ValueError("Either --test-list or --config-file must be provided")

    # Load config if not already loaded
    if not args.config_file:
        with open(config_yaml, "r") as f:
            config = yaml.safe_load(f)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine work_dir
    work_dir = args.work_dir
    if not work_dir:
        work_dir = os.path.join(llm_src, "jenkins", "scripts", "perf", "local", timestamp)
    os.makedirs(work_dir, exist_ok=True)

    # Determine paths
    launch_sh = args.launch_sh if args.launch_sh else os.path.join(work_dir, "slurm_launch.sh")
    run_sh = (
        args.run_sh
        if args.run_sh
        else os.path.join(llm_src, "jenkins", "scripts", "perf", "local", "slurm_run.sh")
    )
    install_sh = (
        args.install_sh
        if args.install_sh
        else os.path.join(llm_src, "jenkins", "scripts", "perf", "local", "slurm_install.sh")
    )
    draft_launch_sh = args.draft_launch_sh
    if not draft_launch_sh:
        draft_launch_sh = os.path.join(
            llm_src,
            "jenkins",
            "scripts",
            "perf",
            "disaggregated" if runtime_mode == "disaggregated" else "aggregated",
            "slurm_launch_draft.sh",
        )

    # Get configs based on mode
    env_config = get_env_config(config, runtime_mode)
    bm_config = get_benchmark_config(config, benchmark_mode)
    hardware_config = get_hardware_config(
        config,
        runtime_mode,
        benchmark_mode,
        test_name=select_pattern,
    )

    # Generate sbatch params
    sbatch_lines = generate_sbatch_params(args, hardware_config, work_dir)

    # Generate srun args
    srun_args_lines = generate_srun_args(args, runtime_mode, timestamp)

    # Generate pytest command
    pytest_command, test_list_content, test_list_path = generate_pytest_command(
        llm_src, work_dir, config_file_base_name, select_pattern, runtime_mode, benchmark_mode
    )

    # Write test list file
    with open(test_list_path, "w") as f:
        f.write(test_list_content + "\n")

    # Build script prefix lines
    script_prefix_lines = sbatch_lines.copy()

    # Add export variables
    script_prefix_lines.extend(
        [
            f"export llmSrcNode='{llm_src}'",
            f"export jobWorkspace='{work_dir}'",
            f"export runScript='{run_sh}'",
            f"export installScript='{install_sh}'",
            f"export configYamlPath='{config_yaml}'",
            f"export BUILD_WHEEL={'true' if args.build_wheel else 'false'}",
        ]
    )

    nsys_prefix = ""
    tllm_profile_start_stop = ""
    if args.capture_nsys:
        if runtime_mode == "disaggregated":
            nsys_output = f"{work_dir}/nsys.%q{{DISAGG_SERVING_TYPE}}.rank%q{{SLURM_PROCID}}"
        else:
            nsys_output = f"{work_dir}/nsys.rank%q{{SLURM_PROCID}}"
        nsys_prefix = (
            "nsys profile"
            " -t cuda,nvtx,python-gil"
            " --sample cpu"
            " --cuda-graph-trace node"
            " -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1,TLLM_TORCH_PROFILE_TRACE=trace.json"
            " --trace-fork-before-exec=true"
            " -f true"
            " --gpu-metrics-devices=none"
            " -c cudaProfilerApi"
            " --capture-range-end=stop"
            " --export=sqlite"
            f" -o {nsys_output}"
        )
        tllm_profile_start_stop = args.nsys_start_stop

    pytest_common_vars = (
        f"LLM_ROOT='{llm_src}' "
        f"LLM_BACKEND_ROOT='{llm_src}/triton_backend' "
        f"LLM_MODELS_ROOT='{args.llm_models_root}' "
        f"AGG_CONFIG_FOLDER='{AGGR_CONFIG_FOLDER}' "
        f"DISAGG_CONFIG_FOLDER='{DISAGG_CONFIG_FOLDER}' "
    )
    llmapi_launch = f"{llm_src}/tensorrt_llm/llmapi/trtllm-llmapi-launch"

    # Add shared exports
    script_prefix_lines.extend(
        [
            f"export CAPTURE_NSYS={'true' if args.capture_nsys else 'false'}",
            f'export NSYS_PREFIX="{nsys_prefix}"',
            f'export LLM_API_LAUNCH="{llmapi_launch}"',
            f'export PYTEST_COMMON_VARS="{pytest_common_vars}"',
            f'export PYTEST_COMMAND="{pytest_command}"',
        ]
    )

    worker_env_vars = (
        f"TLLM_PROFILE_START_STOP='{tllm_profile_start_stop}' "
        f"FLASHINFER_JIT_DIR=/tmp/flashinfer_jit_cache "
    )
    server_env_vars = ""
    benchmark_env_var = ""
    if runtime_mode == "disaggregated":
        # Build worker env vars
        worker_env_vars = env_config.get("worker_env_var", "")
        server_env_vars = env_config.get("server_env_var", "")
        benchmark_env_var = env_config.get("benchmark_env_var", "")
        # Handle gen only mode
        if "gen_only_no_context" in bm_config.get("mode", ""):
            worker_env_vars = f"TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 {worker_env_vars}"
            server_env_vars = f"TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 {server_env_vars}"
            script_prefix_lines.append("export TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1")
            srun_args_lines.append("--container-env=TRTLLM_DISAGG_BENCHMARK_GEN_ONLY")
        elif "gen_only" in bm_config.get("mode", ""):
            concurrency = bm_config.get("concurrency", 1)
            worker_env_vars = (
                f"TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1 "
                f"TLLM_BENCHMARK_REQ_QUEUES_SIZE={concurrency} {worker_env_vars}"
            )

        script_prefix_lines.extend(
            [
                f'export WORKER_ENV_VARS="{worker_env_vars}"',
                f'export SERVER_ENV_VARS="{server_env_vars}"',
                f'export BENCHMARK_ENV_VARS="{benchmark_env_var}"',
                (
                    'export pytestCommandWorker="unset UCX_TLS &&'
                    " $WORKER_ENV_VARS $PYTEST_COMMON_VARS"
                    " $NSYS_PREFIX $LLM_API_LAUNCH"
                    f' $PYTEST_COMMAND --junitxml={work_dir}/report.xml"'
                ),
                'export pytestCommandDisaggServer="$SERVER_ENV_VARS $PYTEST_COMMON_VARS $PYTEST_COMMAND"',
                'export pytestCommandBenchmark="$BENCHMARK_ENV_VARS $PYTEST_COMMON_VARS $PYTEST_COMMAND"',
                f"export numCtxServers={hardware_config.get('num_ctx_servers', '')}",
                f"export numGenServers={hardware_config.get('num_gen_servers', '')}",
                f"export gpusPerNode={hardware_config.get('gpus_per_node', '')}",
                f"export gpusPerCtxServer={hardware_config.get('gpus_per_ctx_server', '')}",
                f"export gpusPerGenServer={hardware_config.get('gpus_per_gen_server', '')}",
                f"export nodesPerCtxServer={hardware_config.get('nodes_per_ctx_server', '')}",
                f"export nodesPerGenServer={hardware_config.get('nodes_per_gen_server', '')}",
                f"export gpusPerNodePerCtxServer={hardware_config.get('gpus_per_node_per_ctx_server', '')}",
                f"export gpusPerNodePerGenServer={hardware_config.get('gpus_per_node_per_gen_server', '')}",
                f"export totalNodes={hardware_config.get('total_nodes', '')}",
                f"export totalGpus={hardware_config.get('total_gpus', '')}",
            ]
        )

        # Add srun args for disagg
        srun_args_lines.extend(
            [
                "--container-env=DISAGG_SERVING_TYPE",
                "--container-env=pytestCommand",
            ]
        )
    else:
        # Aggregated mode (including ctx_only)
        script_prefix_lines.extend(
            [
                f'export WORKER_ENV_VARS="{worker_env_vars}"',
                (
                    'export pytestCommand="$WORKER_ENV_VARS $PYTEST_COMMON_VARS $NSYS_PREFIX $LLM_API_LAUNCH'
                    f' $PYTEST_COMMAND --junitxml={work_dir}/report.xml"'
                ),
                f"export gpusPerNode={hardware_config.get('gpus_per_node', '')}",
                f"export gpusPerNodePerServer={hardware_config.get('gpus_per_node_per_server', '')}",
                f"export totalNodes={hardware_config.get('total_nodes', '')}",
                f"export totalGpus={hardware_config.get('total_gpus', '')}",
            ]
        )

    # Remove whitespace lines
    script_prefix_lines = remove_whitespace_lines(script_prefix_lines)

    # Format srun args
    srun_args_lines = ["srunArgs=("] + [f'  "{line}"' for line in srun_args_lines] + [")"]
    srun_args = "\n".join(srun_args_lines)

    # Read draft launch script
    with open(draft_launch_sh, "r") as f:
        draft_launch_content = f.read()
    draft_launch_lines = draft_launch_content.split("\n")
    draft_launch_lines = remove_whitespace_lines(draft_launch_lines)
    draft_launch_content = "\n".join(draft_launch_lines)

    # Combine and write launch script
    script_prefix = "\n".join(script_prefix_lines)
    final_script = f"{script_prefix}\n\n{srun_args}\n\n{draft_launch_content}"

    with open(launch_sh, "w") as f:
        f.write(final_script)

    # Make scripts executable
    os.chmod(launch_sh, 0o755)
    for script_path in [run_sh, install_sh]:
        if os.path.exists(script_path):
            os.chmod(script_path, 0o755)
        else:
            print(f"Warning: Script not found, skipping chmod: {script_path}")

    print(f"\nLaunch script generated at: {launch_sh}")
    print("\nTo submit the job, run:")
    print(f"  sbatch {launch_sh}")


if __name__ == "__main__":
    main()
