import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Prepare start worker commands")
    parser.add_argument("--allocations", type=str, required=True, help="Allocations")
    parser.add_argument("--gen-world-size", type=int, required=True, help="Generation world size")
    parser.add_argument("--ctx-world-size", type=int, required=True, help="Context world size")
    parser.add_argument("--gpus-per-node", type=int, required=True, help="GPUs per node")
    parser.add_argument("--container-image", type=str, required=True, help="Container image")
    parser.add_argument("--container-name", type=str, required=True, help="Container name")
    parser.add_argument("--container-mount", type=str, required=True, help="Container mounts")
    parser.add_argument("--work-dir", type=str, required=True, help="Work directory")
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument("--benchmark-mode", type=str, required=True, help="Benchmark mode")
    parser.add_argument("--concurrency-list", type=str, required=True, help="Concurrency list")
    parser.add_argument("--numa-bind", type=str, required=True, help="NUMA bind")
    parser.add_argument("--log-dir", type=str, required=True, help="Log directory")
    parser.add_argument("--nsys-on", type=str, required=True, help="NSYS on")
    parser.add_argument(
        "--ctx-profile-range", type=str, required=True, help="Context profile range"
    )
    parser.add_argument(
        "--gen-profile-range", type=str, required=True, help="Generation profile range"
    )
    parser.add_argument("--ctx-config-path", type=str, required=True, help="Context config path")
    parser.add_argument("--gen-config-path", type=str, required=True, help="Generation config path")
    parser.add_argument(
        "--worker-env-var", type=str, required=True, help="Worker environment variables"
    )
    args = parser.parse_args()

    with open(args.allocations, "r") as f:
        allocations = json.load(f)

    start_worker_cmds = []
    for allocation in allocations:
        server_type = allocation["server_type"]
        cuda_devices = ",".join([str(device) for device in list(allocation["nodes"].values())[0]])
        worker_env_var = args.worker_env_var + f" CUDA_VISIBLE_DEVICES={cuda_devices}"
        cmd = [
            "srun",
            "-l",
            "--nodelist",
            ",".join(allocation["nodes"].keys()),
            "-N",
            str(len(allocation["nodes"])),
            "--ntasks",
            str(args.gen_world_size) if server_type == "GEN" else str(args.ctx_world_size),
            "--ntasks-per-node",
            str(args.gpus_per_node),
            "--container-image",
            args.container_image,
            "--container-name",
            args.container_name,
            "--container-mounts",
            args.container_mount,
            "--mpi",
            "pmix",
            "--overlap",
            "bash",
            os.path.join(args.work_dir, "start_worker.sh"),
            server_type,
            str(allocation["server_id"]),
            args.model_path,
            str(allocation["port"]),
            args.benchmark_mode,
            args.concurrency_list,
            args.numa_bind,
            args.log_dir,
            args.nsys_on,
            args.gen_profile_range if server_type == "GEN" else args.ctx_profile_range,
            args.gen_config_path if server_type == "GEN" else args.ctx_config_path,
            f'"{worker_env_var}"',
            f"&> {args.log_dir}/output_{server_type}_{allocation['server_id']}.log &",
        ]
        start_worker_cmds.append(" ".join(cmd))

    for cmd in start_worker_cmds:
        print(cmd)
    with open(os.path.join(args.log_dir, "start_worker_cmds.txt"), "w") as f:
        f.write("\n".join(start_worker_cmds) + "\n")


if __name__ == "__main__":
    main()
