import argparse
import os
import subprocess
import sys

from tensorrt_llm.llmapi.disagg_utils import (get_server_configs_dict,
                                              parse_disagg_config_file)


def get_cmd(server_configs, args):

    if args.communication_protocol == "mpi":
        #Get total number of ranks from config file
        [total_ranks, _] = get_server_configs_dict(server_configs)

        cmd = "mpirun --allow-run-as-root"
        cmd += ' -n ' + str(
            total_ranks
        ) + ' trtllm-serve disaggregated_mpi_worker -c ' + args.disagg_config_file
    else:
        raise NotImplementedError(
            f"Communication protocol {args.communication_protocol} is not supported yet"
        )

    return cmd.split(" ")


def main():
    parser = argparse.ArgumentParser(description="Launch disaggregated workers")
    parser.add_argument("-c",
                        "--disagg_config_file",
                        type=str,
                        required=True,
                        help="Path to the YAML configuration file")
    parser.add_argument("--communication_protocol",
                        choices=["mpi", "ucx"],
                        default="mpi",
                        help="Communication protocol")
    args = parser.parse_args()

    disagg_config = parse_disagg_config_file(args.disagg_config_file)
    cmd = get_cmd(disagg_config.server_configs, args)

    print(f"Launching disaggregated workers with command: {cmd}", flush=True)

    env = os.environ.copy()
    process = subprocess.Popen(cmd,
                               env=env,
                               stdout=sys.stdout,
                               stderr=sys.stderr)

    process.wait()


if __name__ == "__main__":
    main()
