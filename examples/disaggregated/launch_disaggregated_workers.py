import argparse
import logging
import os
import subprocess
from typing import Literal

import torch
from torch.cuda import device_count

if (os.getenv("OMPI_COMM_WORLD_RANK")):
    env_global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
elif (os.getenv("SLURM_PROCID")):
    env_global_rank = int(os.environ["SLURM_PROCID"])
else:
    raise RuntimeError("Could not determine rank from environment")
device_id = env_global_rank % device_count()
print(
    f"env_global_rank: {env_global_rank}, set device_id: {device_id} before importing mpi4py"
)
torch.cuda.set_device(device_id)

from tensorrt_llm.llmapi.disagg_utils import parse_disagg_config_file

logging.basicConfig(level=logging.INFO)


def get_cmd(server_configs, args):

    if args.communication_protocol == "mpi":
        #Total number of ranks parse on config file
        total_ranks = sum(cfg.instance_num_ranks for cfg in server_configs)

        cmd = "mpirun --allow-run-as-root"
        cmd + ' -n ' + str(
            total_ranks
        ) + ' trtllm-serve disaggregated_mpi_worker -c ' + args.disagg_config_file + ' --communication_protocol ' + args.communication_protocol
    else:
        raise NotImplementedError(
            f"Communication protocol {args.communication_protocol} is not supported yet"
        )

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch disaggregated workers")
    parser.add_argument("-c",
                        "--disagg_config_file",
                        type=str,
                        required=True,
                        help="Path to the YAML configuration file")
    parser.add_argument("--communication_protocol",
                        type=Literal["mpi", "ucx"],
                        default="mpi",
                        help="Communication protocol")
    args = parser.parse_args()

    disagg_config = parse_disagg_config_file(args.disagg_config_file)
    cmd = get_cmd(disagg_config.server_configs, args)

    env = os.environ.copy()
    subprocess.Popen(cmd, env=env)


if __name__ == "__main__":
    main()
