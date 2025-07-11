import argparse
import os
import subprocess
import sys
from pathlib import Path

from packaging import version


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument(
        '--tritonserver',
        type=str,
        help='path to the tritonserver exe',
        default='/opt/tritonserver/bin/tritonserver',
    )
    parser.add_argument(
        '--grpc_port',
        type=str,
        help='tritonserver grpc port',
        default='8001',
    )
    parser.add_argument(
        '--http_port',
        type=str,
        help='tritonserver http port',
        default='8000',
    )
    parser.add_argument(
        '--metrics_port',
        type=str,
        help='tritonserver metrics port',
        default='8002',
    )
    parser.add_argument(
        '--force',
        '-f',
        action='store_true',
        help='launch tritonserver regardless of other instances running')
    parser.add_argument('--log',
                        action='store_true',
                        help='log triton server stats into log_file')
    parser.add_argument(
        '--log-file',
        type=str,
        help='path to triton log file',
        default='triton_log.txt',
    )
    parser.add_argument(
        '--no-mpi',
        action='store_true',
        help='Launch tritonserver without MPI (single instance mode)',
        default=False,
    )

    path = str(Path(__file__).parent.absolute()) + '/../all_models/gpt'
    parser.add_argument('--model_repo', type=str, default=path)

    parser.add_argument(
        '--tensorrt_llm_model_name',
        type=str,
        help=
        'Name(s) of the tensorrt_llm Triton model in the repo. Use comma to separate if multiple model names',
        default='tensorrt_llm',
    )

    parser.add_argument(
        '--multi-model',
        action='store_true',
        help=
        'Enable support for multiple TRT-LLM models in the Triton model repository'
    )

    parser.add_argument(
        '--disable-spawn-processes',
        action='store_true',
        help='Disable dynamic spawning of child processes when using multi-model'
    )

    parser.add_argument(
        '--multimodal_gpu0_cuda_mem_pool_bytes',
        type=int,
        default=0,
        help=
        'For multimodal usage, model instances need to transfer GPU tensors which requires to have enough cuda pool memory. We currently assume al multimodal_encoderss are on GPU 0.'
    )

    parser.add_argument(
        '--oversubscribe',
        action='store_true',
        help=
        'Append --oversubscribe to the mpirun command. Mainly for SLURM MPI usecases.'
    )

    parser.add_argument(
        '--trtllm_llmapi_launch',
        action='store_true',
        help='Launch tritonserver with trtllm-llmapi-launch',
        default=False,
    )
    parser.add_argument(
        '--exit_timeout',
        type=int,
        help='Exit timeout in seconds',
        default=None,
    )
    return parser.parse_args()


def number_of_gpus():
    output = os.popen('nvidia-smi --list-gpus').read()
    return len(output.strip().split('\n'))


def check_triton_version(required_version):
    try:
        current_version = version.Version(
            os.environ.get('NVIDIA_TRITON_SERVER_VERSION'))
        required_version = version.Version(required_version)
        return current_version > required_version
    except version.InvalidVersion:
        print("Invalid version format. Please use major.minor format.")
        return False


def add_multi_model_config(cmd, args):
    """Add multi-model configuration to command if enabled."""
    if args.multi_model and check_triton_version(
            '24.06') and not args.disable_spawn_processes:
        cmd += [
            '--pinned-memory-pool-byte-size=0', '--enable-peer-access=false'
        ]
        for j in range(number_of_gpus()):
            cmd += [f'--cuda-memory-pool-byte-size={j}:0']
    return cmd


def add_logging_config(cmd, log, log_file, rank=None):
    """Add logging configuration to command if enabled."""
    if log and (rank is None or rank == 0):
        cmd += ['--log-verbose=3', f'--log-file={log_file}']
    return cmd


def add_port_config(cmd, grpc_port, http_port, metrics_port):
    """Add port configuration to command."""
    cmd += [
        f'--grpc-port={grpc_port}',
        f'--http-port={http_port}',
        f'--metrics-port={metrics_port}',
    ]
    return cmd


def get_cmd(world_size,
            tritonserver,
            grpc_port,
            http_port,
            metrics_port,
            model_repo,
            log,
            log_file,
            tensorrt_llm_model_name,
            oversubscribe,
            multimodal_gpu0_cuda_mem_pool_bytes,
            no_mpi,
            trtllm_llmapi_launch,
            exit_timeout=None):
    if no_mpi:
        assert world_size == 1, "world size must be 1 when using no-mpi"

    use_mpi = not no_mpi
    cmd = []

    if use_mpi:
        cmd = ['mpirun', '--allow-run-as-root']
        if oversubscribe:
            cmd += ['--oversubscribe']

    for i in range(world_size):
        if use_mpi:
            cmd += ['-n', '1']
        if trtllm_llmapi_launch:
            cmd += ['trtllm-llmapi-launch']
        cmd += [tritonserver, f'--model-repository={model_repo}']
        if exit_timeout:
            cmd += [f'--exit-timeout-secs={exit_timeout}']

        # Add port configuration
        cmd = add_port_config(cmd, grpc_port, http_port, metrics_port)

        # Add logging if requested (only for rank 0)
        cmd = add_logging_config(cmd, log, log_file, i)

        # If rank is not 0, skip loading of models other than `tensorrt_llm_model_name`
        if (i != 0):
            cmd += ['--model-control-mode=explicit']
            model_names = tensorrt_llm_model_name.split(',')
            for name in model_names:
                cmd += [f'--load-model={name}']
        elif i == 0 and multimodal_gpu0_cuda_mem_pool_bytes != 0:
            cmd += [
                f'--cuda-memory-pool-byte-size=0:{multimodal_gpu0_cuda_mem_pool_bytes}'
            ]

        # Add multi-model configuration if enabled
        cmd = add_multi_model_config(cmd, args)

        # Add port configuration
        cmd = add_port_config(cmd, grpc_port, http_port, metrics_port)

        cmd += [
            '--disable-auto-complete-config',
            f'--backend-config=python,shm-region-prefix-name=prefix{i}_',
        ]
        if use_mpi:
            cmd += [':']
    return cmd


if __name__ == '__main__':
    args = parse_arguments()
    res = subprocess.run(['pgrep', '-r', 'R', 'tritonserver'],
                         capture_output=True,
                         encoding='utf-8')
    if res.stdout:
        pids = res.stdout.replace('\n', ' ').rstrip()
        msg = f'tritonserver process(es) already found with PID(s): {pids}.\n\tUse `kill {pids}` to stop them.'
        if args.force:
            print(msg, file=sys.stderr)
        else:
            raise RuntimeError(msg + ' Or use --force.')
    cmd = get_cmd(int(args.world_size), args.tritonserver, args.grpc_port,
                  args.http_port, args.metrics_port, args.model_repo, args.log,
                  args.log_file, args.tensorrt_llm_model_name,
                  args.oversubscribe, args.multimodal_gpu0_cuda_mem_pool_bytes,
                  args.no_mpi, args.trtllm_llmapi_launch, args.exit_timeout)
    env = os.environ.copy()
    if args.multi_model:
        if not args.disable_spawn_processes:
            assert args.world_size == 1, 'World size must be 1 when using multi-model without disable-spawn-processes. Processes will be spawned automatically to run the multi-GPU models'
        env['TRTLLM_ORCHESTRATOR'] = '1'
        env['TRTLLM_ORCHESTRATOR_SPAWN_PROCESSES'] = '0' if args.disable_spawn_processes else '1'
    subprocess.Popen(cmd, env=env)
