import subprocess
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--clean', action='store_true', help='')
parser.add_argument('--TensorRT_path', type=str, help='TensorRT path')
args = parser.parse_args()
if args.clean:
    print("Clean compile cache!!!")
    if args.TensorRT_path is None:
        cmd1 = ['./scripts/build_wheel.py', '--cuda_architectures', '87', '-D', 'ENABLE_MULTI_DEVICE=0', '--clean']
    else:
        cmd1 = ['./scripts/build_wheel.py', '--cuda_architectures', '87', '-D', 'ENABLE_MULTI_DEVICE=0', '--trt_root', args.TensorRT_path, '--clean']
else:
    if args.TensorRT_path is None:
        cmd1 = ['./scripts/build_wheel.py', '--cuda_architectures', '87', '-D', 'ENABLE_MULTI_DEVICE=0']
    else:
        cmd1 = ['./scripts/build_wheel.py', '--cuda_architectures', '87', '-D', 'ENABLE_MULTI_DEVICE=0', '--trt_root', args.TensorRT_path]

    

with open('install_0.log', 'w') as f:
    try:
        print("Building ============")
        subprocess.check_call(cmd1, stdout=f, stderr=subprocess.STDOUT)
        print("Building successfully!!!")
    except subprocess.CalledProcessError:
        print("Build failed, stopping execution.Please check install_0.log for more info")
        exit(1)

# If the build was successful, install the wheel using pip3
files = glob.glob('./build/tensorrt_llm-*.whl')

# Check if we found any files
if not files:
    print("No matching .whl files found in ./build directory")
    exit(1)

# Sort the files so the last one is the newest
files.sort()

# Get the last file in the list
file = files[-1]
cmd2 = ['pip3', 'install', file]
with open('install_1.log', 'w') as f:
    try:
        print("Installing ============")
        subprocess.check_call(cmd2, stdout=f, stderr=subprocess.STDOUT)
        print("Install successfully!!!")
    except subprocess.CalledProcessError:
        print("Installation failed, stopping execution.Please check install_1.log for more info")
        exit(1)
