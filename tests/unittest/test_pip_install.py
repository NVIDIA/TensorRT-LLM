import argparse
import os
import subprocess
import sys
import sysconfig

import requests
from utils.llm_data import llm_models_root


def get_expected_license_files():
    """Get expected license files based on platform architecture."""
    platform_tag = sysconfig.get_platform()
    if "x86_64" in platform_tag:
        return ["LICENSE", "ATTRIBUTIONS-CPP-x86_64.md"]
    elif "arm64" in platform_tag or "aarch64" in platform_tag:
        return ["LICENSE", "ATTRIBUTIONS-CPP-aarch64.md"]
    else:
        raise RuntimeError(f"Unrecognized CPU architecture: {platform_tag}")


def verify_license_files():
    """Verify that the correct platform-specific license files are packaged."""
    expected_files = get_expected_license_files()

    result = subprocess.run(
        'python3 -c "from importlib.metadata import distribution; '
        'import os; '
        'dist_path = distribution(\'tensorrt_llm\')._path; '
        'print(dist_path); '
        'license_dir = os.path.join(dist_path, \'licenses\'); '
        'print(license_dir); '
        'files = os.listdir(license_dir) if os.path.exists(license_dir) else []; '
        'print(f\'Found: {files}\'); '
        'import json; print(json.dumps(files))"',
        shell=True,
        capture_output=True,
        text=True)

    if result.returncode != 0:
        print(f"ERROR: License files check failed!")
        print(result.stdout)
        print(result.stderr)
        exit(1)

    # Parse the output to get the list of files
    output_lines = result.stdout.strip().split('\n')
    try:
        import json
        found_files = json.loads(output_lines[-1])
    except (json.JSONDecodeError, IndexError):
        print(f"ERROR: Could not parse license files from output!")
        print(result.stdout)
        exit(1)

    # Check for missing or unexpected files
    missing = [f for f in expected_files if f not in found_files]
    unexpected = [f for f in found_files if f not in expected_files]

    if missing or unexpected:
        print(f"ERROR: License files mismatch!")
        print(f"Expected: {expected_files}")
        print(f"Found: {found_files}")
        if missing:
            print(f"Missing: {missing}")
        if unexpected:
            print(f"Unexpected: {unexpected}")
        exit(1)

    print(f"✓ License files verified: {', '.join(expected_files)}")


def get_cpython_version():
    python_version = sys.version_info[:]
    assert python_version[0] == 3
    assert python_version[1] in [10, 12]
    return "cp{}{}".format(python_version[0], python_version[1])


def download_wheel(args):
    if not args.wheel_path.startswith(("http://", "https://")):
        args.wheel_path = "https://" + args.wheel_path
    res = requests.get(args.wheel_path)
    if res.status_code != 200:
        print(f"Fail to get the result of {args.wheel_path}")
        exit(1)
    wheel_name = None
    for line in res.text.split("\n"):
        if not line.startswith('<a href="'):
            continue
        name = line.split('"')[1]
        if not name.endswith(".whl"):
            continue
        if get_cpython_version() not in name:
            continue
        wheel_name = name
        break
    if not wheel_name:
        print(f"Fail to get the wheel name of {args.wheel_path}")
        exit(1)
    if args.wheel_path[-1] == "/":
        args.wheel_path = args.wheel_path[:-1]
    wheel_url = f"{args.wheel_path}/{wheel_name}"
    subprocess.check_call("rm *.whl || true", shell=True)
    subprocess.check_call(f"apt-get install -y wget && wget -q {wheel_url}",
                          shell=True)


def install_tensorrt_llm():
    """
    Installs the tensorrt_llm wheel, dynamically creating a torch constraint
    if torch is already installed to prevent it from being replaced.
    """
    print("##########  Install tensorrt_llm package  ##########")

    install_command = "pip3 install tensorrt_llm-*.whl"

    # Always check for an existing torch installation, regardless of OS.
    try:
        print("Checking for existing torch installation...")
        torch_version_result = subprocess.run(
            ['python3', '-c', 'import torch; print(torch.__version__)'],
            capture_output=True,
            text=True,
            check=True)
        torch_version = torch_version_result.stdout.strip()

        if torch_version:
            print(f"Found installed torch version: {torch_version}")
            constraint_filename = "torch-constraint.txt"
            with open(constraint_filename, "w") as f:
                f.write(f"torch=={torch_version}\n")
            print(
                f"Created {constraint_filename} to constrain torch to version {torch_version}."
            )

            # Modify install command to use the constraint
            install_command += f" -c {constraint_filename}"
        else:
            # This case is unlikely if the subprocess call succeeds
            print(
                "Could not determine installed torch version. Installing without constraint."
            )

    except (subprocess.CalledProcessError, FileNotFoundError):
        # This handles cases where python3 fails or 'import torch' raises an error.
        print("Torch is not installed. Proceeding without constraint.")

    # Execute the final installation command
    print(f"Executing command: {install_command}")
    subprocess.check_call(install_command, shell=True)


def create_link_for_models():
    models_root = llm_models_root()
    if not models_root.exists():
        print(f"ERROR: Models root {models_root} does not exist")
        exit(1)
    src_dst_dict = {
        # TinyLlama-1.1B-Chat-v1.0
        f"{models_root}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0":
        f"{os.getcwd()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }

    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)


def test_pip_install():
    parser = argparse.ArgumentParser(description="Check Pip Install")
    parser.add_argument("--wheel_path",
                        type=str,
                        required=False,
                        default="Default",
                        help="The wheel path")
    args = parser.parse_args()

    print("##########  Install required system libs  ##########")
    if not os.path.exists("/usr/local/mpi/bin/mpicc"):
        subprocess.check_call("apt-get -y install libopenmpi-dev", shell=True)

    subprocess.check_call("apt-get -y install libzmq3-dev", shell=True)
    subprocess.check_call("apt-get -y install python3-pip", shell=True)
    subprocess.check_call("pip3 install --ignore-installed pip || true",
                          shell=True)
    subprocess.check_call("pip3 install --ignore-installed setuptools || true",
                          shell=True)
    subprocess.check_call("pip3 install --ignore-installed wheel || true",
                          shell=True)

    download_wheel(args)
    install_tensorrt_llm()

    print("##########  Test import tensorrt_llm  ##########")
    subprocess.check_call(
        'python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"',
        shell=True)
    print("##########  Verify license files  ##########")
    verify_license_files()

    print("##########  Create link for models  ##########")
    create_link_for_models()

    print("##########  Test quickstart example  ##########")
    subprocess.check_call(
        "python3 ../../examples/llm-api/quickstart_example.py", shell=True)


if __name__ == "__main__":
    test_pip_install()
