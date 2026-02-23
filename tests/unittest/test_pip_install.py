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


def get_wheel_url(wheel_path):
    """Get direct wheel URL from wheel_path (directory listing or direct URL)."""
    if not wheel_path.startswith(("http://", "https://")):
        wheel_path = "https://" + wheel_path

    if wheel_path.endswith(".whl"):
        return wheel_path

    res = requests.get(wheel_path)
    if res.status_code != 200:
        print(f"Fail to get the result of {wheel_path}")
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
        print(f"Fail to get the wheel name of {wheel_path}")
        exit(1)
    if wheel_path[-1] == "/":
        wheel_path = wheel_path[:-1]
    return f"{wheel_path}/{wheel_name}"


def download_wheel(args):
    wheel_url = get_wheel_url(args.wheel_path)
    subprocess.check_call("rm *.whl || true", shell=True)
    subprocess.check_call(f"apt-get install -y wget && wget -q {wheel_url}",
                          shell=True)


def get_torch_constraint_file(constraint_dir="."):
    """Create a torch constraint file if torch is already installed.

    This prevents pip from changing the pre-installed PyTorch version,
    which could cause ABI incompatibility issues.

    Args:
        constraint_dir: Directory to create the constraint file in.

    Returns:
        Path to constraint file if torch is installed, None otherwise.
    """
    try:
        torch_version_result = subprocess.run(
            ['python3', '-c', 'import torch; print(torch.__version__)'],
            capture_output=True,
            text=True,
            check=True)
        torch_version = torch_version_result.stdout.strip()
        if torch_version:
            print(f"Found installed torch version: {torch_version}")
            constraint_file = os.path.join(constraint_dir,
                                           "torch-constraint.txt")
            with open(constraint_file, "w") as f:
                f.write(f"torch=={torch_version}\n")
            print(f"Created {constraint_file} to constrain torch version.")
            return constraint_file
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Torch is not installed. Proceeding without constraint.")
    return None


def install_tensorrt_llm():
    """Install the tensorrt_llm wheel with torch version constraint."""
    print("##########  Install tensorrt_llm package  ##########")

    install_command = "pip3 install tensorrt_llm-*.whl"
    constraint_file = get_torch_constraint_file()
    if constraint_file:
        install_command += f" -c {constraint_file}"

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


def run_sanity_check(examples_path="../../examples"):
    """Run sanity checks after installation."""
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
        f"python3 {examples_path}/llm-api/quickstart_example.py", shell=True)


def install_system_libs():
    """Install required system libraries for tensorrt_llm."""
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


def test_pip_install(args):
    install_system_libs()

    download_wheel(args)
    install_tensorrt_llm()

    run_sanity_check()


def test_python_builds(args):
    """Test Python builds using precompiled wheel (sanity check only).

    This test verifies the TRTLLM_PRECOMPILED_LOCATION workflow:
    1. Install required system libs
    2. Use precompiled wheel URL to extract C++ bindings
    3. Build Python-only wheel (editable install with torch constraint)
    4. Verify installation works correctly
    5. Run quickstart example
    6. Clean up editable install to leave env in clean state
    """
    print("##########  Python Builds Test  ##########")

    install_system_libs()

    wheel_url = get_wheel_url(args.wheel_path)
    print(f"Using precompiled wheel: {wheel_url}")

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    print(f"Repository root: {repo_root}")

    # Uninstall existing tensorrt_llm to test fresh editable install
    subprocess.run("pip3 uninstall -y tensorrt_llm", shell=True, check=False)

    print("##########  Install with TRTLLM_PRECOMPILED_LOCATION  ##########")
    env = os.environ.copy()
    env["TRTLLM_PRECOMPILED_LOCATION"] = wheel_url

    install_cmd = ["pip3", "install", "-e", ".", "-v"]
    constraint_file = get_torch_constraint_file(repo_root)
    if constraint_file:
        install_cmd.extend(["-c", constraint_file])

    subprocess.check_call(install_cmd, cwd=repo_root, env=env)
    run_sanity_check(examples_path=f"{repo_root}/examples")

    # Clean up: uninstall editable install to leave env in clean state
    print("##########  Clean up editable install  ##########")
    subprocess.run("pip3 uninstall -y tensorrt_llm", shell=True, check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Pip Install")
    parser.add_argument("--wheel_path",
                        type=str,
                        required=True,
                        help="The wheel path")
    args = parser.parse_args()
    test_python_builds(args)
    test_pip_install(args)
