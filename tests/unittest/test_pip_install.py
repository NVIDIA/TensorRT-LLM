import argparse
import os
import subprocess
import sys

import requests


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
    subprocess.check_call("pip3 install --upgrade pip || true", shell=True)
    subprocess.check_call("pip3 install --upgrade setuptools || true",
                          shell=True)

    download_wheel(args)
    print("##########  Install tensorrt_llm package  ##########")
    subprocess.check_call("pip3 install tensorrt_llm-*.whl", shell=True)
    print("##########  Test import tensorrt_llm  ##########")
    subprocess.check_call(
        'python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"',
        shell=True)
    print("##########  Verify license files in installed package  ##########")
    result = subprocess.run(
        'python3 -c "import tensorrt_llm, os; '
        'dist_info = [d for d in os.listdir(os.path.dirname(tensorrt_llm.__file__) + \'/../\') if d.endswith(\'.dist-info\')][0]; '
        'license_dir = os.path.join(os.path.dirname(tensorrt_llm.__file__), \'../\', dist_info, \'licenses\'); '
        'files = os.listdir(license_dir) if os.path.exists(license_dir) else []; '
        'expected = [\'LICENSE\', \'ATTRIBUTIONS-CPP.md\']; '
        'missing = [f for f in expected if f not in files]; '
        'print(f\'Found: {files}\'); '
        'exit(1 if missing else 0)"',
        shell=True,
        capture_output=True,
        text=True)
    if result.returncode != 0:
        print(f"ERROR: License files check failed!")
        print(result.stdout)
        exit(1)
    print("License files verified: LICENSE, ATTRIBUTIONS-CPP.md")
    print("##########  Test quickstart example  ##########")
    subprocess.check_call(
        "python3 ../../examples/llm-api/quickstart_example.py", shell=True)


if __name__ == "__main__":
    test_pip_install()
