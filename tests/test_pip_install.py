import argparse
import subprocess

import requests


def download_wheel(args):
    if not args.wheel_path.startswith(('http://', 'https://')):
        args.wheel_path = 'https://' + args.wheel_path
    res = requests.get(args.wheel_path)
    if res.status_code != 200:
        print(f"Fail to get the result of {args.wheel_path}")
        exit(1)
    wheel_name = None
    for line in res.text.split("\n"):
        if not line.startswith("<a href=\""):
            continue
        name = line.split('"')[1]
        if not name.endswith(".whl"):
            continue
        wheel_name = name
        break
    if not wheel_name:
        print(f"Fail to get the wheel name of {args.wheel_path}")
        exit(1)
    if args.wheel_path[-1] == '/':
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
    subprocess.check_call(f"apt-get -y install python3-pip libopenmpi-dev",
                          shell=True)
    download_wheel(args)
    print("##########  Install tensorrt_llm package  ##########")
    subprocess.check_call("pip3 install tensorrt_llm-*.whl", shell=True)
    print("##########  Test import tensorrt_llm  ##########")
    subprocess.check_call('python3 -c "import tensorrt_llm"', shell=True)
    print("##########  Test quickstart example  ##########")
    subprocess.check_call("python3 ../examples/llm-api/quickstart_example.py",
                          shell=True)


if __name__ == '__main__':
    test_pip_install()
