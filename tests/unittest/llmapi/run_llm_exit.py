import os
import subprocess
import sys

import click


@click.command()
@click.option("--model_dir", type=str, required=True)
@click.option("--tp_size", type=int, required=True)
def main(model_dir: str, tp_size: int):
    run_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_llm.py"),
        "--model_dir",
        model_dir,
        "--tp_size",
        str(tp_size),
        "--prompt",
        "This is an over-long prompt that intentionlly trigger failure. " *
        1000,
    ]
    # Will raise TimeoutExpired exception if timeout
    res = subprocess.run(run_cmd, check=False, timeout=600)
    assert res.returncode != 0


if __name__ == '__main__':
    main()
