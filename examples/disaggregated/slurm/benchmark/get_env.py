#!/usr/bin/env python3
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Get TensorRT-LLM environment variables and save to JSON"
    )
    parser.add_argument("-e", "--env-file", required=True, help="Environment file path")
    args = parser.parse_args()

    # read env file, append new envs to it
    with open(args.env_file, "r") as f:
        env_data = json.load(f)

    # Get environment variables
    new_env_data = {
        "TRT_LLM_GIT_COMMIT": os.environ.get("TRT_LLM_GIT_COMMIT", ""),
        "TRT_LLM_VERSION": os.environ.get("TRT_LLM_VERSION", ""),
    }
    print(f"Environment variables: {new_env_data}")
    env_data.update(new_env_data)
    # Save to environment file
    with open(args.env_file, "w") as f:
        json.dump(env_data, f, indent=2)

    print(f"Environment variables saved to {args.env_file}")


if __name__ == "__main__":
    main()
