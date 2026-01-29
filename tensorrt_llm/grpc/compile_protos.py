# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile Protocol Buffer definitions for TensorRT-LLM gRPC server.

This script generates Python bindings from the trtllm_service.proto file.

Usage:
    python -m tensorrt_llm.grpc.compile_protos

Or directly:
    python compile_protos.py

Requirements:
    pip install grpcio-tools
"""

import subprocess  # nosec B404
import sys
from pathlib import Path


def compile_protos(proto_dir: Path = None, output_dir: Path = None) -> bool:
    """Compile proto files to Python.

    Args:
        proto_dir: Directory containing .proto files. Defaults to this script's directory.
        output_dir: Directory for generated Python files. Defaults to proto_dir.

    Returns:
        True if compilation succeeded, False otherwise.
    """
    if proto_dir is None:
        proto_dir = Path(__file__).parent
    if output_dir is None:
        output_dir = proto_dir

    proto_file = proto_dir / "trtllm_service.proto"

    if not proto_file.exists():
        print(f"Error: Proto file not found: {proto_file}")
        return False

    # Check for grpcio-tools
    try:
        from grpc_tools import protoc
    except ImportError:
        print("grpcio-tools not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "grpcio-tools"])
        from grpc_tools import protoc

    # Compile proto file
    print(f"Compiling {proto_file}...")

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file),
        ]
    )

    if result != 0:
        print(f"Error: protoc failed with code {result}")
        return False

    # Fix imports in generated files (grpc_tools generates absolute imports)
    pb2_file = output_dir / "trtllm_service_pb2.py"
    pb2_grpc_file = output_dir / "trtllm_service_pb2_grpc.py"

    if pb2_grpc_file.exists():
        content = pb2_grpc_file.read_text()
        # Fix import to use relative import
        content = content.replace(
            "import trtllm_service_pb2 as trtllm__service__pb2",
            "from . import trtllm_service_pb2 as trtllm__service__pb2",
        )
        pb2_grpc_file.write_text(content)
        print(f"Fixed imports in {pb2_grpc_file}")

    print("Generated files:")
    print(f"  - {pb2_file}")
    print(f"  - {pb2_grpc_file}")
    print("Proto compilation successful!")

    return True


def verify_generated_files(output_dir: Path = None) -> bool:
    """Verify that generated files can be imported.

    Args:
        output_dir: Directory containing generated files.

    Returns:
        True if files can be imported, False otherwise.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent

    pb2_file = output_dir / "trtllm_service_pb2.py"
    pb2_grpc_file = output_dir / "trtllm_service_pb2_grpc.py"

    if not pb2_file.exists() or not pb2_grpc_file.exists():
        print("Generated files not found. Run compile_protos() first.")
        return False

    # Try to import
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location("trtllm_service_pb2", pb2_file)
        pb2_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pb2_module)
        print(f"Successfully imported {pb2_file.name}")

        # Verify key message types exist
        assert hasattr(pb2_module, "GenerateRequest")
        assert hasattr(pb2_module, "GenerateResponse")
        assert hasattr(pb2_module, "SamplingConfig")
        assert hasattr(pb2_module, "OutputConfig")
        print("  - GenerateRequest, GenerateResponse, SamplingConfig, OutputConfig OK")

    except Exception as e:
        print(f"Error importing {pb2_file.name}: {e}")
        return False

    print("Verification successful!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile TensorRT-LLM gRPC protos")
    parser.add_argument(
        "--proto-dir", type=Path, default=None, help="Directory containing .proto files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Directory for generated Python files"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify generated files after compilation"
    )

    args = parser.parse_args()

    success = compile_protos(args.proto_dir, args.output_dir)

    if success and args.verify:
        success = verify_generated_files(args.output_dir)

    sys.exit(0 if success else 1)
