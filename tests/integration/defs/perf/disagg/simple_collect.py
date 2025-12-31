#!/usr/bin/env python3
"""System Information Collection Script.

This script collects system information and generates:
1. session_properties.csv - Comprehensive system information
2. gpu.txt - GPU model name
3. cpu.txt - CPU model name
4. driver.txt - GPU driver version

Usage: python simple_collect.py [output_dir]
"""

import csv
import os
import platform
import re
import socket
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime


def collect_system_info():
    """Collect all system information and return as dictionary."""
    print("=== Collecting System Information ===")

    data = {}
    # Try multiple ways to get username
    username = os.getenv("USER") or os.getenv("USERNAME") or os.getenv("LOGNAME") or "unknown"
    data["username"] = username
    data["start_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["hostname"] = socket.gethostname()

    # IP address
    try:
        data["ip"] = socket.gethostbyname(data["hostname"])
    except Exception:
        data["ip"] = "unknown"

    # NVIDIA information
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,pci.device_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")
        if lines and lines[0]:
            parts = lines[0].split(", ")
            data["nvidia_driver_version"] = parts[0] if len(parts) > 0 else "unknown"
            gpu_name = parts[1] if len(parts) > 1 else "unknown"
            pci_device_id = parts[2] if len(parts) > 2 else "unknown"
            data["nvidia_device_count"] = len(lines)

            # Convert PCI device ID to integer if possible
            try:
                pci_device_id_int = (
                    int(pci_device_id, 16) if pci_device_id != "unknown" else "unknown"
                )
            except Exception:
                pci_device_id_int = "unknown"

            data["gpu_properties"] = str(
                OrderedDict(
                    [("device_product_name", gpu_name), ("pci_device_id", pci_device_id_int)]
                )
            )
        else:
            raise Exception("No GPU data")
    except Exception as e:
        print(f"NVIDIA info error: {e}")
        data["nvidia_driver_version"] = "unknown"
        data["nvidia_device_count"] = 0
        data["gpu_properties"] = str(OrderedDict([]))

    # OS information
    data["os_properties"] = str(
        OrderedDict([("name", platform.system()), ("version", platform.version())])
    )

    # CPU information
    cpu_model = "unknown"
    cpu_freq_info = OrderedDict([("current", "unknown"), ("min", "unknown"), ("max", "unknown")])

    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, check=True)
        match = re.search(r"Model name:\s*(.+)", result.stdout)
        if match:
            cpu_model = match.group(1).strip()
    except Exception:
        cpu_model = platform.processor() or "unknown"

    # Try to get CPU frequency information
    try:
        import psutil

        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            cpu_freq_info = OrderedDict(
                [("current", cpu_freq.current), ("min", cpu_freq.min), ("max", cpu_freq.max)]
            )
    except ImportError:
        # psutil not available, try to get from /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                # Try to extract current frequency from first processor
                match = re.search(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
                if match:
                    current_freq = float(match.group(1))
                    cpu_freq_info["current"] = current_freq
        except Exception:
            pass
    except Exception:
        pass

    data["cpu_properties"] = str(
        OrderedDict(
            [("cpu_count", os.cpu_count()), ("cpu_model", cpu_model), ("cpu_freq", cpu_freq_info)]
        )
    )

    # CUDA version
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        match = re.search(r"release (\d+\.\d+)", result.stdout)
        data["cuda_version"] = match.group(1) if match else "unknown"
    except Exception:
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
            data["cuda_version"] = match.group(1) if match else "unknown"
        except Exception:
            data["cuda_version"] = "unknown"

    # Other fields
    data["cublas_version"] = "unknown"
    data["cudnn_version"] = "unknown"
    data["trt_change_id"] = ""
    data["trt_branch"] = ""
    data["commit_timestamp"] = ""
    data["end_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return data


class SessionPropertiesWriter:
    """Writer class for session properties CSV file."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.csv_file = os.path.join(output_dir, "session_properties.csv")

    def write_csv(self, data):
        """Write system information to CSV file."""
        print(f"Writing CSV to: {self.csv_file}")

        fieldnames = [
            "username",
            "start_timestamp",
            "hostname",
            "ip",
            "nvidia_driver_version",
            "nvidia_device_count",
            "os_properties",
            "cpu_properties",
            "gpu_properties",
            "trt_change_id",
            "trt_branch",
            "commit_timestamp",
            "cuda_version",
            "cublas_version",
            "cudnn_version",
            "end_timestamp",
        ]

        with open(self.csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data)

        print(f"CSV file generated: {self.csv_file}")


class TextWriter:
    """Writer class for individual text files (gpu.txt, cpu.txt, driver.txt)."""

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def write_gpu_info(self, data):
        """Write GPU information to gpu.txt."""
        gpu_name = "unknown"
        try:
            # Extract GPU name from gpu_properties
            gpu_props = data.get("gpu_properties", "unknown")
            # Parse the OrderedDict string to extract device_product_name
            # Format: OrderedDict({'device_product_name': 'value', ...})
            match = re.search(r"'device_product_name':\s*'([^']+)'", gpu_props)
            if match:
                gpu_name = match.group(1).replace("_", " ")  # Replace underscores with spaces
        except Exception:
            pass

        gpu_file = os.path.join(self.output_dir, "gpu.txt")
        with open(gpu_file, "w") as f:
            f.write(gpu_name)
        print(f"Generated GPU file: {gpu_file}")
        return gpu_name

    def write_cpu_info(self, data):
        """Write CPU information to cpu.txt."""
        cpu_model = "unknown"
        try:
            # Extract CPU model from cpu_properties
            cpu_props = data.get("cpu_properties", "unknown")
            # Parse the OrderedDict string to extract cpu_model
            # Format: OrderedDict({'cpu_count': N, 'cpu_model': 'value', ...})
            match = re.search(r"'cpu_model':\s*'([^']+)'", cpu_props)
            if match:
                cpu_model = match.group(1)
        except Exception:
            pass

        cpu_file = os.path.join(self.output_dir, "cpu.txt")
        with open(cpu_file, "w") as f:
            f.write(cpu_model)
        print(f"Generated CPU file: {cpu_file}")
        return cpu_model

    def write_driver_info(self, data):
        """Write GPU driver information to driver.txt."""
        driver_version = data.get("nvidia_driver_version", "unknown")

        driver_file = os.path.join(self.output_dir, "driver.txt")
        with open(driver_file, "w") as f:
            f.write(driver_version)
        print(f"Generated driver file: {driver_file}")
        return driver_version

    def write_all_txt_files(self, data):
        """Write all text files and return their contents for display."""
        gpu_info = self.write_gpu_info(data)
        cpu_info = self.write_cpu_info(data)
        driver_info = self.write_driver_info(data)

        return {
            "GPU": gpu_info,
            "CPU": cpu_info,
            "Driver": driver_info,
        }


def main():
    """Main entry point for the script."""
    # Determine output directory
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "."

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    # Collect system information
    system_data = collect_system_info()

    # Write CSV file
    csv_writer = SessionPropertiesWriter(output_dir)
    csv_writer.write_csv(system_data)

    # Write text files
    txt_writer = TextWriter(output_dir)
    txt_contents = txt_writer.write_all_txt_files(system_data)

    # Display summary
    print("\n=== Collection Summary ===")
    print("Generated files:")
    print("  - session_properties.csv")
    print("  - gpu.txt")
    print("  - cpu.txt")
    print("  - driver.txt")

    print("\n=== Collected Information ===")
    for key, value in system_data.items():
        print(f"{key}: {value}")

    print("\n=== Text Files Content ===")
    for file_type, content in txt_contents.items():
        print(f"{file_type}: {content}")

    print(f"\nAll files written to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
