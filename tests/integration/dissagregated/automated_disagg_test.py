#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time

import psutil
import requests

from tensorrt_llm.logger import logger


class DisaggregatedTester:

    def __init__(self, config):
        self.config = config
        self.processes = {}
        self.all_pids = []

    def start_context_server(self, gpu_id: int, port: int) -> int:
        """Start a context server on specified GPU and port."""
        cmd = [
            "trtllm-serve", self.config['model_path'], "--host", "localhost",
            "--port",
            str(port), "--backend", "pytorch", "--extra_llm_api_options",
            self.config['extra_llm_api_path'], "--metadata_server_config_file",
            self.config['etcd_config_path'], "--server_role", "CONTEXT"
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        logger.info(f"Starting CONTEXT server on GPU {gpu_id} (port {port})...")
        process = subprocess.Popen(cmd,
                                   env=env,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   bufsize=1)
        self.processes[f"context_{port}"] = process
        self.all_pids.append(process.pid)
        return process.pid

    def start_generation_server(self, gpu_id: int, port: int) -> int:
        """Start a generation server on specified GPU and port."""
        cmd = [
            "trtllm-serve", self.config['model_path'], "--host", "localhost",
            "--port",
            str(port), "--backend", "pytorch", "--extra_llm_api_options",
            self.config['extra_llm_api_path'], "--metadata_server_config_file",
            self.config['etcd_config_path'], "--server_role", "GENERATION"
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        logger.info(
            f"Starting GENERATION server on GPU {gpu_id} (port {port})...")
        process = subprocess.Popen(cmd,
                                   env=env,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   bufsize=1)
        self.processes[f"generation_{port}"] = process
        self.all_pids.append(process.pid)
        return process.pid

    def start_disaggregated_service(self) -> int:
        """Launch the disaggregated service."""
        cmd = [
            "trtllm-serve", "disaggregated", "-c",
            self.config['disagg_config_path'], "-m",
            self.config['etcd_config_path']
        ]

        logger.info("Launching disaggregated service...")
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   bufsize=1)
        self.processes["disagg_service"] = process
        self.all_pids.append(process.pid)
        return process.pid

    def wait_for_server_health(self, port: int, timeout: int = 120) -> bool:
        """Wait for server to be healthy by checking /health endpoint."""
        url = f"http://localhost:{port}/health"
        start_time = time.time()
        logger.info(f"Waiting for server on port {port} to be healthy...")

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Server on port {port} is healthy")
                    return True
            except requests.RequestException:
                pass

            # Check if process is still running
            process_key = next((k for k in self.processes if str(port) in k),
                               None)
            if process_key and self.processes[process_key].poll() is not None:
                logger.error(
                    f"Server process on port {port} exited prematurely with code {self.processes[process_key].returncode}"
                )
                stderr = self.processes[process_key].stderr.read()
                logger.error(f"Error output: {stderr}")
                return False

            time.sleep(2)

        logger.error(f"Timed out waiting for server on port {port}")
        return False

    def wait_for_disagg_service_health(self,
                                       port: int = 8000,
                                       timeout: int = 120) -> bool:
        """Wait for disaggregated service to be healthy."""
        url = f"http://localhost:{port}/health"
        start_time = time.time()
        logger.info(
            f"Waiting for disaggregated service on port {port} to be healthy..."
        )

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(
                        f"Disaggregated service on port {port} is healthy")
                    return True
            except requests.RequestException:
                pass

            # Check if process is still running
            if self.processes["disagg_service"].poll() is not None:
                logger.error(
                    f"Disaggregated service exited prematurely with code {self.processes['disagg_service'].returncode}"
                )
                stderr = self.processes["disagg_service"].stderr.read()
                logger.error(f"Error output: {stderr}")
                return False

            time.sleep(2)

        logger.error(
            f"Timed out waiting for disaggregated service on port {port}")
        return False

    def run_client_test(self) -> bool:
        """Run the disaggregated client test."""
        cmd = [
            "python3", f"{self.config['client_script_path']}", "-c",
            self.config['disagg_config_path'], "-p", self.config['prompts_path']
        ]

        logger.info("Running disaggregated client test...")
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)

        if result.returncode == 0:
            logger.info("Client test succeeded")
            logger.info(f"Client output: {result.stdout}")
            return True
        else:
            logger.error(
                f"Client test failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            logger.error(f"Standard output: {result.stdout}")
            return False

    def kill_server(self, port: int) -> bool:
        """Kill the server running on specified port."""
        logger.info(f"Killing server running on port {port}...")

        # Find the process by port
        process_key = next((k for k in self.processes if str(port) in k), None)
        if not process_key:
            logger.warning(
                f"No process found for port {port} in tracked processes")
            return self.kill_server_by_port(port)

        # Kill the process
        process = self.processes[process_key]
        if process.poll() is None:  # Still running
            logger.info(f"Terminating process {process.pid}")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Process {process.pid} did not terminate gracefully, force killing..."
                )
                process.kill()

            logger.info(f"Process on port {port} killed")
            return True
        else:
            logger.warning(
                f"Process already exited with code {process.returncode}")
            return False

    def kill_server_by_port(self, port: int) -> bool:
        """Find and kill a process by port using lsof."""
        try:
            # Find PID using port
            cmd = ["lsof", "-t", f"-i:{port}"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)

            if result.stdout.strip():
                pid = int(result.stdout.strip())
                os.kill(pid, signal.SIGKILL)
                logger.info(f"Killed process {pid} on port {port}")
                return True
            else:
                logger.warning(f"No process found on port {port}")
                return False
        except Exception as e:
            logger.error(f"Error killing process on port {port}: {e}")
            return False

    def cleanup(self):
        """Kill all started processes."""
        logger.info("Cleaning up all processes...")

        for name, process in self.processes.items():
            if process.poll() is None:  # Still running
                logger.info(f"Terminating {name} (PID: {process.pid})")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    logger.warning(f"Force killing {name} (PID: {process.pid})")
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass

        # Double-check with psutil to make sure child processes are also killed
        for pid in self.all_pids:
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)

                for child in children:
                    logger.info(f"Killing child process {child.pid}")
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass

            except psutil.NoSuchProcess:
                pass

    def run_complete_test(self) -> bool:
        """Run the complete automated test."""
        try:
            # Start initial servers
            self.start_context_server(gpu_id=0, port=8001)
            self.start_generation_server(gpu_id=1, port=8002)

            # Wait for initial servers to be healthy
            if not self.wait_for_server_health(port=8001):
                return False
            if not self.wait_for_server_health(port=8002):
                return False

            # Start disaggregated service
            self.start_disaggregated_service()

            # Wait for disaggregated service
            if not self.wait_for_disagg_service_health():
                return False

            # Start second context server
            self.start_context_server(gpu_id=2, port=8003)

            # Wait for second context server
            if not self.wait_for_server_health(port=8003):
                return False

            # Run the first client test
            first_test_success = self.run_client_test()
            if not first_test_success:
                logger.error("First client test failed")
                return False

            # Kill the first context server
            self.kill_server(port=8001)

            # Wait a moment for the service to recognize the server is gone
            logger.info(
                "Waiting for the service to recognize the server removal...")
            time.sleep(20)

            # Run the second client test
            second_test_success = self.run_client_test()
            if not second_test_success:
                logger.error("Second client test failed")
                return False

            logger.info("Both client tests passed successfully!")
            return True

        except Exception as e:
            logger.exception(f"Error during test: {e}")
            return False
        finally:
            self.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Disaggregated Testing")
    parser.add_argument("--model-path",
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Path to the model")
    parser.add_argument(
        "--extra-llm-api-path",
        default="examples/disaggregated/extra-llm-api-config.yml",
        help="Path to extra LLM API config")
    parser.add_argument("--etcd-config-path",
                        default="examples/disaggregated/etcd_config.yaml",
                        help="Path to etcd config")
    parser.add_argument("--disagg-config-path",
                        default="examples/disaggregated/disagg_config.yaml",
                        help="Path to disaggregation config")
    parser.add_argument(
        "--client-script-path",
        default="examples/disaggregated/clients/disagg_client.py",
        help="Path to client script")
    parser.add_argument("--prompts-path",
                        default="examples/disaggregated/clients/prompts.json",
                        help="Path to prompts file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = {
        "model_path": args.model_path,
        "extra_llm_api_path": args.extra_llm_api_path,
        "etcd_config_path": args.etcd_config_path,
        "disagg_config_path": args.disagg_config_path,
        "client_script_path": args.client_script_path,
        "prompts_path": args.prompts_path
    }

    tester = DisaggregatedTester(config)
    success = tester.run_complete_test()

    # Exit with appropriate status code
    sys.exit(0 if success else 1)
