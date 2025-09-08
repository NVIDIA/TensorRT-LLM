"""
Local implementation of our Python venv testing.
All the bootstrapping is done with our original core work.
"""
import copy
import os
import shlex
import subprocess
import tempfile
import textwrap as tw

from defs.runner_interface import PythonRunnerInterface
from defs.trt_test_alternative import check_call, check_output, exists, makedirs


class PythonVenvRunnerImpl(PythonRunnerInterface):
    """
    Object that understands how to run Python scripts in a virtual environment.
    Local implementation of our runner.

    Args:
        pip_opts (list): Options to pass to pip when installing packages
        venv_dir (str): Path to the virtualenv root directory, or None if this is
                        an externally-built virtualenv
        venv_bin (str): Path to the Python executable to use when running tests
        workspace (str): Path to the test workspace
    """

    def __init__(self, pip_opts, venv_dir, venv_bin, workspace):
        self._venv_bin = venv_bin
        self.venv_dir = venv_dir
        self.pip_opts = pip_opts
        self.venv_exe_name = os.path.basename(venv_bin)
        self.venv_exe_dir = os.path.dirname(venv_bin)
        self.workspace = workspace
        if not exists(self.workspace):
            try:
                makedirs(self.workspace)
            except Exception as e:
                print(f"Error creating workspace directory: {e}")
        self._new_env = os.environ.copy()

    def get_working_directory(self, translate_wsl_path=True):
        """
        Common interface required by RunnerInterface.
        Both TRTExecutableRunner and VirtualenvRunner both have a workspace attribute but they are
        of different types (function vs variable), as a result, this function is introduced instead.
        """
        return self.workspace

    def _get_repro_envvars(self):
        return sorted(os.environ)

    def set_working_directory(self, value):
        """
        Common interface required by RunnerInterface.
        Both TRTExecutableRunner and VirtualenvRunner both have a workspace attribute but they are
        of different types (function vs variable), as a result, this function is introduced instead.
        """
        self.workspace = value

    def _run_internal(self, script, caller, print_script: bool):

        if isinstance(script, bytes):
            script = script.decode()
        if print_script:
            print("Run with Python: {}".format(self._venv_bin))
            print("=== BEGIN SCRIPT ===")
            print(script)
            print("=== END SCRIPT =====")

        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace, exist_ok=True)
        f = tempfile.NamedTemporaryFile(dir=self.workspace,
                                        mode="w",
                                        delete=False)
        f.write(script)
        # On Windows, we have to first close the file before we can read it
        f.close()
        try:
            path = f.name
            out = caller([self._venv_bin, path])
        finally:
            os.remove(f.name)

        return out

    def run(self, body, caller=check_call, print_script=True):
        """Run Python code in a script with a predefined prolog."""
        script = ""
        script += tw.dedent(body)
        return self._run_internal(script, caller, print_script)

    def run_output(self, body):
        """Runs Python code and captures the output in a string."""
        return self.run(body, caller=check_output, print_script=False)

    def run_raw(self, script, caller=check_call, print_script=True):
        """Run Python code without any pre-processing."""
        return self._run_internal(script, caller, print_script)

    def run_cmd(self,
                args,
                caller=check_call,
                env=None,
                print_script=True,
                **kwargs):
        """Call <python-exe> <args> on the command-line (can be used to run Python script files)."""
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace, exist_ok=True)

        call_args = [self._venv_bin] + args
        if env:
            new_env = copy.deepcopy(os.environ)
            new_env.update(env)
        else:
            new_env = os.environ

        if caller.__name__ == 'check_output':
            try:
                result = subprocess.run(call_args,
                                        env=new_env,
                                        check=True,
                                        capture_output=True,
                                        **kwargs)
                return result.stdout.decode('utf-8')
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to run `{shlex.join(e.cmd)}`:\n"
                                   f"Stdout: {e.stdout.decode()}\n"
                                   f"Stderr: {e.stderr.decode()}\n")
        else:
            print(f"Start subprocess with {caller}({call_args}, env={new_env})")
            return caller(call_args, env=new_env, **kwargs)

    def install_packages(self, packages):
        """Install Python packages by name."""

    def install_from_requirements(self, requirements_file):
        pass
