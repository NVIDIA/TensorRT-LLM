"""
    File that holds interfaces for runners.
    Needs to be in this file to prevent cyclic loops.
"""

from abc import ABC, abstractmethod


class RunnerInterface(ABC):
    """Extend this class to use with runner related functions such as misc.temp_wd"""

    @abstractmethod
    def get_working_directory(self):
        """
            Since workspace is a function in TRTExectuableRunner and workspace is a variable in VritualenvRunner we introduce this particular
            function as a common interface between the two runners.
        """

    @abstractmethod
    def set_working_directory(self, value):
        pass

    @abstractmethod
    def run(self):
        pass


class PythonRunnerInterface(RunnerInterface, ABC):
    """
    A public interface for runners working with Python based tests.
    Implementation is dependent on local and remote modes and so implementations
    can be found under misc/venv_runner_remote.py and misc/venv_runner_local.py
    """

    @abstractmethod
    def install_from_requirements(self, requirements_file):
        """Install Python packages listed in the given requirements file."""

    @abstractmethod
    def install_packages(self, packages):
        """Install Python packages by name."""

    @abstractmethod
    def run_cmd(self, args, caller, env, print_script):
        """Call <python-exe> <args> on the command-line (can be used to run Python script files)."""

    @abstractmethod
    def run_raw(self, script, caller, print_script):
        """Run Python code without any pre-processing."""

    @abstractmethod
    def run_output(self, body):
        """Runs Python code and captures the output in a string."""

    @abstractmethod
    def run(self, body, caller, print_script):
        """Run Python code in a script with a predefined prolog."""
