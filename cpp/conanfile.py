import os
import sys

from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class TensorRT_LLM(ConanFile):
    name = "TensorRT-LLM"
    settings = "os", "arch", "compiler", "build_type"
    virtualbuildenv = False
    virtualrunenv = False

    def requirements(self):
        self.requires("libnuma/system")

    def generate(self):
        cmake = CMakeDeps(self)
        cmake.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build_requirements(self):
        # register libnuma_conan.py for conan
        base_dir = os.path.dirname(os.path.abspath(__file__))
        libnuma_path = os.path.join(base_dir, "libnuma_conan.py")
        conan_bin = os.path.abspath(sys.argv[0])
        if not os.path.isfile(conan_bin) or not os.access(conan_bin, os.X_OK):
            raise RuntimeError(f"Conan binary not found {sys.argv[0]}")

        self.run(
            f"{conan_bin} export {libnuma_path} --name=libnuma --version=system"
        )
