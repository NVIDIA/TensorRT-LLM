from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class TensorRT_LLM(ConanFile):
    name = "TensorRT-LLM"
    settings = "os", "arch", "compiler", "build_type"
    virtualbuildenv = False
    virtualrunenv = False

    def requirements(self):
        pass  # TODO add dependencies here

    def generate(self):
        cmake = CMakeDeps(self)
        cmake.generate()
        tc = CMakeToolchain(self)
        tc.generate()
