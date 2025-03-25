from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class TensorRT_LLM(ConanFile):
    name = "TensorRT-LLM"
    settings = "os", "arch", "compiler", "build_type"
    virtualbuildenv = False
    virtualrunenv = False

    def requirements(self):
        self.requires(
            f"tensorrt_llm_nvrtc_wrapper/{self.conan_data['tensorrt_llm_nvrtc_wrapper']}"
        )

    def generate(self):
        cmake = CMakeDeps(self)
        cmake.generate()
        tc = CMakeToolchain(self)
        tc.blocks.enabled("find_paths")
        tc.generate()
