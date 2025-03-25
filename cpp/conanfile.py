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
        tc.variables["NVRTC_WRAPPER_LIB_SOURCE_REL_LOC"] = self.dependencies[
            "tensorrt_llm_nvrtc_wrapper"].package_folder + "/lib/libtensorrt_llm_nvrtc_wrapper.so"
        tc.generate()
