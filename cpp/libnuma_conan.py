import os

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration


class LibnumaSystemConan(ConanFile):
    name = "libnuma"
    version = "system"
    package_type = "shared-library"
    settings = "os", "arch"

    def package_info(self):
        if self.settings.os == "Windows":
            self.output.info("libnuma not needed on Windows.")
            return

        self.cpp_info.includedirs = ["/usr/include"]

        arch = str(self.settings.arch)
        os_name = str(self.settings.os)

        if os_name == "Linux":
            lib_candidates = [
                "/usr/lib64/libnuma.so",  # RHEL/CentOS/Rocky
                "/lib64/libnuma.so",  # possible fallback
            ]
            if arch == "x86_64":
                # Debian/Ubuntu x86_64
                lib_candidates.append("/usr/lib/x86_64-linux-gnu/libnuma.so")
            elif arch in ["armv8", "aarch64"]:
                # Debian/Ubuntu aarch64
                lib_candidates.append("/usr/lib/aarch64-linux-gnu/libnuma.so")
            else:
                self.output.info(
                    f"Unrecognized architecture: {arch}, falling back to /usr/lib/libnuma.so"
                )
                lib_candidates.append("/usr/lib/libnuma.so")
            for lib in lib_candidates:
                if os.path.exists(lib):
                    self.output.info(f"Using libnuma from: {lib}")
                    self.cpp_info.set_property("cmake_link_options", [lib])
                    break
            else:
                raise ConanInvalidConfiguration(
                    "libnuma.so not found on system")
        else:
            raise ConanInvalidConfiguration(f"Unsupported OS: {os_name}")

        self.cpp_info.libdirs = []
        self.cpp_info.system_libs = ["numa"]
