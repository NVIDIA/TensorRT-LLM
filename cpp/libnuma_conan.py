from conan import ConanFile


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
        libdirs = []

        arch = str(self.settings.arch)
        os_name = str(self.settings.os)

        if os_name == "Linux":
            if arch == "x86_64":
                libdirs.append("/usr/lib/x86_64-linux-gnu")
            elif arch in ["armv8", "aarch64"]:
                libdirs.append("/usr/lib/aarch64-linux-gnu")
            else:
                self.output.warn(
                    f"Unrecognized architecture: {arch}, falling back to /usr/lib"
                )
                libdirs.append("/usr/lib")
        else:
            self.output.warn(f"Unsupported OS: {os_name}, assuming /usr/lib")
            libdirs.append("/usr/lib")

        self.cpp_info.libdirs = libdirs
        self.cpp_info.system_libs = ["numa"]
