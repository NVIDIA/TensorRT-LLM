# Adding new C++ Dependencies

## Step 1: Make the package available to the build

First, decide if you must install the package in the container or if you
may defer fetching until the build phase. In general, *prefer to fetch
packages during the build phase*. You may be required to install
packages into the container, however, if there is a runtime component
(e.g. shared objects) that cannot be reasonably distributed with the
wheel.

### Install in the container

#### Debian Packages via os package manager (e.g. apt, dnf)

Add your package to one of the existing shell scripts used by the docker build
under [docker/common/][1] Find the location where the package manager is
invoked, and add the name of your package there.

NOTE: Internal compliance tooling will automatically detect the
installation of this package and fetch sources using the source-fetching
facilities of the OS package manager.

[1]: https://github.com/NVIDIA/TensorRT-LLM/tree/main/docker/common.

#### Python Packages via pip

If it makes sense, add your package to one of the existing shell scripts used by
the docker build under [docker/common/][2]. Grep for "pip3 install" to see
existing invocations. If none of the existing shell scripts make sense, add a
new shell script to install your package and then invoke that script in
Dockerfile.multi.

NOTE: If the new python package you are adding has a compiled component (e.g. a
python extension module), you must coordinate with the [Security Team][20] to
ensure that the source for this component is managed correctly.

[2]: https://github.com/NVIDIA/TensorRT-LLM/tree/main/docker/common

#### Tarball packages via HTTP/FTP

Invoke `wget` in a shell script which is called from the docker build file.
When it makes sense, please prefer to extend an existing script in
[docker/common/][3] rather than creating a new one. If you are downloading a
binary package, you must also download the source package that produced that
binary.

Ensure that the source package is copied to /third-party-source and retained
after all cleanup within the docker image layer.

[3]: https://github.com/NVIDIA/TensorRT-LLM/tree/main/docker/common

### Fetch during the build

#### Python Packages via pip

Add an entry to [requirements-dev.txt][4].
The package will be installed by build\_wheel.py during virtual
environment initialization prior to configuring the build with cmake.
Include a comment indicating the intended usage of the package.

[4]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/requirements-dev.txt

**Example:**

`requirements-dev.txt`:

``` requirements.txt
# my-package is needed by <feature> where it is used for <reason>
my-package==1.2.24
```

#### C/C++ Packages via conan

Add a new entry to [conandata.yml][6] indicating the package version for the
dependency you are adding. Include a yaml comment indicating the intended usage
of the package. Then add a new invocation of `self.require()` within the `def
requirements(self)` method of [conanfile.py], referencing the version you added
to conandata.

[6]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/conandata.yml
[7]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/conanfile.py

**Example:**

`conandata.yml`:

```.yml
# my_dependency is needed by <feature> where it is used for <reason>
my_dependency: 1.2.24+1
```

`conanfile.py`:

```.py
def requirements(self):
    ...
    my_dependency_version = self.conandata["my_dependency"]
    self.requires(f"my_dependency/{my_dependency_version}")
```

#### Source integration via CMake

If you have a package you need to build from source then use CMake
[FetchContent][8] of [ExternalProject][9] to fetch the package sources and
integrate it with the build. See the details in the next section.

[8]: https://cmake.org/cmake/help/latest/module/FetchContent.html
[9]: https://cmake.org/cmake/help/latest/module/ExternalProject.html#id1

#### git Submodule - Don't Use

Please *avoid use of git-submodule*. If, for some reason, the CMake integrations
described below don't work and git-submodule is absolutely required, please add
the submodule under the 3rdparty directory.

**Rationale:**

For a source-code dependency distributed via git,
FetchContent/ExternalProject and git submodules both ultimately contain
the same referential information (repository URL, commit sha) and, at
the end of the day, do the same things. However
FetchContent/ExternalProject have the following advantages:

1.  The git operations happen during the build and are interleaved with the rest
    of the build processing, rather than requiring an additional step managed
    outside of CMake.

2.  The fetch, patch, and build steps for the sub project are individually named
    in the build, so any failures are more clearly identified

3.  The build state is better contained within the build tree where it is less
    prone to interference by development actions.

4.  For source code that is modified, FetchContent/ExternalProject can manage
    application of the patches making it clear what modifications are present.

5.  The build does not have to make assumptions about the version control
    configuration of the source tree, which may be incorrect due to the fact
    that it is bind-mounted in a container. For example, `git submodule --init`
    inside a container will corrupt the git configuration outside the container
    if the source tree is a git worktree.

6.  External project references and their patches are collected under a more
    narrow surface, rather than being spread across different tools. This makes
    it easier to track third part dependencies as well as to recognize them
    during code review.

**Example:**

``` bash
git submodule add https://github.com/some-organization/some-project.git 3rdparty/some-project
```


## Step 2: Integrate the package

There are many ways to integrate a package with the build through cmake.

### find\_package for binary packages

For binary packages (os-provided via apt-get or yum, or conan-provided), prefer
the use of [find\_package][10] to integrate the package into the build. Conan
will generate a find-script for packages that don't already come with a Cmake
configuration file and the conan-specific logic is provided through the
conan-generated toolchain already used in our build.

For any packages which do not have provided find modules (either built-in, or
available from conan), please implement one in [cpp/cmake/modules][11]. Please
do not add "direct" invocations of `find_library` / `add_library` / `find_file`
/ `find_path` outside of a find module the package.

Please add invocations of `find_package` directly in the root Cmake file.

[10]: https://cmake.org/cmake/help/latest/command/find_package.html
[11]: https://github.com/NVIDIA/TensorRT-LLM/tree/main//cpp/cmake/modules?ref_type=heads

**Example:**

cpp/CMakeLists.txt

```.cmake
find_package(NIXL)
```

cpp/cmake/modules/FindNIXL.cmake
```.cmake
...
    find_library(
NIXL_LIBRARY nixl
HINTS
    ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH}
           ${NIXL_ROOT}/lib64)
...
    add_library(NIXL::nixl SHARED IMPORTED)
    set_target_properties(
      NIXL::nixl
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NIXL_INCLUDE_DIR}
        IMPORTED_LOCATION ${NIXL_LIBRARY}
    ${NIXL_BUILD_LIBRARY}
${SERDES_LIBRARY}
)
```

### FetchContent for source packages with compatible cmake builds

For source packages that have a compatible cmake (e.g. where add\_subdirectory
will work correctly), please use [FetchContent][12] to download the sources and
integrate them into the build. Please add new invocations of
FetchContent\_Declare in [3rdparty/CMakeLists.txt][13]. Add new invocations for
FetchContent\_MakeAvailable wherever it makes sense in the build where you are
integrating it, but prefer the root listfile for that build
([cpp/CMakeLists.txt][14] for the primary build).

CODEOWNERS for this file will consist of PLC reviewers who verify that
third-party license compliance strategies are being followed.

If the dependency you are adding has modified sources, please do the
following:

1.  Create a repository on gitlab to mirror the upstream source files. If the
    upstream is also in git, please use the gitlab "mirror" repository option.
    Otherwise, please use branches/tags to help identify the upstream source
    versions.

2.  Track nvidia changes in a branch. Use a linear sequence (trunk-based)
    development strategy. Use meaningful, concise commit message subjects and
    comprehensive commit messages for the changes applied.

3.  Use `git format-patch \<upstream-commit\>\...HEAD` to create a list of
    patches, one file per commit,

4.  Add your patches under 3rdparty/patches/\<package-name\>

5.  Use CMake's [PATCH\_COMMAND][15] option to apply the patches during the
    build process.

[12]: https://cmake.org/cmake/help/latest/module/FetchContent.html
[13]: https://github.com/NVIDIA/TensorRT-LLM/tree/main//3rdparty/CMakeLists.txt?ref_type=heads
[14]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/CMakeLists.txt
[15]: https://cmake.org/cmake/help/latest/module/ExternalProject.html#patch-step-options

**Example:**

3rdparty/CMakeLists.txt

```.cmake
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        f99ffd7e03001810a3e722bf48ad1a9e08415d7d
)
```

cpp/CmakeLists.txt

```.cmake
FetchContent_MakeAvailable(pybind11)
```

### ExternalProject

If the package you are adding doesn't support FetchContent (e.g. if it's not
built by CMake or if its CMake configuration doesn't nest well), then please use
[ExternalProject][16]. In this case that project's build system will be invoked
as a build step of the primary build system. Note that, unless both the primary
and child build systems are GNU Make, they will not share a job server and will
independently schedule parallelism (e.g. -j flags).

[16]: https://cmake.org/cmake/help/latest/module/ExternalProject.html#id1

**Example:**

```.cmake
ExternalProject_Add(
  nvshmem_project
  URL https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.2.5_cuda12-archive.tar.xz
  URL_HASH ${NVSHMEM_URL_HASH}
  PATCH_COMMAND patch -p1 --forward --batch -i
                ${DEEP_EP_SOURCE_DIR}/third-party/nvshmem.patch
  ...
  CMAKE_CACHE_ARGS
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER:STRING=${CMAKE_C_COMPILER_LAUNCHER}
  ...
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build
  BUILD_BYPRODUCTS
    ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build/src/lib/libnvshmem.a
)
add_library(nvshmem_project::nvshmem STATIC IMPORTED)
add_dependencies(nvshmem_project::nvshmem nvshmem_project)
...
set_target_properties(
  nvshmem_project::nvshmem
  PROPERTIES IMPORTED_LOCATION
             ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build/src/lib/libnvshmem.a
             INTERFACE_INCLUDE_DIRECTORIES
             ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build/src/include)
```

## Step 3: Update third-party attributions and license tracking

1.  Clone the dependency source code to an NVIDIA-controlled repository. The
    consumed commit must be stored as-received (ensure the consumed commit-sha
    is present in the clone). For sources available via git (or git-adaptable)
    SCM, mirror the repository in the [oss-components][18] gitlab project.

2.  Collect the license text of the consumed commit

3.  If the license does not include a copyright notice, collect any copyright
    notices that were originally published with the dependency (these may be on
    individual file levels, in metadata files, or in packaging control files).

4.  Add the license and copyright notices to the ATTRIBUTIONS-CPP-x86\_64.md and
    ATTRIBUTIONS-CPP-aarch64.md files

CODEOWNERS for ATTRIBUTIONS-CPP-\*.md are members of the PLC team and modifying
this file will signal to reviewers that they are verifying that your change
follows the process in this document.

[18]: https://gitlab.com/nvidia/tensorrt-llm/oss-components

## Step 4: File a JIRA ticket if you need help from the Security team

This step is optional, if you need assistance from the Security team.

File a Jira ticket using the issue template [TRTLLM-8383][19] to request
inclusion of this new dependency and initiate license and/or security review.
The Security Team will triage and assign the ticket.

If you don’t have access to the JIRA project, please email the [Security
Team][20].


[19]: https://jirasw.nvidia.com/browse/TRTLLM-8383
[20]: mailto://TensorRT-LLM-Security@nvidia.com
