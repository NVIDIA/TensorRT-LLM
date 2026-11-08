# A helper script to detect unused NVSHMEM object files.
#
# The script links NVSHMEM to DeepEP with one object file removed at a time and
# checks whether there are any undefined symbols. See README.md for details.
# This script is not tested or QA'ed, so you may need to update this script if
# the project structure changes or compilation options change.
import pathlib
import re
import subprocess

project_dir = pathlib.Path(__file__).parent.parent.parent.parent

# Run `find cpp/build | grep kernels/internode_ll.cu.o$` to get the directory
deep_ep_obj_dir = project_dir / "cpp/build/tensorrt_llm/deep_ep/CMakeFiles/deep_ep_cpp_tllm.dir/__/__/_deps/deep_ep_download-src/csrc"
assert deep_ep_obj_dir.is_dir()

# Run `find cpp/build | grep host/bootstrap/bootstrap.cpp.o$` to get the directory
# Please set it to `nvshmem.dir` rather than `nvshmem_host.dir`
nvshmem_obj_dir = project_dir / "cpp/build/tensorrt_llm/deep_ep/nvshmem-build/src/CMakeFiles/nvshmem.dir"
assert nvshmem_obj_dir.is_dir()

# Parse the `-gencode` arguments
with (project_dir /
      "cpp/build/tensorrt_llm/deep_ep/cuda_architectures.txt").open() as f:
    cuda_architectures = f.read()
pattern = re.compile(r'^([1-9][0-9]*[0-9][af]?)(-real|-virtual)?$')
gencode_args = []
for cuda_arch in cuda_architectures.split(";"):
    matches = re.match(pattern, cuda_arch)
    assert matches is not None, f"Invalid cuda arch \"{cuda_arch}\""
    sm_version = matches.group(1)
    postfix = matches.group(2) or ""
    code = {
        "": f"[compute_{sm_version},sm_{sm_version}]",
        "-real": f"[sm_{sm_version}]",
        "-virtual": f"[compute_{sm_version}]",
    }[postfix]
    gencode_args.append(f"-gencode=arch=compute_{sm_version},{code=:s}")

temp_dir = project_dir / "cpp/build/tensorrt_llm/deep_ep/strip_nvshmem_helper"
temp_dir.mkdir(exist_ok=True)
ranlib = temp_dir / "liba.a"
if ranlib.exists():
    ranlib.unlink()

deep_ep_obj_list = sorted(deep_ep_obj_dir.glob("kernels/**/*.o"))
nvshmem_obj_set = set(nvshmem_obj_dir.glob("**/*.o"))
for exclude_obj in sorted(nvshmem_obj_set):
    # Create liba.a with one object file removed
    subprocess.check_call(
        ["ar", "rcs", ranlib, *(nvshmem_obj_set - {exclude_obj})])
    # Test whether there are undefined symbols
    res = subprocess.call([
        "/usr/local/cuda/bin/nvcc", *gencode_args, "-Xlinker", "--no-undefined",
        "-shared", *deep_ep_obj_list, ranlib, "-o", temp_dir / "a.out"
    ])
    # If there are no undefined symbols, print "-" to indicate the file can be omitted
    print("-" if res == 0 else "+",
          str(exclude_obj.relative_to(nvshmem_obj_dir))[:-2])
    # Unlink the archive file because `ar` appends existing archives
    ranlib.unlink()
