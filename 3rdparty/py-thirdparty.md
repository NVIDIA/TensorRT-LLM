# Adding new python dependencies via pip

If you add a new python dependency and that dependency will be installed in
(and, thus, distributed with) the container, please follow this process.

## Third-party packages without modification

If the package you wish to add does not require modification, then please follow
these steps:

1. Add your new dependency to one of the "pip install" invocations among the
   scripts in docker/common.sh. If none of the existing ones make sense, then
   add a new script to install your package and add a new line to
   Dockerfile.multi to run your script.
2. Update ATTRIBUTIONS-Python.md to include all new dependencies. Note that this
   must cover the transitive closure of all dependencies. The dependency you
   added may have pulled in new transitive dependencies and we must ensure all
   are attributed in this file.
3. Verify that your newly added package is listed in the compliance reports and
   that sources are pulled via the compliance tooling.

## Third-party packages with modification

If you wish to depend on a package with nvidia-contributed modifications that
haven't been upstreamed then please follow these steps:

1. File an OSRB request to fork/contribute to a 3rd party open source package.
   https://confluence.nvidia.com/display/OSS/Contribution+to+Open+Source
2. Clone the original repository to a new public nvidia-controlled location
   (e.g. https://gitlab.com/nvidia/tensorrt-llm/oss-components/)
3. Register this new repository under nspec
4. Make modifications in that public repository. Ensure that the clone
   repository clearly indicates the software license via /LICENSE.txt in the
   root of the repository. Ensure that this file contains a copyright statement
   indicating copyright held by the original author(s) and Nvidia.
5. Publish the modified package to pypi under a new name (e.g. nvidia-<package>)
6. Add your new dependency to one of the "pip install" invocations among the
   scripts in docker/common.sh. If none of the existing ones make sense, then
   add a new script to install your package and add a new line to
   Dockerfile.multi to run your script.
7. Update ATTRIBUTIONS-Python.md to include all new dependencies. Note that this
   must cover the transitive closure of all dependencies. The dependency you
   added may have pulled in new transitive dependencies and we must ensure all
   are attributed in this file.
8. Verify that your newly added package is listed in the compliance reports and
   that sources are pulled via the compliance tooling.

Notes:
* For pip/uv-installed versions of TensorRT-LLM, the modified package will be
  installed as a transitive dependency by the package manager
* For the container distribution of TensorRT-LLM, the modified package will be
  pre-installed from the same pypi location via pip

## Individual third-party sources with modification

If you wish to integrate third-party source files with nvidia-contributed
modifications that haven't been upstreamed then please follow these steps:

1. File an OSRB request to use open source:
   https://confluence.nvidia.com/display/OSS/So+you+want+to+use+open+source+in+your+product
2. Clone the original repository to a new nvidia-controlled location
   (e.g. https://gitlab.com/nvidia/tensorrt-llm/oss-components/)
3. Make modifications in that repository on branch so that the versions
   "as-used" can be easily found and the diff against upstream easily viewed.
4. Copy the desired source files into the TensorRT-LLM repository.
5. Update ATTRIBUTIONS-Python.md to include attribution for the source files
   you have added. Note the terms of the license on the original repository
   and see the examples already in the file to understand what all needs to be
   stated.
