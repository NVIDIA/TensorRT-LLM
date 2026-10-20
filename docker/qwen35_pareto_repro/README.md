# Qwen3.5-397B-A17B NVFP4 GB300 reproduction image

One image for every Qwen3.5-397B-A17B NVFP4 deployment on GB300 covered by this branch (low / mid TEP layouts and
the DEP16 high end).

Image = base release image `nvidia/tensorrt-llm/release:1.3.0a0.dev202610151219` (built from `1b480b5af9`)
+ every file this branch adds or changes under `tensorrt_llm/` (relative to `1b480b5af9`)
+ four compiled files rebuilt from this branch with the image's own toolchain and flags:
`bindings.cpython-312-aarch64-linux-gnu.so`, `deep_ep_cpp_tllm.cpython-312-aarch64-linux-gnu.so`,
`libs/libtensorrt_llm.so`, `libs/libth_common.so` (no other compiled file differs from a build of `1b480b5af9`).

Everything the branch adds is switchable by environment variable and off by default unless noted in the commit that
adds it (e.g. `TRTLLM_TOKEN_INGEST_I32=0` turns the int32 token-ingest bridge off), so runs made with older trees can
be replayed on this image.

## Rebuild

1. C++ (aarch64, inside the base image so the toolchain matches; flags = the image's CI wheel build):
   ```
   python3 scripts/build_wheel.py --build_dir <dir> -G Ninja --use_ccache -a "90-real;100-real;103-real;120-real" \
       --skip_building_wheel --skip-stubs \
       --extra-cmake-vars NIXL_ROOT=/opt/nvidia/nvda_nixl --extra-cmake-vars ENABLE_BOLT_COMPATIBLE=ON
   ```
   This leaves the compiled files under `tensorrt_llm/` of the checkout.
2. Payload: `docker/qwen35_pareto_repro/make_payload.sh <payload dir> [<dir holding the compiled files>]`
   (Python files are taken from `HEAD`, not from the working tree; refuses to run with uncommitted changes there).
3. Image, either
   - Docker: `docker build -f docker/qwen35_pareto_repro/Dockerfile -t <tag> <payload dir>`, or
   - Slurm + pyxis (no GPU needed):
     `ACCOUNT=<acct> docker/qwen35_pareto_repro/save_sqsh.sh <base .sqsh> <payload dir> <out .sqsh>`.

`apply_payload.sh` checks every installed file against the payload's `MANIFEST.sha256` and writes the provenance to
`/opt/qwen35_pareto_repro/` (`SOURCE.txt`, `MANIFEST.sha256`, `IMAGE_INFO.txt`) inside the image.
