#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Rebuild the pinned package with the platform trust store instead of embedded roots.
version=$(python3 -c 'from importlib.metadata import version; print(version("openai-harmony"))')
if [[ "$version" != "0.0.4" ]]; then
    echo "Unsupported Harmony version for source rebuild: $version" >&2
    exit 1
fi
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
export CARGO_HOME="$work_dir/cargo"
export RUSTUP_HOME="$work_dir/rustup"
HARMONY_SOURCE_URL=https://github.com/openai/harmony/archive/72079ca4971f326dddcd5ff62bfb7e2fff37a07a.tar.gz
export RUSTUP_DIST_SERVER=${RUSTUP_DIST_SERVER:-https://static.rust-lang.org}
export RUSTUP_UPDATE_ROOT=${RUSTUP_UPDATE_ROOT:-https://static.rust-lang.org/rustup}

if [[ -n "${GITHUB_MIRROR:-}" ]]; then
    HARMONY_SOURCE_URL=https://artifactory.pdx.nvidia.com/artifactory/github-releases-remote/openai/harmony/archive/72079ca4971f326dddcd5ff62bfb7e2fff37a07a.tar.gz
    export PIP_INDEX_URL=https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple
    export RUSTUP_DIST_SERVER=https://artifactory.nvidia.com/artifactory/rustup-dist-remote
    export RUSTUP_UPDATE_ROOT="${RUSTUP_DIST_SERVER}/rustup"
    mkdir -p "$CARGO_HOME"
    cat > "$CARGO_HOME/config.toml" <<'CARGO_CONFIG'
[source.crates-io]
replace-with = "artifactory"
[source.artifactory]
registry = "sparse+https://artifactory.nvidia.com/artifactory/api/cargo/cargo-crates-remote/index/"
CARGO_CONFIG
fi

case "$(uname -m)" in
    x86_64|aarch64) rust_target="$(uname -m)-unknown-linux-gnu" ;;
    *) echo "Unsupported Rust build architecture: $(uname -m)" >&2; exit 1 ;;
esac
curl --fail --show-error --silent --location \
    "${RUSTUP_UPDATE_ROOT}/dist/${rust_target}/rustup-init" -o "$work_dir/rustup-init"
chmod +x "$work_dir/rustup-init"
"$work_dir/rustup-init" -y --no-modify-path --profile minimal --default-toolchain 1.90.0
export PATH="$CARGO_HOME/bin:$PATH"
curl --fail --show-error --silent --location "$HARMONY_SOURCE_URL" -o "$work_dir/harmony.tar.gz"
mkdir "$work_dir/source"
tar -xzf "$work_dir/harmony.tar.gz" --strip-components=1 -C "$work_dir/source"
sed -i 's/"rustls-tls"/"rustls-tls-native-roots"/' "$work_dir/source/Cargo.toml"
python3 -m pip wheel --no-cache-dir --no-deps "$work_dir/source" --wheel-dir "$work_dir/wheels"
python3 -m pip install --no-cache-dir --no-deps --force-reinstall "$work_dir"/wheels/openai_harmony-*.whl
