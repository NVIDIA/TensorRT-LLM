#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

set -ex

install_ci_ubuntu() {
  apt-get update && apt-get install -y --no-install-recommends \
    autossh \
    libffi-dev \
    openssh-client \
    openssh-server \
    rsync \
    sshpass \
  && apt-get clean && rm -rf /var/lib/apt/lists/*
}

install_ci_rocky() {
  dnf install -y \
    autossh \
    libffi-devel \
    openssh-clients \
    openssh-server \
    rsync \
    sshpass
}

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu) install_ci_ubuntu ;;
  rocky)  install_ci_rocky  ;;
  *) echo "Unsupported OS: $ID"; exit 1 ;;
esac
