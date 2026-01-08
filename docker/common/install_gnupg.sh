#!/bin/bash
set -e  # Exit on error
set -x  # Print commands being executed

# Install GnuPG 2.4.9 from source

GNUPG_VERSION="2.4.9"
GNUPG_TAR="gnupg-${GNUPG_VERSION}.tar.bz2"
# Use NVIDIA internal artifactory for faster/reliable downloads
GNUPG_URL="https://urm.nvidia.com/artifactory/sw-tensorrt-generic-local/llm-artifacts/LLM/BuildDockerImages/packages/${GNUPG_TAR}"

# Check and display existing GnuPG installation (if any)
which gpg || echo "No gpg found in PATH"
gpg --version || echo "No gpg version available"

# Remove all existing GnuPG packages to avoid conflicts
# Use 'purge' to remove packages AND their configuration files from dpkg database
echo "Purging all GnuPG-related packages..."
apt-get update && apt-get purge -y --allow-remove-essential \
    gnupg gnupg2 gnupg1 \
    gpg gpg-agent gpgconf gpgsm \
    dirmngr \
    || true

# Clean up package lists and caches to ensure scanners don't detect old metadata
apt-get autoremove -y || true
apt-get clean || true

# Remove any remaining gpg-related files in /usr/bin if they exist
echo "Checking for remaining gpg files in /usr/bin..."
ls /usr/bin/ | grep gpg || echo "No gpg files found"

if ls /usr/bin/ | grep -q gpg; then
    echo "Removing remaining gpg files from /usr/bin..."
    rm -f /usr/bin/*gpg*
    echo "Cleanup completed"
fi


# Change to temporary directory
cd /tmp

# Install all dependencies
apt-get update && apt-get install -y --no-install-recommends \
  wget \
  lbzip2 \
  build-essential \
  libgpg-error-dev \
  libgcrypt-dev \
  libassuan-dev \
  libksba-dev \
  libnpth-dev \
  zlib1g-dev

# Download GnuPG source from NVIDIA artifactory
wget ${GNUPG_URL}

# Extract source code
tar xf ${GNUPG_TAR}
cd gnupg-${GNUPG_VERSION}

# Configure, build, and install
./configure
make -j$(nproc)
make install

# Clean up
cd /tmp
rm -rf gnupg-${GNUPG_VERSION} ${GNUPG_TAR}

# Verify installation
# Note: Execute gpg first to ensure it's found in PATH after hash -r
hash -r
gpg --version
which gpg

# Verify installation
ls /usr/bin/ | grep keyboxd || echo "No keyboxd found in /usr/bin/"
ls /usr/bin/ | grep gpg || echo "No gpg found in /usr/bin/"
ls /usr/bin/ | grep gnupg || echo "No gnupg found in /usr/bin/"
ls /usr/bin/ | grep dirmngr || echo "No dirmngr found in /usr/bin/"

dpkg -l | grep keyboxd || echo "No keyboxd found in dpkg"
dpkg -l | grep gpg || echo "No gpg found in dpkg"
dpkg -l | grep gnupg || echo "No gnupg found in dpkg"
dpkg -l | grep dirmngr || echo "No dirmngr found in dpkg"

echo "GnuPG ${GNUPG_VERSION} installed successfully"
