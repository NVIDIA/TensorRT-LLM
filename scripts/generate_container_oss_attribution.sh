#!/bin/bash

set -e

# Generate open source attribution file with parameterized URL
# Usage: ./generate_container_oss_attribution.sh <image_name> <tag> <arch> [output_file]

show_usage() {
  local error_msg="${1}"

  if [ -n "${error_msg}" ]; then
    echo "ERROR: ${error_msg}"
    echo ""
  fi

  echo "Usage: $0 <image_name> <tag> <arch> [output_file]"
  echo ""
  echo "Arguments:"
  echo "  image_name   - Name of the image (e.g., tensorrt-llm)"
  echo "  tag          - Image tag/version (e.g., 1.0.0)"
  echo "  arch         - Architecture (e.g., amd64, arm64)"
  echo "  output_file  - Optional output file path (default: /third-party-source/NOTICE.txt)"
  echo ""
  echo "Example:"
  echo "  $0 tensorrt-llm 1.0.0 amd64"
  echo ""
  exit 1
}

IMAGE_NAME="${1}"
TAG="${2}"
ARCH="${3}"
OUTPUT_FILE="/NOTICE.txt"

# Validate required parameters
[ -z "${IMAGE_NAME}" ] && show_usage "Missing required parameter IMAGE_NAME"
[ -z "${TAG}" ] && show_usage "Missing required parameter TAG"
[ -z "${ARCH}" ] && show_usage "Missing required parameter ARCH"

# Construct the URL
ROOT_URL="https://opensource.nvidia.com/oss/teams/nvidia"
OSS_URL="${ROOT_URL}/${IMAGE_NAME}/${TAG}:linux-${ARCH}/index.html"

# Create output directory if needed
OUTPUT_DIR="$(dirname "${OUTPUT_FILE}")"
mkdir -p "${OUTPUT_DIR}"

# Generate the attribution file
cat > "${OUTPUT_FILE}" << EOF
This container image includes open-source software whose source code is archived in the /third-party-source directory or at ${OSS_URL}.

For further inquiries or assistance, contact us at oss-requests@nvidia.com
EOF

echo "✓ Attribution file generated: ${OUTPUT_FILE}"
echo "  URL: ${OSS_URL}"
