set -ex
root=$(dirname $(realpath $0))
pushd $root
python scripts/build_wheel.py --trt_root="/usr/local/tensorrt" -i --always_build
# -c
#-b Debug
