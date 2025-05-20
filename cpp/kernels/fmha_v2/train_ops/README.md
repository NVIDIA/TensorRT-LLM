


# Running the unit test

Under the `train_ops` folder, clone APEX

```
cd train_ops
git clone https://github.com/NVIDIA/apex.git
cd apex
git submodule update --init --recursive
cd ..
```

from the project root, launch the build container:

```
cd docker
make launch_docker
```

Then inside the container, /repo is the repository mount point:

```
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
cd /repo/train_ops
python train_setup.py
mkdir -p build && cd build && cmake .. && make -j
```
Note that we use flash attention by default.

Then in `train_ops`, run the test script

```
python fmha_unit_test.py
```
