# Usage: put this script under TensorRT-LLM/launch.sh, then `source launch.sh`
# It will pull the official dev docker image that best matches your current commit, and mount the current directory as /code/tensorrt_llm, and the llm-models repo
# when you have two ssh sessions to computelab, just `source launch.sh`` again, it will enter the same container

CONTAINER_NAME="tensorrt_llm-jenkins-$(id --user --name)"
if docker ps | grep -q $CONTAINER_NAME
then
  echo "container already running. entering..."
else
  echo "start container"
  make -C docker jenkins_run LOCAL_USER=1 EXTRA_VOLUMES="-v /home/scratch.trt_llm_data/llm-models:/scratch.trt_llm_data/llm-models"
fi
docker exec -it $CONTAINER_NAME bash



# if $NV_GPU is set correctly on computelab, we can use the following; however, sometimes it's just empty...
# --gpus \"device=$NV_GPU\" 
# GPU="all" # full node access
# GPU="device=0" # partial node access, like '"device=0,1,2,3"' -- the single quote is important!

# ## Option 1: pull official dev image
# PWD=$(pwd)
# CONTAINER_NAME="hhh_tllm" # change to whatever name you like
# IMAGE_NAME=$(grep -oP 'LLM_DOCKER_IMAGE = "\K[^"]+' jenkins/L0_MergeRequest.groovy) # Copilot taught me this!
# MOUNT_DIR="/mnt/scratch.haohangh_gpu/git_haohangh/TensorRT-LLM"
# # launch container
# if docker ps | grep -q $CONTAINER_NAME
# then
#   echo "container exists. launch it"
#   docker exec -it $CONTAINER_NAME bash
# else
#   echo "start container from image $IMAGE_NAME"
#   docker run --gpus $GPU --name $CONTAINER_NAME --rm -itd --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --privileged -v $PWD:$MOUNT_DIR -v /home/scratch.trt_llm_data/llm-models:/scratch.trt_llm_data/llm-models --workdir $MOUNT_DIR $IMAGE_NAME 
#   docker exec -it $CONTAINER_NAME bash
# fi
################

## Option 2: build image and save locally
# make -C docker build
# make -C docker run
# docker commit <hash> hhh_tllm:latest
# docker save hhh_tllm:latest > hhh_tllm.tar

# CONTAINER_NAME="hhh_tllm"
# GPU=\"device=0\" # partial node access

# # load image
# if docker images | grep -q $CONTAINER_NAME
# then
#   echo "image exists. launch container"
# else
#   echo "load image from disk"
#   docker load < $CONTAINER_NAME.tar
# fi

# # launch container
# if docker ps | grep -q $CONTAINER_NAME
# then
#   echo "container exists. launch it"
#   docker exec -it $CONTAINER_NAME bash
# else
#   echo "start container from image"
#   docker run --gpus $GPU --name $CONTAINER_NAME --rm -itd --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --tmpfs /tmp:exec --privileged -v $PWD:/code/tensorrt_llm -v /home/scratch.trt_llm_data/llm-models:/home/scratch.trt_llm_data/llm-models --workdir /code/tensorrt_llm hhh_tllm:latest 
#   docker exec -it $CONTAINER_NAME bash
# fi
################

## Memo: computelab commands
# tmux new -s tllm
# tmux attach -t tllm
# cdb find
# crun --time=10:00:00 -q 'gpu.product_name=*H100* and gpus=1 and node=*ipp2-* and cpu.arch=x86_64' -i # single gpu not shared with others
# crun --time=10:00:00 -q 'gpu.product_name=*H100* and gpus=8 and memory_size_gb>900' -i --cpu-arch-agnostic
# crun --time=10:00:00 -q 'gpu.product_name=*A100* and gpu.memory_total_gb>70 and gpus=8 and memory_size_gb>900' -i --cpu-arch-agnostic
################