(build-image-to-dockerhub)=

# Build the TensorRT-LLM Docker Image
When you develop trt-llm on cloud platform such as runpod, you may need to provide a docker image for the platform. So you firstly need to upload the image to dockerhub.

## Build the TensorRT-LLM Docker Image and Upload to DockerHub

```bash
make -C docker build
```
Then we can get the docker image named `tensorrt_llm/devel:latest`

### Enable ssh access to the container
Since the default docker image doesn’t have ssh support, we can’t ssh into it. We need to add ssh support to the container.
Let’s first create a new Dockerfile with below content:

```Dockerfile
FROM tensorrt_llm/devel:latest

RUN apt update && apt install openssh-server -y
RUN mkdir -p /run/sshd && chmod 755 /run/sshd
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
# add sshd to entrypoint script
RUN echo "sshd -E /opt/sshd.log" >> /opt/nvidia/entrypoint.d/99-start-sshd.sh
```

If we save this Dockerfile as `Dockerfile.ssh`. Then we can build the docker image with below command:

```bash
docker build -t tensorrt_llm/devel:with_ssh -f Dockerfile.ssh .
```

Then we can get the docker image named `tensorrt_llm/devel:with_ssh`

## Upload the Docker Image to DockerHub

You need to register a [dockerhub](https://hub.docker.com) account first if you don't have one.

Then you can click 'Personal Access Tokens' in the user menu and create a new token.

With the token, you can login to dockerhub with below command:

```bash
docker login -u <your_dockerhub_username>
```

Enter the token to the console.

After login, you can tag and push the docker image to dockerhub with below command:

```bash
docker tag tensorrt_llm/devel:with_ssh <your_dockerhub_username>/tensorrt_llm:devel
docker push <your_dockerhub_username>/tensorrt_llm:devel
```

Finally, you can see the docker image in your dockerhub repository and can use it with the link such as `docker.io/<your_dockerhub_username>/tensorrt_llm:devel`.
