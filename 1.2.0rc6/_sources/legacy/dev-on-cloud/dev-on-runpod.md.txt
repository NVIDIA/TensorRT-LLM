(dev-on-runpod)=

# Develop TensorRT LLM on Runpod
[Runpod](https://runpod.io) is a popular cloud platform among many researchers. This doc describes how to develop TensorRT LLM on Runpod.

## Prepare

### Create a Runpod account
Please refer to the [Runpod Getting Started](https://docs.runpod.io/get-started/).

### Configure SSH Key
Please refer to the [Configure SSH Key](https://docs.runpod.io/pods/configuration/use-ssh).

Note that we can skip the step of "Start your Pod. Make sure of the following things" here as we will introduce it below.

## Build the TensorRT LLM Docker Image and Upload to DockerHub
Please refer to the [Build Image to DockerHub](build-image-to-dockerhub.md).

Note that the docker image must enable ssh access. See on [Enable ssh access to the container](build-image-to-dockerhub.md#enable-ssh-access-to-the-container).

## Create a Pod Template
Click "Template" bottom on the menus and click "Create Template" bottom.

Fill the docker image link of DockerHub such as `docker.io/<your_dockerhub_username>/tensorrt_llm:devel` on "Docker Image" field.

Fill "22" into "Expose TCP Ports" field.

Fill
```bash
sleep infinity
```
into 'Container Start Command' field.

## Connect to the Pod
Please refer to the [Connect to the Pod](https://docs.runpod.io/pods/connect-to-a-pod).

You can connect the pod with SSH or Web Terminal.

If you want to connect the pod with SSH, you can copy the command from "SSH over exposed TCP" field and run it on your host.

In some scenarios such as using a team account, your public key has not been added to the pod successfully. You can directly add this command to the 'Container Start Command' field as:

```bash
bash -c 'echo "<your_public_key>" >> ~/.ssh/authorized_keys;sleep infinity'
```

Enjoy your development!
