Dynamo K8s Example
=================================

.. toctree::
   :maxdepth: 2
   :caption: Scripts


1. Install Dynamo Cloud

Please follow `this guide <https://docs.nvidia.com/dynamo/latest/guides/dynamo_deploy/dynamo_cloud.html>`_
to install Dynamo cloud for your Kubernetes cluster.

2. Deploy the TRT-LLM Deployment

Dynamo uses custom resource definitions (CRDs) to manage the lifecycle of the
deployments.  You can use the `DynamoDeploymentGraph yaml <https://github.com/ai-dynamo/dynamo/tree/main/components/backends/trtllm/deploy>`_
files to create aggregated, and disaggregated TRT-LLM deployments.

Please see `Deploying Dynamo Inference Graphs to Kubernetes using the Dynamo
Cloud Platform <https://docs.nvidia.com/dynamo/latest/guides/dynamo_deploy/operator_deployment.html>`_
for more details.
