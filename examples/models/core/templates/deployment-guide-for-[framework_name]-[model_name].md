# [Model_Name] Deployment Guide for [Framework_Name]
>For TensorRT-LLM example, see: https://github.com/jamieliNVIDIA/TensorRT-LLM/tree/feat/llama-3.3-deployment-guide
# Introduction

This deployment guide should provide step-by-step instructions for running [Model_Name] on [Framework_Name] on NVIDIA GPUs. 

The guide covers the complete setup, from preparing the software environment and configuring any parameters to launching tests and validating output.

# Access & Licensing

Any licensing or access instructions as well as associated links should be provided here.

# Prerequisites 

GPU: [eg. NVIDIA Blackwell or Hopper Architecture]   
OS:   [eg. Linux]   
Drivers: [eg. CUDA Driver 575 or Later]     
\[Others\]

# Models 

* \[Model\_Name]:\[Link\_to\_model\_checkpoint\]

# Deployment Steps

## Run Docker Container

Steps to build and run a docker container if using. Use code blocks for commands and outputs.

```shell

```

## Framework configuration 

Any settings or steps needed to configure the framework.

```shell

```

## Launching Framework

Any commands to launch the framework.

```shell

```

## Configs and Parameters 

Description and recommendations for configurations/parameters. Take the following flags for example.

\--tp\_size

**Description**: Sets the **tensor-parallel size**. This should typically match the number of GPUs you intend to use for a single model instance.

\--ep\_size

**Description**: Sets the **expert-parallel size** for Mixture-of-Experts (MoE) models. Like tp\_size, this should generally match the number of GPUs you're using. This setting has no effect on non-MoE models.

# Functionality Testing

If running functionality tests, such as testing the API endpoint, include that in this section.

## Test Name 

Component installation if applicable.

```shell

```

Description of test and how to execute. 

```shell

```

Output of test execution.

```shell

```

# Benchmarking Performance

Steps to configure performance benchmarking. For example, creating benchmarking scripts or launching additional containers will go here.

```shell

```

Additional benchmarking steps.

```shell

```

Sample Benchmark Output. For example, the following takes the performance metrics outputs from a sample test.

```
============ Serving Benchmark Result ============

---------------Time to First Token----------------

-----Time per Output Token (excl. 1st token)------

---------------Inter-token Latency----------------

----------------End-to-end Latency----------------

==================================================
```

## Key Metrics

Description and interpretation of key benchmark metrics. The following can be used as an example.

* Median Time to First Token (TTFT)  
  * The typical time elapsed from when a request is sent until the first output token is generated.  
* Total Token Throughput  
  * The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens. 

# References

Add references if applicable.

# Next Steps

Next steps if applicable.
