=======================================================
LLM Examples Introduction
=======================================================

Here is a simple example to show how to use the LLM with TinyLlama.

.. literalinclude:: ../../../examples/llm-api/quickstart_example.py
   :language: python
   :linenos:

The LLM API can be used for both offline or online usage. See more examples of the LLM API here:

.. toctree::
    :maxdepth: 1
    :caption: LLM API Examples

    llm_inference
    llm_inference_distributed
    llm_inference_async
    llm_inference_async_streaming
    llm_quantization
    llm_auto_parallel
    llm_multilora
    llm_logits_processor
    llm_guided_decoding
    llm_lookahead_decoding

For more details on how to fully utilize this API, check out:

* `Common customizations <customization.html>`_
* `LLM API Reference <../llm-api/index.html>`_

.. _supported_models:

Supported Models
================

* Llama (including variants Mistral, Mixtral, InternLM)
* GPT (including variants Starcoder-1/2, Santacoder)
* Gemma-1/2
* Phi-1/2/3
* ChatGLM (including variants glm-10b, chatglm, chatglm2, chatglm3, glm4)
* QWen-1/1.5/2
* Falcon
* Baichuan-1/2
* GPT-J
* Mamba-1/2

.. _model_preparation:

Model Preparation
==================

The ``LLM`` class supports input from any of the following:

1. **Hugging Face Hub**: Triggers a download from the Hugging Face model hub, such as ``TinyLlama/TinyLlama-1.1B-Chat-v1.0``.

2. **Local Hugging Face models**: Uses a locally stored Hugging Face model.

3. **Local TensorRT-LLM engine**: Built by ``trtllm-build`` tool or saved by the Python LLM API.

Any of these formats can be used interchangeably with the ``LLM(model=<any-model-path>)`` constructor.

The following sections show how to use these different formats for the LLM API.

.. _hugging_face_hub:

Hugging Face Hub
#######################

Using the Hugging Face hub is as simple as specifying the repo name in the LLM constructor:

.. code-block:: python

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")


Local Hugging Face Models
##############################

Given the popularity of the Hugging Face model hub, the API supports the Hugging Face format as one of the starting points.
To use the API with Llama 3.1 models, download the model from the `Meta Llama 3.1 8B model page <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B>`_ by using the following command:


.. code-block:: console

   git lfs install
   git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B


After the model downloading finished, we can load the model as below:

.. code-block:: python

   llm = LLM(model=<path_to_meta_llama_from_hf>)


Note:
    Using this model is subject to a `particular license <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`_. Agree to the terms and `authenticate with HuggingFace <https://huggingface.co/meta-llama/Meta-Llama-3-8B?clone=true>`_ to begin the download.

.. _from_tensorrt_llm_engine:

From TensorRT-LLM Engine
#############################

There are two ways to build the TensorRT-LLM engine:

1. **Using the ``trtllm-build`` Tool**: You can build the TensorRT-LLM engine from the Hugging Face model directly with the ``trtllm-build`` tool and then save the engine to disk for later use.
   Refer to the `README <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama>`_ in the ``examples/llama`` repository on GitHub.

   After the engine building is finished, we can load the model as below::

.. code-block:: python

    llm = LLM(model=<path_to_trt_engine>)


2. **Using an ``LLM`` Instance**: Use an ``LLM`` instance to create the engine and persist to local disk::

.. code-block:: python

    llm = LLM(<model-path>)

    # Save engine to local disk
    llm.save(<engine-dir>)

The engine can be reloaded as above.
