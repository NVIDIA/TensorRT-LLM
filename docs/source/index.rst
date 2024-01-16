.. TensorRT-LLM documentation master file, created by
   sphinx-quickstart on Wed Sep 20 08:35:21 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TensorRT-LLM's documentation!
========================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   architecture.md
   gpt_runtime.md
   batch_manager.md
   inference_request.md
   gpt_attention.md
   precision.md
   build_from_source.md
   performance.md
   2023-05-19-how-to-debug.md
   2023-05-17-how-to-add-a-new-model.md
   graph-rewriting.md
   memory.md
   new_workflow.md

Python API
----------

- :doc:`tensorrt_llm.layers <python-api/tensorrt_llm.layers>`
- :doc:`tensorrt_llm.functional <python-api/tensorrt_llm.functional>`
- :doc:`tensorrt_llm.models <python-api/tensorrt_llm.models>`
- :doc:`tensorrt_llm.plugin <python-api/tensorrt_llm.plugin>`
- :doc:`tensorrt_llm.quantization <python-api/tensorrt_llm.quantization>`
- :doc:`tensorrt_llm.runtime <python-api/tensorrt_llm.runtime>`


.. toctree::
   :maxdepth: 2
   :caption: Python API
   :hidden:

   python-api/tensorrt_llm.layers
   python-api/tensorrt_llm.functional
   python-api/tensorrt_llm.models
   python-api/tensorrt_llm.plugin
   python-api/tensorrt_llm.quantization
   python-api/tensorrt_llm.runtime


C++ API
---------

- :doc:`cpp/runtime <_cpp_gen/runtime>`


.. toctree::
   :maxdepth: 2
   :caption: C++ API
   :hidden:

   _cpp_gen/runtime


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Blogs
----------

.. toctree::
   :maxdepth: 2
   :caption: Blogs
   :hidden:

   blogs/H100vsA100.md
   blogs/H200launch.md
   blogs/Falcon180B-H200.md
