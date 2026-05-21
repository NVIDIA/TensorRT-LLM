Model Recipes
================

Preconfigured Recipes
---------------------

.. _recipe-selector:

Recipe selector
^^^^^^^^^^^^^^^

.. trtllm_config_selector::

.. note::

   These configs are for aggregated (in-flight batched) serving where prefill and decode run on the same GPU. For disaggregated serving setups, see the model-specific deployment guides below.

.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-traffic-patterns
   :end-before: .. end-note-traffic-patterns

Model-Specific Deployment Guides
---------------------------------

The deployment guides below provide more detailed instructions for serving specific models with TensorRT LLM.

.. toctree::
   :maxdepth: 1
   :name: Deployment Guides

   deployment-guide-for-nemotron-3-super-on-trtllm.md
   deployment-guide-for-deepseek-r1-on-trtllm.md
   deployment-guide-for-llama3.3-70b-on-trtllm.md
   deployment-guide-for-llama4-scout-on-trtllm.md
   deployment-guide-for-gpt-oss-on-trtllm.md
   deployment-guide-for-qwen3-on-trtllm.md
   deployment-guide-for-qwen3-next-on-trtllm.md
   deployment-guide-for-kimi-k2-thinking-on-trtllm.md
   deployment-guide-for-glm-5-on-trtllm.md
