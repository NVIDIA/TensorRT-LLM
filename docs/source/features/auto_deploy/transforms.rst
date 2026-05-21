AutoDeploy Transforms
=====================

This page enumerates the graph transforms that ship with AutoDeploy. Each transform
lives under ``tensorrt_llm._torch.auto_deploy.transform.library`` and is registered
with the AutoDeploy ``InferenceOptimizer`` pipeline. Public classes, functions, and
configuration models are rendered directly from the transform docstrings via Sphinx
``automodule``.

For an overview of how transforms fit into the AutoDeploy pipeline, see
:doc:`auto-deploy`. For information on configuring which transforms run and in what
order, see :doc:`advanced/expert_configurations`.

Transform Interface
-------------------

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.interface
   :members:
   :undoc-members:
   :show-inheritance:

Optimizer
---------

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

Graph Module Visualizer
-----------------------

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.graph_module_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Transform Library
-----------------

The transforms below are discovered automatically at import time by
``tensorrt_llm._torch.auto_deploy.transform.library``.

Attention
~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.attention
   :members:
   :undoc-members:
   :show-inheritance:

Build Model
~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.build_model
   :members:
   :undoc-members:
   :show-inheritance:

Cleanup Identity Dtype Cast
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.cleanup_identity_dtype_cast
   :members:
   :undoc-members:
   :show-inheritance:

Cleanup Input Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.cleanup_input_constraints
   :members:
   :undoc-members:
   :show-inheritance:

Cleanup No-op Add
~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.cleanup_noop_add
   :members:
   :undoc-members:
   :show-inheritance:

Cleanup No-op Slice
~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.cleanup_noop_slice
   :members:
   :undoc-members:
   :show-inheritance:

Collectives
~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.collectives
   :members:
   :undoc-members:
   :show-inheritance:

Compile Model
~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.compile_model
   :members:
   :undoc-members:
   :show-inheritance:

Eliminate Redundant Transposes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.eliminate_redundant_transposes
   :members:
   :undoc-members:
   :show-inheritance:

Export to Graph Module
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.export_to_gm
   :members:
   :undoc-members:
   :show-inheritance:

Fuse Causal Conv
~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_causal_conv
   :members:
   :undoc-members:
   :show-inheritance:

Fuse GDN Gating
~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_gdn_gating
   :members:
   :undoc-members:
   :show-inheritance:

Fuse Mamba A-Log
~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_mamba_a_log
   :members:
   :undoc-members:
   :show-inheritance:

Fuse Quant
~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_quant
   :members:
   :undoc-members:
   :show-inheritance:

Fuse ReLU2 Quant NVFP4
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_relu2_quant_nvfp4
   :members:
   :undoc-members:
   :show-inheritance:

Fuse RMSNorm Quant FP8
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_rmsnorm_quant_fp8
   :members:
   :undoc-members:
   :show-inheritance:

Fuse RoPE into TRT-LLM Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_rope_into_trtllm_attention
   :members:
   :undoc-members:
   :show-inheritance:

Fuse RoPE MLA
~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_rope_mla
   :members:
   :undoc-members:
   :show-inheritance:

Fuse SiLU Mul
~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_silu_mul
   :members:
   :undoc-members:
   :show-inheritance:

Fuse SwiGLU
~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_swiglu
   :members:
   :undoc-members:
   :show-inheritance:

Fuse TRT-LLM Attention Quant FP8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fuse_trtllm_attention_quant_fp8
   :members:
   :undoc-members:
   :show-inheritance:

Fused Add RMSNorm
~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fused_add_rms_norm
   :members:
   :undoc-members:
   :show-inheritance:

Fused MoE
~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fused_moe
   :members:
   :undoc-members:
   :show-inheritance:

Fusion
~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.fusion
   :members:
   :undoc-members:
   :show-inheritance:

Gather Logits Before LM Head
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.gather_logits_before_lm_head
   :members:
   :undoc-members:
   :show-inheritance:

Hidden States
~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.hidden_states
   :members:
   :undoc-members:
   :show-inheritance:

KV Cache
~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.kvcache
   :members:
   :undoc-members:
   :show-inheritance:

KV Cache (Transformers)
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.kvcache_transformers
   :members:
   :undoc-members:
   :show-inheritance:

L2 Norm
~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.l2_norm
   :members:
   :undoc-members:
   :show-inheritance:

Load Weights
~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.load_weights
   :members:
   :undoc-members:
   :show-inheritance:

MLIR Elementwise Fusion
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.mlir_elementwise_fusion
   :members:
   :undoc-members:
   :show-inheritance:

MoE Routing
~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.moe_routing
   :members:
   :undoc-members:
   :show-inheritance:

mRoPE Delta Cache
~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.mrope_delta_cache
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Stream Attention
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_attn
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Stream GEMM
~~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_gemm
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Stream MoE
~~~~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_moe
   :members:
   :undoc-members:
   :show-inheritance:

MXFP4 MoE
~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.mxfp4_moe
   :members:
   :undoc-members:
   :show-inheritance:

Quantization
~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.quantization
   :members:
   :undoc-members:
   :show-inheritance:

Quantize MoE
~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.quantize_moe
   :members:
   :undoc-members:
   :show-inheritance:

RMSNorm
~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.rms_norm
   :members:
   :undoc-members:
   :show-inheritance:

RoPE
~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.rope
   :members:
   :undoc-members:
   :show-inheritance:

Sharding
~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.sharding
   :members:
   :undoc-members:
   :show-inheritance:

Sharding IR
~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir
   :members:
   :undoc-members:
   :show-inheritance:

SSM Cache
~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.ssm_cache
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
~~~~~~~~~~~~~

.. automodule:: tensorrt_llm._torch.auto_deploy.transform.library.visualization
   :members:
   :undoc-members:
   :show-inheritance:
