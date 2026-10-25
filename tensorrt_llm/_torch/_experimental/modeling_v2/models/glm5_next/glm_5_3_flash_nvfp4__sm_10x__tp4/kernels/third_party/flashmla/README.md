Vendored unmodified from deepseek-ai/FlashMLA at 15f13e5030374295491c5ce31b02d7e63a7772c6 (the commit TensorRT-LLM
pins), MIT license (LICENSE): the SM100 sparse prefill kernels (head64 / head128, d_qk 512) and the header-only kerutils
they use. ../../csrc/kernels/fmla_fwd.cu instantiates them for this target's sparse MLA prefill
(glm::k::fmla_sparse_fwd); ../../csrc/kernels/fmla_union.cu adapts the head64 kernel to query tiles.
