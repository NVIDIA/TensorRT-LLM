# Integration for OpenAI Triton

The typical approach to integrate a kernel into TensorRT LLM is to create TensorRT plugins.
Specially for integrating OpenAI Triton kernels, there are two methods:

1. Creating TensorRT plugin manually, you can refer to [manual plugin example](./manual_plugin/) for details,
2. Generate the TensorRT plugins automatically, please refer to [automatic plugin example](./plugin_autogen/) for details.
