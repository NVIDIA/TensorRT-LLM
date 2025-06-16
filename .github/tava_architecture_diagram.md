```mermaid
graph TB
    subgraph "User API & CLI Tools"
        CLI[CLI Tools]
        LLMAPI[LLM API]
        CLI --> LLMAPI
    end

    subgraph "Model Checkpoint"
        Checkpoint[Huggingface Models]
        Checkpoint --> CLI
        Checkpoint --> LLMAPI

    end

    subgraph "TensorRT_Flow"
        trtllmExecutor[trtllm.Executor]
        Engine[TensorRT Engine]
        Plugins[TensorRT Plugins]
        LLMAPI --> trtllmExecutor
        trtllmExecutor --> |build|Engine
        trtllmExecutor --> |compile|TRTGraph
        trtllmExecutor --> |compile|Plugins
        Engine --> Executor
        Plugins --> Executor
        TRTGraph --> Executor
        Executor --> Decoder[Decoder]
        Decoder --> Sampling[Sampling]
        Executor --> Scheduler[Scheduler]
        Scheduler --> |In-flight Batching| BatchManager[Batch Manager]
        BatchManager --> KVCache[KV Cache Manager]
    end

    subgraph "PyTorch_Flow"
        PyExecutor[PyExecutor]
        PyEngine[PyTorch Engine]
        CustomOps[Custom Ops]
        PyTorchOps[Pytorch Ops]
        KernelLibs[Kernel Libs]
        LLMAPI --> PyExecutor
        PyExecutor --> PyEngine[PyTorch Engine]
        PyEngine --> CustomOps
        PyEngine --> PyTorchOps
        PyEngine --> KernelLibs
        PyEngine --> PyScheduler[Scheduler]
        PyEngine --> PyDecoder[Decoder]
        PyScheduler --> |Pybind|Scheduler
        PyDecoder --> |Pybind|Decoder
        
    end

    subgraph "TensorRT-LLM Kernel Libs"
        cudaKernel[CUDA Kernels]
    end

    subgraph "Output_Results"
        Tokens[Generated Tokens]
        Stats[Performance Stats]
        Metrics[Accuracy Metrics]
    end

    TensorRT_Flow --> Output_Results
    PyTorch_Flow --> Output_Results

    CustomOps --> cudaKernel
    Plugins --> cudaKernel


    %% TensorRT_Flow --> |Generate| Tokens
    %% TensorRT_Flow --> |Report| Stats
    %% TensorRT_Flow --> |Evaluate| Metrics
    %% PyTorch_Flow --> |Generate| Tokens
    %% PyTorch_Flow --> |Report| Stats
    %% PyTorch_Flow --> |Evaluate| Metrics

    %% Add descriptions for CLI tools
    classDef cli fill:#f9f,stroke:#333,stroke-width:2px;
    class CLI cli;
    
    %% Add descriptions for components
    classDef component fill:#bbf,stroke:#333,stroke-width:2px;
    class TRTGraph,Plugins,Engine,Executor,Decoder,Scheduler,BatchManager,KVCache component;
    
    %% Add descriptions for APIs
    classDef api fill:#bfb,stroke:#333,stroke-width:2px;
    class PythonAPI,CppAPI,LLMAPI api;

    %% Add descriptions for results
    classDef result fill:#fbb,stroke:#333,stroke-width:2px;
    class Tokens,Stats,Metrics result;
```

## CLI Tools Description

1. **trtllm-build**: Builds TensorRT engines from TensorRT-LLM checkpoints
2. **trtllm-bench**: Benchmarks the performance of TensorRT-LLM models
3. **trtllm-eval**: Evaluates model accuracy and performance
4. **trtllm-llmapi-launch**: Launches the LLM API server
5. **trtllm-prune**: Prunes model weights for optimization
6. **trtllm-refit**: Updates weights in a TensorRT engine
7. **trtllm-serve**: Serves models through various interfaces

## Key Components

1. **Model Definition & Conversion**:
   - Converts models from various sources (HuggingFace, NeMo, JAX, etc.)
   - Creates TensorRT-LLM checkpoints
   - Supports different model formats and architectures

2. **TensorRT Flow**:
   - TensorRT Engine: Optimized inference engine
   - Runtime: Core execution engine
   - C++ Runtime: High-performance implementation
   - Executor: Manages model execution
   - Scheduler: Handles request scheduling and batching
   - Batch Manager: Manages in-flight batching
   - KV Cache Manager: Handles KV cache memory
   - Decoder: Generates output tokens
   - TensorRT Plugins: Custom operations
   - Custom Kernels: Optimized CUDA implementations

3. **PyTorch Flow**:
   - PyTorch Runtime: Python-based implementation
   - PyExecutor: Manages PyTorch model execution
   - PyScheduler: Handles PyTorch request scheduling
   - PyDecoder: PyTorch token generation
   - PyModelEngine: PyTorch model engine

4. **API Layer**:
   - Python API: High-level Python interface
   - C++ API: Low-level C++ interface
   - LLM API: Interface for model interaction (generate, chat, stream, batch, embed, tokenize)

5. **Output Results**:
   - Generated Tokens: Output text from the model
   - Performance Stats: Throughput, latency, memory usage
   - Accuracy Metrics: Model evaluation metrics
