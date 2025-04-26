(executor)=

# Executor API

TensorRT-LLM includes a high-level C++ API called the Executor API which allows you to execute requests
asynchronously, with in-flight batching, and without the need to define callbacks.

A software component (referred to as "the client" in the text that follows) can interact
with the executor using the API defined in the [`executor.h`](source:cpp/include/tensorrt_llm/executor/executor.h) file.
For details about the API, refer to the {ref}`_cpp_gen/executor.rst`.

The following sections provide an overview of the main classes defined in the Executor API.

## API

### The Executor Class

The `Executor` class is responsible for receiving requests from the client, and providing responses for those requests. The executor is constructed by providing a path to a directory containing the TensorRT-LLM engine or buffers containing the engine and the model JSON configuration. The client can create requests and enqueue those requests for execution using the `enqueueRequest` or `enqueueRequests` methods of the `Executor` class. Enqueued requests will be scheduled for execution by the executor, and multiple independent requests can be batched together at every iteration of the main execution loop (a process often referred to as continuous batching or iteration-level batching). Responses for a particular request can be awaited for by calling the `awaitResponses` method, and by providing the request id. Alternatively, responses for any requests can be awaited for by omitting to provide the request id when calling `awaitResponses`. The `Executor` class also allows to cancel requests using the `cancelRequest` method and to obtain per-iteration and per-request statistics using the `getLatestIterationStats`.

### The Request Class

The `Request` class is used to define properties of the request, such as the input token ids and the maximum number of tokens to generate. The `streaming` parameter can be used to indicate if the request should generate a response for each new generated tokens (`streaming = true`) or only after all tokens have been generated (`streaming = false`). Other mandatory parameters of the request include the sampling configuration (defined by the `SamplingConfig` class) which contains parameters controlling the decoding process and the output configuration (defined by the `OutputConfig` class) which controls what information should be included in the `Result` for a particular response.

Optional parameters can also be provided when constructing a request such as a list of bad words, a list of stop words, a client id, or configurations objects for prompt tuning, LoRA, or speculative decoding, or a number of sequences to generate for example.

### The Response Class

The `awaitResponses` method of the `Executor` class returns a vector of responses. Each response contains the request id associated with this response, and also contains either an error or a `Result`. Check if the response has an error by using the `hasError` method before trying to obtain the `Result` associated with this response using the `getResult` method.

### The Result Class

The `Result` class holds the result for a given request. It contains a Boolean parameter called `isFinal` that indicates if this is the last `Result` that will be returned for the given request id. It also contains the generated tokens. If the request is configured with `streaming = false` and `numReturnSequences = 1`, a single response will be returned, the `isFinal` Boolean will be set to `true` and all generated tokens will be included in the `outputTokenIds`. If `streaming = true` and `numReturnSequences = 1` is used, a `Result` will include one or more tokens (depending on the request `returnAllGeneratedTokens` parameter) except the last result and the `isFinal` flag will be set to `true` for the last result associated with this request.

The request `numReturnSequences` parameter controls the number of output sequences to generate for each prompt. When this option is used, the Executor will return at least `numReturnSequences` responses for each request, each containing one Result. In beam search (`beamWidth > 1`), the number of beams to be returned will be limited by `numReturnSequences` and the `sequenceIndex` attribute of the `Result` class will always be zero. Otherwise, in sampling (`beamWidth = 1`), the `sequenceIndex` attribute indicates the index of the generated sequence in the result (`0 <= sequenceIndex < numReturnSequences`). It contains a Boolean parameter called `isSequenceFinal` that indicates if this is the last result for the sequence and also contains a Boolean parameter `isFinal` that indicates when all sequences for the request have been generated. When `numReturnSequences = 1`, `isFinal` is identical to `isSequenceFinal`.

Here is an example that shows how a subset of 3 responses might look like for `numReturnSequences = 3`:

```
Response 1: requestId = 1, Result with sequenceIndex = 0, isSequenceFinal = false, isFinal = false
Response 2: requestId = 1, Result with sequenceIndex = 1, isSequenceFinal = true,  isFinal = false
Response 3: requestId = 1, Result with sequenceIndex = 2, isSequenceFinal = false, isFinal = false
```

In this example, each response contains one result for different sequences. The `isSequenceFinal` flag of the second Result is set to true, indicating that it is the last result for `sequenceIndex = 1`, however, the isFinal flag of each Response is set to false because sequences 0 and 2 are not completed.

### Sending Requests with Different Beam Widths

The executor can process requests with different beam widths if the following conditions are met:

- The model was built with a `max_beam_width > 1`.
- The executor is configured with a `maxBeamWidth > 1` (the configured `maxBeamWidth` must be less than or equal to the model's `max_beam_width`).
- The requested beam widths are less than or equal to the configured `maxBeamWidth`.

The executor may schedule successive requests with the same beam width at the same time. For successive requests with two different beam widths, `x` and `y`, requests with beam width `y` are not scheduled until all requests with beam width `x` have been processed.
This allows the runtime to reconfigure itself for a new beam width when no requests are in flight. The reconfiguration happens automatically each time requests with a different beam width than currently configured are detected. Waiting for previous requests to finish and reconfiguring the runtime may cause significant overhead and reduce overall throughput.

### Controlling output with Logits Post-Processor

Optionally, you can alter the logits produced by the network by providing an instance of `Executor::LogitsPostProcessorConfig`. For instance, this feature can be used to generate JSON formatted output. {cpp:class}`Executor::LogitsPostProcessorConfig <tensorrt_llm::executor::LogitsPostProcessorConfig>` specifies a map of named callbacks in the following form

```cpp
std::unordered_map<std::string, function<Tensor(IdType, Tensor&, BeamTokens const&, StreamPtr const&, std::optional<IdType>)>>
```

The map key is the name associated with that logits post-processing callback. Each request can then specify the name of the logits post-processor to use for that particular request, if any.

The first argument to the callback is the request id, second is the logits tensor, third are the tokens produced by the request so far, fourth is the operation stream used by the logits tensor, and last one is an optional client id. The callback returns a modified tensor of logits. Multiple requests can share same client id and callback can use different logic based on client id.

You must use the stream to access the logits tensor. For example, to perform an addition with a bias tensor, the addition operation is enqueued on that stream. Alternatively, you can call `stream->synchronize()`, however, that will slow down the entire execution pipeline.

The executor also includes a {cpp:class}`LogitsPostProcessorBatched <tensorrt_llm::executor::LogitsPostProcessorBatched>` method that enables altering logits of multiple requests in a batch. The batched method allows further optimizations and reduces callback overheads.

```cpp
std::function<void(std::vector<IdType> const&, std::vector<Tensor>&, std::vector<std::reference_wrapper<BeamTokens const>> const&, StreamPtr const&, std::vector<std::optional<IdType>> const&)>
```

A single batched callback can be specified in `LogitsPostProcessorConfig`. Each request can opt to apply this callback by specifying the name of the logits post-processor as `Request::kBatchedPostProcessorName`.

Note: Neither callback variant is supported with the `STATIC` batching type for the moment.

In a multi-GPU run, the callback is invoked on all ranks in the first tensor-parallel group, by default. To ensure correct execution, replicate the client-side state that is accessed by the callback on these ranks. If replication is expensive or infeasible, use `LogitsPostProcessorConfig::setReplicate(false)` to invoke the callback only on rank 0. The executor broadcasts the sampled tokens internally to ensure correct execution.

### Structured output with guided decoding
Guided decoding controls the generation outputs to be amenable to pre-defined structured formats, e.g., JSON or XML. Currently, guided decoding is supported with the [XGrammar](https://github.com/mlc-ai/xgrammar) backend.

To enable guided decoding, a valid instance of `GuidedDecodingConfig` must be provided when constructing `Executor`. `GuidedDecodingConfig` should be constructed with some tokenizer information, including `encodedVocab`, `tokenizerStr` (optional) and `stopTokenIds` (optional). Given a Hugging Face tokenizer, these can be extracted by:

```python
encoded_vocab = tokenizer.get_vocab()
encoded_vocab = [token for token, _ in sorted(encoded_vocab.items(), key=lambda x: x[1])]
tokenizer_str = tokenizer.backend_tokenizer.to_str()
stop_token_ids = [tokenizer.eos_token_id]
```

Refer to [`tensorrt_llm/llmapi/tokenizer.py`](source:tensorrt_llm/llmapi/tokenizer.py) for more details. You may dump these materials to disk, and reload them to C++ runtime for use.

Each request can be optionally specified with a `GuidedDecodingParams`, which defines the desired structured format. Currently, it supports four types:
* `GuidedDecodingParams::GuideType::kJSON`: The generated text is amenable to JSON format;
* `GuidedDecodingParams::GuideType::kJSON_SCHEMA`: The generated text is amenable to JSON format with additional restrictions;
* `GuidedDecodingParams::GuideType::kREGEX`: The generated text is amenable to regular expression;
* `GuidedDecodingParams::GuideType::kEBNF_GRAMMAR`: The generated text is amenable to the extended Backus-Naur form (EBNF) grammar.

The latter three types should be used with the schema/regex/grammar provided to `GuidedDecodingParams`.

### Obtaining Arbitrary Output Tensors
The executor API gives the user the possibility to read the arbitrary outputs from the model. For example, it is possible to obtain hidden states or logits.

#### Mark Tensors As Output
For a tensor to be obtainable using this feature, it needs to be marked as an output in the model definition (e.g. add `topk_logits.mark_output("TopKLogits")`) before building the TRT engine.

#### Configure The Executor
Assuming the TensorRT engine you are planning to use has a tensor named `TopKLogits` marked as output, you should then configure the `Executor` to read from this output tensor by passing its name to the `ExecutorConfig` configuration object:
```cpp
auto const executorConfig = ExecutorConfig{};

std::vector<executor::AdditionalModelOutput> additionalOutputs{
    executor::AdditionalModelOutput{"TopKLogits", /*whether or not to get the output for the context too */ true}};
executorConfig.setAdditionalModelOutputs(additionalOutputs);

// ... set more configuration options if needed
// ... create the `Executor` instance
```

### Request Additional Output
Construct a request to enqueue in the executor to query this tensor output:
```cpp
executor::Request request{requestTokens, parameters.maxOutputLength, true, executor::SamplingConfig{},
    executor::OutputConfig{false, false, false, true, false, false, additionalOutputs}};
executor.enqueueRequest(request);
```

The output can be found at the `additionalOutputs` property of each response.

#### Note on context outputs

If KV cache reuse is enabled, context outputs will not contain outputs for the part of the context that has been reused. This part of the outputs can only be obtained from the prior request with the same prefix that generated this part of the KV cache.

## C++ Executor API Example

Two C++ examples are provided that shows how to use the Executor API and can be found in the [`examples/cpp/executor`](source:examples/cpp/executor/) folder.

## Python Bindings for the Executor API

Python bindings for the Executor API are also available to use the Executor API from Python. The Python bindings are defined in [bindings.cpp](source:cpp/tensorrt_llm/pybind/executor/bindings.cpp) and once built, are available in package `tensorrt_llm.bindings.executor`. Running `'help('tensorrt_llm.bindings.executor')` in a Python interpreter will provide an overview of the classes available.

In addition, three Python examples are provided to demonstrate how to use the Python bindings to the Executor API for single and multi-GPU models. They can be found in [`examples/bindings`](source:examples/bindings).

## In-flight Batching with the Triton Inference Server

A Triton Inference Server C++ [backend](https://github.com/triton-inference-server/tensorrtllm_backend) is provided with TensorRT-LLM that
includes the mechanisms needed to serve models using in-flight batching. That
backend is also a good starting example of how to implement in-flight batching using
the TensorRT-LLM C++ Executor API.
