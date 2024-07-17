(executor)=

# Executor API

TensorRT-LLM includes a high-level C++ API called the Executor API which allows you to execute requests
asynchronously, with in-flight batching, and without the need to define callbacks.

A software component (referred to as "the client" in the text that follows) can interact
with the executor using the API defined in the [`executor.h`](source:cpp/include/tensorrt_llm/executor/executor.h) file.
For details about the API, refer to the {ref}`_cpp_gen/executor.rst`.

The following sections provide an overview of the main classes defined in the Executor API.

### The Executor Class

The `Executor` class is responsible for receiving requests from the client, and providing responses for those requests. The executor is constructed by providing a path to a directory containing the TensorRT-LLM engine or buffers containing the engine and the model JSON configuration. The client can create requests and enqueue those requests for execution using the `enqueueRequest` or `enqueueRequests` methods of the `Executor` class. Enqueued requests will be scheduled for execution by the executor, and multiple independent requests can be batched together at every iteration of the main execution loop (a process often referred to as continuous batching or iteration-level batching). Responses for a particular request can be awaited for by calling the `awaitResponses` method, and by providing the request id. Alternatively, responses for any requests can be awaited for by omitting to provide the request id when calling `awaitResponses`. The `Executor` class also allows to cancel requests using the `cancelRequest` method and to obtain per-iteration and per-request statistics using the `getLatestIterationStats`.

#### Logits Post-Processor (optional)

Users can alter the logits produced the network, by providing a map of named callbacks of the form:

```
std::unordered_map<std::string, function<Tensor(IdType, Tensor&, BeamTokens const&, StreamPtr const&)>>
```
to the `ExecutorConfig`. The map key is the name associated with that logits post-processing callback. Each request can then specify the name of the logits post-processor to use for that particular request, if any.

The first argument to the callback is the request id, second is the logits tensor, third are the tokens produced by the request so far, and last one is the operation stream used by the logits tensor. The callback returns a modified tensor of logits.

Users *must* use the stream to access the logits tensor. For example, performing a addition with a bias tensor should be enqueued on that stream.
Alternatively, users may call `stream->synchronize()`, however, that will slow down the entire execution pipeline.

We also provide a batched version that allows altering logits of multiple requests in a batch. This allows further optimizations and reduces callback overheads.

```
std::function<void(std::vector<IdType> const&, std::vector<Tensor>&, std::vector<std::reference_wrapper<BeamTokens const>> const&, StreamPtr const&)>
```

A single batched callback can be specified in `ExecutorConfig`. Each request can opt to apply this callback by specifying the name of the logits
post-processor as `Request::kBatchedPostProcessorName`.

Note: Both callback variants are not supported with the `STATIC` batching type for the moment.

### The Request Class

The `Request` class is used to define properties of the request, such as the input token ids and the maximum number of tokens to generate. The `streaming` parameter can be used to indicate if the request should generate a response for each new generated tokens (`streaming = true`) or only after all tokens have been generated (`streaming = false`). Other mandatory parameters of the request include the sampling configuration (defined by the `SamplingConfig` class) which contains parameters controlling the decoding process and the output configuration (defined by the `OutputConfig` class) which controls what information should be included in the `Result` for a particular response.

Optional parameters can also be provided when constructing a request such as a list of bad words, a list of stop words, or configurations objects for prompt tuning, LoRA, or speculative decoding for example.

### The Response Class

The `awaitResponses` method of the `Executor` class returns a vector of responses. Each response contains the request id associated with this response, and also contains either an error or a `Result`. Check if the response has an error by using the `hasError` method before trying to obtain the `Result` associated with this response using the `getResult` method.

### The Result Class

The `Result` class holds the result for a given request. It contains a Boolean parameter called `isFinal` that indicates if this is the last `Result` that will be returned for the given request id. It also contains the generated tokens. If the request is configured with `streaming = false`, the `isFinal` Boolean will be set to `true` and all generated tokens will be included in the `outputTokenIds`. If `streaming = true` is used, a `Result` will only include 1 token and the `isFinal` flag will be set to `true` for the last result associated with this request.

## C++ Executor API Example

Two C++ examples are provided that shows how to use the Executor API and can be found in the [`examples/cpp/executor`](source:examples/cpp/executor/) folder.

## Python Bindings for the Executor API

Python bindings for the Executor API are also available to use the Executor API from Python. The Python bindings are defined in [bindings.cpp](source:cpp/tensorrt_llm/pybind/executor/bindings.cpp) and once built, are available in package `tensorrt_llm.bindings.executor`. Running `'help('tensorrt_llm.bindings.executor')` in a Python interpreter will provide an overview of the classes available.

In addition, three Python examples are provided to demonstrate how to use the Python bindings to the Executor API for single and multi-GPU models. They can be found in [`examples/bindings`](source:examples/bindings).
