# In-flight Batching in TensorRT-LLM

TensorRT-LLM supports in-flight batching of requests (also known as continuous
batching or iteration-level batching). It is a technique that aims at reducing
wait times in queues, eliminating the need for padding requests and allowing
for higher GPU utilization.

In more details, this feature allows for the inclusion of newly arrived
requests and the return of newly completed requests at each iteration of the
token generation loop. In-flight batching is accessed via a Tensor-RT component
called the *Batch Manager*. That batch manager exposes hooks for the user to
register function pointers to define how TensorRT-LLM reads in new requests and
how it returns completed requests to the user.

## The Batch Manager API

A software component (called the client in the text that follows) can interact
with the batch manager using two main callbacks. Their signatures are defined
in the [`callbacks.h`](../cpp/include/tensorrt_llm/batch_manager/callbacks.h) file.

### Get and Send Callbacks

The entry point to pass new requests to the batch manager is a callback of type
`GetInferenceRequestsCallback`. An implementation of that callback must return
a list of requests (`std::list<std::shared_ptr<InferenceRequest>`) to be
processed by the batch manager. It takes a parameter indicating the maximum
number of requests that can be accepted (a negative value indicates that an
unbounded number of requests can be accepted). The complete signature of that
callback is:

```cpp
using GetInferenceRequestsCallback = std::function<std::list<std::shared_ptr<InferenceRequest>>(int32_t)>;
```

For each new request, the client must provide the batch manager with its input
tensors and a 64-bit unsigned number (`uint64_t`) that will uniquely identify
the request. That identifier is called the *correlation ID* in the text that
follows (and in the code of the batch manager). The input tensors are collected
in a map (`std::map<std::string, Tensor>`) that associates input names to
tensor. See
[`InferenceRequest.h`](../cpp/include/tensorrt_llm/batch_manager/InferenceRequest.h)
for more details.

The responses are delivered to the client through a callback of type
`SendResponseCallback`. A conforming callback must accept the 64-bit
correlation ID that uniquely identifies the request, the list of output tensors,
a boolean (identifying the last response for the request when set to
`true`) and a potentially non-empty error message.
A non-empty error message indicates that an error has been encountered.
In that case, the boolean indicating that this is the last response will be set to true,
and the callback must properly handle the error.
Its signature is:

```cpp
using SendResponseCallback = std::function<void(uint64_t, std::list<std::shared_ptr<Tensor>> const&, bool, const std::string&)>;
```

Note that the batch manager will reject any request sent using the
`GetInferenceRequestsCallback` callback if the correlation ID passed by the
client corresponds to the correlation ID of a request that is being processed
by the batch manager.  A correlation ID can be reused after it appears in a
call to the `SendResponseCallback` callback marked as final (third argument set
to `true`).

### GPTManager Creation

The TensorRT-LLM Triton backend is a good starting example how to implement
+in-flight batching using the TensorRT-LLM batch manager.


In a more realistic case, the inference server will likely manage a queue of
active work items that will be populated with the requests actively processed
by the server.  The batch manager will execute in a worker thread and will
receive requests to process through the `GetInferenceRequests` callback. The
server (or the model instance) will retire requests from its queue of work
items when notified of a completion by the batch manager through the
`SendResponse` callback.  The instance of the batch manager to serve an
auto-regressive model like GPT can be created as follows:

```cpp
#include <tensorrt_llm/batch_manager/GptManager.h>

using namespace tensorrt_llm::batch_manager;

GptManager batchManager(pathToTrtEngine,                   // Path to the TensorRT engine of the model,
                        TrtGptModelType::InflightBatching, // Use in-flight batching,
                        maxSeqLen,                         // Maximum sequence length,
                        maxNumRequests,                    // Maximum number of requests,
                        getInferenceRequestsCb,            // The Get callback (see above),
                        sendResponseCb);                   // The Send callback (see above).
```

## In-flight Batching with the Triton Inference Server

A Triton Inference Server C++ backend is provided with TensorRT-LLM that
includes the mechanisms needed to serve models using in-flight batching.
