# Introduction to KV Cache Transmission Module

This article provides a general overview of the components used for device-to-device transmission of KV cache, which is relied upon by dist-serving. It is intended as a reference for users who wish to understand the internal implementation or develop extended functionalities.

## Workflow

A diagram needs to be added here.

1. Context phase completes computation, KV cache stays in device memory awaiting transmission.
2. Context returns its communicator handle to the user, who selects the generation executor for continued communication.
3. If no prior connection exists, it's established now. Generation phase shares its cache layout with context.
4. Generation phase requests KV cache for specific tokens.
5. Context sends KV cache to generation phase.
6. Generation phase resumes computation, context releases KV cache.

## Key Components

### Transceiver

Responsible for coordinating the sending and receiving of cache among different ranks within the same executor.

### Sender and Receiver

Responsible for handling and exchanging control information required during cache transmission, such as the RequestInfo data structure.

### Formatter

Performs KV cache data transmission and correctly handles the mapping between caches across different TP/PP configurations.

### Connection

Bidirectional byte-stream protocol facility. Apart from essential operations such as connection establishment, it mainly provides send and receive functionalities. UCX accesses the system through this facility. The `AgentConnection` data structure adapts the upper-layer bidirectional send/receive semantics into a unidirectional read/write operation model.

### Transfer Agent

Unidirectional byte-stream read/write protocol facility. Apart from essential operations such as connection establishment, it primarily provides read and write functionalities. NIXL accesses the system through this facility.

## Customization Method

### Encapsulation and Overloading of Low-Level Communication Libraries

Each layer of interface described in the previous section supports overloading. Here, based on whether the underlying library uses a unidirectional or bidirectional protocol, we describe the customization methods respectively.

#### Unidirectional Communication

If the underlying library you are integrating uses a unidirectional communication model, with read/write as its primary interfaces, you should inherit the `executor::kv_cache::BaseTransferAgent` data structure. This structure mainly provides interfaces for memory registration, remote agent loading, and transfer request submission.

#### Bidirectional Communication

If the underlying library you are integrating uses a bidirectional communication model, you should inherit the `executor::kv_cache::Connection` data structure. This structure mainly provides send and receive interfaces.

### Modifications to Upper-Level Runtime Logic

TBD.
