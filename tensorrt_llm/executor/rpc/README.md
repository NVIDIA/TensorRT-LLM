# A Lightweight RPC
This is a pure-Python lightweight RPC we build to simplify our existing IPC code in the orchestrator part. It provides multiple call modes (sync, async, future, streaming) and supports both IPC and TCP connections.

## Examples
### Create Server and Client

```python
from tensorrt_llm.executor.rpc import RPCServer, RPCClient

# Define your application
class App:
    def add(self, a: int, b: int) -> int:
        return a + b
    
    async def async_multiply(self, x: int, y: int) -> int:
        return x * y

# Create and start server
app = App()
with RPCServer(app) as server:
    server.bind("ipc:///tmp/my_rpc")  # or "tcp://127.0.0.1:5555"
    server.start()
    
    # Create client and make calls
    with RPCClient("ipc:///tmp/my_rpc") as client:
        result = client.add(5, 3).remote()
        print(result)  # Output: 8
```

### Different Remote Calls

#### Synchronous Call
```python
# Blocking call that waits for result
result = client.add(10, 20).remote()
# or with timeout
result = client.add(10, 20).remote(timeout=5.0)
```

#### Asynchronous Call
```python
# Async call that returns a coroutine
result = await client.async_multiply(3, 4).remote_async()
```

#### Future-based Call
```python
# Returns a concurrent.futures.Future
future = client.add(1, 2).remote_future()
# Get result later
result = future.result()
```

#### Fire-and-Forget Call
```python
# Send request without waiting for response
client.submit_task(task_id=123).remote(need_response=False)
```

#### Streaming Call
```python
# For async generator methods
async for value in client.stream_data(n=10).remote_streaming():
    print(f"Received: {value}")
```

### Error Handling
```python
from tensorrt_llm.executor.rpc import RPCError, RPCTimeout

try:
    result = client.risky_operation().remote(timeout=1.0)
except RPCTimeout:
    print("Operation timed out")
except RPCError as e:
    print(f"RPC Error: {e}")
    print(f"Original cause: {e.cause}")
    print(f"Traceback: {e.traceback}")
```

### Graceful Shutdown
```python
# Shutdown server from client
client.shutdown_server()
```
