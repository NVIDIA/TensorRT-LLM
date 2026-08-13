# System-Level Patterns

---

## GC_MGMT: Manage Garbage Collection for Inference

Python's cyclic GC causes unpredictable pauses (~1-10ms). For inference servers with large long-lived model objects, GC scanning is particularly wasteful — the collector repeatedly inspects millions of tensor/parameter objects that will never be freed. See [../hotspot-classification.md](../hotspot-classification.md) (GC section) for detection heuristics.

**Pattern 1: gc.freeze() after model load**
```python
import gc

# After model and KV cache are fully initialized:
model = load_model(config)
kv_pool = allocate_kv_cache(model, max_batch)
gc.collect()        # Clean up initialization temporaries
gc.freeze()         # Freeze all current objects — GC won't scan them

# New allocations (per-request objects) are still tracked by GC.
# Long-lived model weights and cache buffers are exempt from scanning.
```

**Impact:** Reduces GC pause time proportionally to the number of frozen objects. For a 70B model with millions of parameter objects, this can reduce GC scan time from ~10ms to <1ms.

**Pattern 2: gc.disable() during latency-sensitive sections**
```python
import gc

gc.disable()
try:
    for batch in inference_batches:
        output = model.forward(batch)    # No GC pauses during inference
        handle_output(output)
finally:
    gc.enable()
    gc.collect()    # Explicit collection at known safe point
```

**When to use:** Only for bounded sections where temporary object count is manageable. Do NOT disable GC for unbounded loops — memory will grow.

**Pattern 3: gc.set_threshold() tuning**
```python
import gc

# Default: (700, 10, 10) — gen-0 collection every 700 allocations
# For inference (few short-lived cycles, many long-lived objects):
gc.set_threshold(50000, 10, 10)   # Less frequent gen-0 collection
```

**When to use:** When gc.freeze() is not an option (e.g., model objects are modified during serving). Less effective than freeze but lower risk.

---

## GIL_CONTENTION: Reduce Threading Overhead

GIL contention and lock overhead in multi-threaded TRT-LLM components (request processing, tokenization threads, async output handling). See [../hotspot-classification.md](../hotspot-classification.md) (GIL section) for detection heuristics.

**Pattern 1: Fine-grained locking**
```python
# BEFORE: single coarse lock serializes all threads
global_lock = threading.Lock()
def process(req):
    with global_lock:           # All threads serialize here
        config = read_config()
        result = compute(config)  # Expensive, holds lock!
        update_state(result)

# AFTER: separate locks for separate resources
config_lock = threading.Lock()
state_lock = threading.Lock()
def process(req):
    with config_lock:
        config = read_config()  # Short critical section
    result = compute(config)    # No lock during compute
    with state_lock:
        update_state(result)    # Short critical section
```

**Pattern 2: Batch work under single lock acquisition**
```python
# BEFORE: lock convoy — acquire/release per item
lock = threading.Lock()
for item in items:
    with lock:
        tiny_operation(item)    # Threads convoy: wake -> acquire -> release

# AFTER: batch under single acquisition
with lock:
    for item in items:
        tiny_operation(item)    # One acquisition for all items
```

**Pattern 3: Thread-local storage to avoid contention**
```python
# BEFORE: shared buffer with lock
shared_buffer = []
lock = threading.Lock()
def process():
    with lock:
        shared_buffer.append(result)

# AFTER: thread-local buffers, merge at sync points
thread_data = threading.local()
def process():
    if not hasattr(thread_data, 'buffer'):
        thread_data.buffer = []
    thread_data.buffer.append(result)   # No contention
```

---

## SERIALIZE: Optimize Serialization in Hot Paths

Serialization overhead from pickle, JSON, or protobuf in request processing or IPC paths. See [../hotspot-classification.md](../hotspot-classification.md) (SERIALIZE section) for detection heuristics.

**Pattern 1: Avoid pickle for large tensors in IPC**
```python
# BEFORE: pickle serializes tensor metadata + data (~ms for large tensors)
import pickle
data = pickle.dumps(tensor)       # Copies entire tensor to bytes
send(data)

# AFTER: raw buffer transfer for same-architecture IPC
raw = tensor.cpu().numpy().tobytes()   # Raw bytes, no pickle overhead
send(raw)
# Reconstruct:
arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
tensor = torch.from_numpy(arr)
```

**Pattern 2: Use SharedMemory for same-node IPC**
```python
# BEFORE: multiprocessing.Queue pickles everything
from multiprocessing import Queue
q = Queue()
q.put(large_array)               # Pickles! Copies data!

# AFTER: SharedMemory (zero-copy, Python 3.8+)
from multiprocessing.shared_memory import SharedMemory
shm = SharedMemory(create=True, size=array.nbytes, name='kv_buffer')
shared_arr = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
shared_arr[:] = array[:]          # One copy into shared memory
# Consumer: np.ndarray(shape, dtype, buffer=SharedMemory(name='kv_buffer').buf)
```

**Pattern 3: Use msgpack for structured metadata**
```python
# BEFORE: json.dumps for request metadata (~100-500us for complex dicts)
data = json.dumps(request_metadata)

# AFTER: msgpack (5-10x faster than json, 2-3x faster than pickle for dicts)
import msgpack
data = msgpack.packb(request_metadata)
metadata = msgpack.unpackb(data)
```

**Impact:** msgpack is 5-10x faster than JSON for structured data. SharedMemory eliminates serialization entirely for large buffers.
