# Plan: Dynamic MAX_SEQUENCE_LENGTH for Suffix Automaton

## Goal

Make `MAX_SEQUENCE_LENGTH` configurable at runtime instead of a compile-time constant. This will allow:
- Smaller memory footprint for models with shorter context lengths (e.g., 8K, 32K)
- Better resource utilization - no need to allocate 256K token capacity for a 4K context model

## Approach

**Replace the static implementation entirely** with a dynamic version. No backward compatibility layer — cleaner code, single code path.

## Current State

The suffix automaton uses fixed-size arrays determined at compile time:

```cpp
// saConfig.h
#define SA_MAX_SEQUENCE_LENGTH 262144  // 256K tokens

// suffixAutomaton.h
struct SuffixAutomaton {
    using Graph = SAFlatGraph<Token, NodeData, 2 * MAX_SEQUENCE_LENGTH>;  // 512K nodes
    using TokenVec = SADynamicBuffer<Token, MAX_SEQUENCE_LENGTH>;          // 256K tokens
    // ...
};
```

**Memory per slot**: ~61 MB (fixed, regardless of actual sequence length needed)

---

## Phase 1: Convert Buffer Types to Dynamic

**Files**: `saBuffer.h`

**Objective**: Replace static `SABuffer<T, Size>` and `SADynamicBuffer<T, Size>` with runtime-sized versions.

### Changes

1. Replace `SABuffer` with pointer-based version:

```cpp
// BEFORE (static)
template <typename T, size_t Size, typename IndexT = size_t>
struct SABuffer {
    std::array<T, Size> mData;
    // ...
};

// AFTER (dynamic)
template <typename T, typename IndexT = size_t>
struct SABuffer {
    T* mData;           // Pointer to external memory
    size_t mCapacity;   // Runtime capacity
    
    SA_CUDA_CALLABLE T const& at(IndexT row) const {
        assert(static_cast<size_t>(+row) < mCapacity);
        return mData[+row];
    }
    SA_CUDA_CALLABLE T& at(IndexT row) {
        assert(static_cast<size_t>(+row) < mCapacity);
        return mData[+row];
    }
    SA_CUDA_CALLABLE size_t size() const { return mCapacity; }
    // ... other methods (same interface)
};
```

2. Replace `SADynamicBuffer` similarly:

```cpp
// BEFORE (static)
template <typename T, size_t Size, typename IndexT = size_t>
struct SADynamicBuffer {
    IndexT mLength{0};
    SABuffer<T, Size, IndexT> mData;
};

// AFTER (dynamic)
template <typename T, typename IndexT = size_t>
struct SADynamicBuffer {
    T* mData;           // Pointer to external memory
    size_t mCapacity;   // Runtime capacity
    IndexT mLength{0};  // Current length
    
    SA_CUDA_CALLABLE void clear() { mLength = IndexT(0); }
    SA_CUDA_CALLABLE T& pushBack(T const& value) {
        assert(static_cast<size_t>(+mLength) < mCapacity);
        T& result = mData[+mLength];
        result = value;
        mLength = IndexT(+mLength + 1);
        return result;
    }
    SA_CUDA_CALLABLE bool hasCapacity() const { return +mLength < mCapacity; }
    // ... other methods (same interface)
};
```

### Key Points
- Same method names/signatures — minimal changes to callers
- Must remain `trivially_copyable` (pointer + primitives only)
- Remove the `Size` template parameter entirely

---

## Phase 2: Convert Graph Types to Dynamic

**Files**: `saFlatGraph.h`, `saFlatHashMap.h`

**Objective**: Update graph types to use the new dynamic buffers.

### Changes to `saFlatHashMap.h`

Replace `SAHashMapAllocator` to use runtime size:

```cpp
// BEFORE (static)
template <typename Key, typename Value, size_t MaxSizeBytes>
struct SAHashMapAllocator {
    SADynamicBuffer<char, MaxSizeBytes, Ptr> mMem;
    // ...
};

// AFTER (dynamic)
template <typename Key, typename Value>
struct SAHashMapAllocator {
    using HashMap = SAFlatHashMap<Key, Value, SAHashMapAllocator>;
    
    char* mMemory;       // Pointer to external memory
    size_t mCapacity;    // Runtime capacity in bytes
    size_t mUsed{0};     // Current usage
    
    SA_CUDA_CALLABLE Ptr alloc(size_t capacity) {
        // Same logic, but check against mCapacity instead of MaxSizeBytes
    }
    // ... same interface
};
```

### Changes to `saFlatGraph.h`

Remove the `MaxSize` template parameter:

```cpp
// BEFORE (static)
template <typename Key, typename NodeData, size_t MaxSize>
struct SAFlatGraph {
    using Allocator = SAHashMapAllocator<Key, NodeIndex, 10 * MaxSize * (...)>;
    SADynamicBuffer<Node, MaxSize, NodeIndex> mNodes;
    // ...
};

// AFTER (dynamic)
template <typename Key, typename NodeData>
struct SAFlatGraph {
    using Allocator = SAHashMapAllocator<Key, NodeIndex>;
    using HashMap = typename Allocator::HashMap;
    
    struct Node { /* unchanged */ };
    
    SADynamicBuffer<Node, NodeIndex> mNodes;
    Allocator mAllocator;
    
    // Same interface — no changes to method signatures
};
```

---

## Phase 3: Convert SuffixAutomaton to Dynamic

**Files**: `suffixAutomaton.h`

**Objective**: Update `SuffixAutomaton` to use runtime-sized components.

### Changes

1. Update `SuffixAutomaton` struct (replace, not add):

```cpp
// BEFORE (static)
struct SuffixAutomaton {
    using Graph = SAFlatGraph<Token, NodeData, 2 * SAConfig::MAX_SEQUENCE_LENGTH>;
    using TokenVec = SADynamicBuffer<Token, SAConfig::MAX_SEQUENCE_LENGTH, TextIndex>;
    
    Graph mStates;
    TokenVec mTokens;
    NodeIndex mLast;
    // ...
};

// AFTER (dynamic)
struct SuffixAutomaton {
    using Graph = SAFlatGraph<Token, NodeData>;
    using TokenVec = SADynamicBuffer<Token, TextIndex>;
    
    Graph mStates;
    TokenVec mTokens;
    NodeIndex mLast;
    size_t mMaxSeqLen;  // Store for reference
    
    // Initialize with external memory and max sequence length
    void init(void* memory, size_t maxSeqLen);
    
    // Calculate required memory size for a given max_seq_len
    static size_t getRequiredMemorySize(size_t maxSeqLen);
    
    // Existing interface unchanged
    SA_CUDA_CALLABLE void extend(Token token);
    SA_CUDA_CALLABLE SAOptional<LookupResult> lookup() const;
    SA_CUDA_CALLABLE SAOptional<LookupResult> lookupFixed(int targetLen) const;
    SA_CUDA_CALLABLE void getDraftTokens(Token::ValueType* buf, int bufLen, TextIndex startPos) const;
};
```

2. Add memory size calculation:

```cpp
// In suffixAutomaton.cpp or as inline
size_t SuffixAutomaton::getRequiredMemorySize(size_t maxSeqLen) {
    size_t maxNodes = 2 * maxSeqLen;
    
    size_t headerSize = sizeof(SuffixAutomaton);
    size_t nodeSize = maxNodes * sizeof(Graph::Node);
    size_t tokenSize = maxSeqLen * sizeof(Token);
    size_t allocatorSize = 10 * maxNodes * (sizeof(Token) + sizeof(NodeIndex));
    
    // Add alignment padding (64-byte for GPU cache lines)
    // Layout: [header][nodes][tokens][allocator]
    return alignUp(headerSize, 64) + alignUp(nodeSize, 64) + 
           alignUp(tokenSize, 64) + alignUp(allocatorSize, 64);
}
```

3. Add init function:

```cpp
// Memory layout for a single slot:
// [SuffixAutomaton header (with pointers)][nodes data][tokens data][allocator data]
//
// The header contains pointers that point INTO the same memory block.
// This works because:
// - On host: init() sets up pointers using the host base address
// - Before GPU copy: relocate() adjusts pointers to GPU addresses
// - On GPU: pointers are valid GPU addresses

void SuffixAutomaton::init(void* base, size_t maxSeqLen) {
    mMaxSeqLen = maxSeqLen;
    size_t maxNodes = 2 * maxSeqLen;
    
    // Data starts after the header
    uint8_t* ptr = static_cast<uint8_t*>(base) + alignUp(sizeof(SuffixAutomaton), 64);
    
    // Layout: [header][nodes][tokens][allocator memory]
    mStates.mNodes.mData = reinterpret_cast<Graph::Node*>(ptr);
    mStates.mNodes.mCapacity = maxNodes;
    mStates.mNodes.mLength = NodeIndex(0);
    ptr += alignUp(maxNodes * sizeof(Graph::Node), 64);
    
    mTokens.mData = reinterpret_cast<Token*>(ptr);
    mTokens.mCapacity = maxSeqLen;
    mTokens.mLength = TextIndex(0);
    ptr += alignUp(maxSeqLen * sizeof(Token), 64);
    
    mStates.mAllocator.mMemory = reinterpret_cast<char*>(ptr);
    mStates.mAllocator.mCapacity = 10 * maxNodes * (sizeof(Token) + sizeof(NodeIndex));
    mStates.mAllocator.mUsed = 0;
    
    // Reset state
    mLast = NodeIndex(0);
}

// Relocate pointers from host addresses to GPU addresses
void SuffixAutomaton::relocate(void* oldBase, void* newBase) {
    ptrdiff_t delta = static_cast<uint8_t*>(newBase) - static_cast<uint8_t*>(oldBase);
    
    mStates.mNodes.mData = reinterpret_cast<Graph::Node*>(
        reinterpret_cast<uint8_t*>(mStates.mNodes.mData) + delta);
    mTokens.mData = reinterpret_cast<Token*>(
        reinterpret_cast<uint8_t*>(mTokens.mData) + delta);
    mStates.mAllocator.mMemory = reinterpret_cast<char*>(
        reinterpret_cast<uint8_t*>(mStates.mAllocator.mMemory) + delta);
}
```

### Key Points
- Same struct name `SuffixAutomaton` — no "Dynamic" suffix
- Same method signatures for `extend()`, `lookup()`, etc.
- `trivially_copyable` constraint still satisfied (all pointer + primitives)
- `relocate()` function adjusts pointers when copying between host and GPU

---

## Phase 4: Update Kernel Functions

**Files**: `suffixAutomatonKernels.h`, `suffixAutomatonKernels.cu`

**Objective**: Update kernel functions to accept `maxSeqLen` parameter.

### Changes to `suffixAutomatonKernels.h`

1. Update existing param struct to include `maxSeqLen`:

```cpp
struct SuffixAutomatonExtendParams {
    int batchSize{0};
    int draftLength{0};
    int maxSlots{0};
    int maxSeqLen{0};                    // NEW: runtime sequence length
    void* slots{nullptr};
    int const* batchIndices{nullptr};
    int* matchLenOut{nullptr};
    int* draftTokensOut{nullptr};
    int const* acceptedTokensIn{nullptr};
    int const* acceptedLensIn{nullptr};
    
    void checkParams() const;
};
```

2. Update helper functions:

```cpp
// Get state size for a given max_seq_len (replaces static STATE_SIZE_BYTES)
size_t getSuffixAutomatonStateSize(size_t maxSeqLen);

// Initialize automaton at given memory location
void initAutomaton(void* memory, size_t maxSeqLen);

// Remove: MAX_SEQUENCE_LENGTH constant (no longer exposed)
// Remove: STATE_SIZE_BYTES constant (now computed dynamically)
```

### Changes to `suffixAutomatonKernels.cu`

1. Update kernel to compute slot offset dynamically:

```cpp
__global__ void suffixAutomatonExtendKernel(
    int batchSize, int draftLength, int maxSlots, int maxSeqLen,
    void* slotsMemory, int const* batchIndices, 
    int* matchLenOut, int* draftTokensOut,
    int const* acceptedTokensIn, int const* acceptedLensIn)
{
    if (threadIdx.x > 0) return;
    
    int i = blockIdx.x;
    if (i >= batchSize) return;
    
    int batchIndex = batchIndices[i];
    assert(batchIndex >= 0 && batchIndex < maxSlots);
    
    // Calculate slot pointer based on dynamic state size
    size_t stateSize = SuffixAutomaton::getRequiredMemorySize(maxSeqLen);
    uint8_t* slotMemory = static_cast<uint8_t*>(slotsMemory) + batchIndex * stateSize;
    
    // Automaton already initialized with correct pointers
    SuffixAutomaton* slot = reinterpret_cast<SuffixAutomaton*>(slotMemory);
    
    // ... rest unchanged
}
```

2. Implement helper functions:

```cpp
size_t getSuffixAutomatonStateSize(size_t maxSeqLen) {
    return SuffixAutomaton::getRequiredMemorySize(maxSeqLen);
}

void initAutomaton(void* memory, size_t maxSeqLen) {
    SuffixAutomaton* sa = reinterpret_cast<SuffixAutomaton*>(memory);
    sa->init(memory, maxSeqLen);
}
```

---

## Phase 5: Update Python Bindings and Interface

**Files**: `bindings.cpp`, `suffix_automaton.py`, `suffix_automaton.pyi`

**Objective**: Update bindings to accept `max_seq_len` parameter.

### Changes to `bindings.cpp`

1. Update `allocate_workspace` to accept `max_seq_len`:

```cpp
// BEFORE
m.def("allocate_workspace", [](int maxSlots) { ... });

// AFTER
m.def("allocate_workspace",
    [](int maxSlots, int maxSeqLen) {
        size_t stateSize = sa::getSuffixAutomatonStateSize(maxSeqLen);
        size_t totalSize = static_cast<size_t>(maxSlots) * stateSize;
        
        auto options = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
        return at::zeros({static_cast<int64_t>(totalSize)}, options);
    },
    nb::arg("max_slots"), nb::arg("max_seq_len"),
    "Allocate GPU workspace for suffix automaton states");
```

2. Update state size query:

```cpp
// BEFORE: exposed static constant
m.attr("STATE_SIZE_BYTES") = sizeof(SuffixAutomaton);

// AFTER: function that computes size
m.def("get_state_size",
    [](int maxSeqLen) {
        return sa::getSuffixAutomatonStateSize(maxSeqLen);
    },
    nb::arg("max_seq_len"),
    "Get state size in bytes for given max sequence length");
```

3. Update extend functions to accept `max_seq_len`:

```cpp
m.def("invoke_extend",
    [](int batchSize, int draftLength, int maxSlots, int maxSeqLen,
       at::Tensor slots, at::Tensor batchIndices, ...) {
        sa::SuffixAutomatonExtendParams params;
        params.batchSize = batchSize;
        params.draftLength = draftLength;
        params.maxSlots = maxSlots;
        params.maxSeqLen = maxSeqLen;  // NEW
        // ...
    },
    nb::arg("batch_size"), nb::arg("draft_length"), 
    nb::arg("max_slots"), nb::arg("max_seq_len"),  // NEW
    ...);
```

4. Remove static constants:

```cpp
// REMOVE these:
// m.attr("MAX_SEQUENCE_LENGTH") = SAConfig::MAX_SEQUENCE_LENGTH;
// m.attr("STATE_SIZE_BYTES") = sizeof(SuffixAutomaton);
```

5. Update `build_automaton_host` to accept `max_seq_len`:

```cpp
m.def("build_automaton_host",
    [](std::vector<int> const& tokens, int maxSeqLen) {
        size_t stateSize = sa::getSuffixAutomatonStateSize(maxSeqLen);
        
        auto options = at::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true);
        at::Tensor hostState = at::zeros({static_cast<int64_t>(stateSize)}, options);
        
        sa::SuffixAutomaton* sa_ptr = reinterpret_cast<sa::SuffixAutomaton*>(hostState.data_ptr());
        sa::initAutomaton(sa_ptr, maxSeqLen);  // Now takes maxSeqLen
        sa::buildAutomatonFromTokens(sa_ptr, tokens.data(), static_cast<int>(tokens.size()));
        
        return hostState;
    },
    nb::arg("tokens"), nb::arg("max_seq_len"),
    "Build a suffix automaton on host from context tokens. Returns pinned CPU tensor.");
```

6. Update `copy_state_to_slot` to handle pointer relocation:

```cpp
m.def("copy_state_to_slot",
    [](at::Tensor hostState, at::Tensor gpuSlots, int slotIndex, int maxSeqLen) {
        size_t stateSize = sa::getSuffixAutomatonStateSize(maxSeqLen);
        size_t offset = static_cast<size_t>(slotIndex) * stateSize;
        
        // Calculate GPU destination address
        void* gpuDst = static_cast<uint8_t*>(gpuSlots.data_ptr()) + offset;
        void* hostSrc = hostState.data_ptr();
        
        // Relocate pointers from host to GPU addresses
        sa::SuffixAutomaton* sa_ptr = reinterpret_cast<sa::SuffixAutomaton*>(hostSrc);
        sa_ptr->relocate(hostSrc, gpuDst);
        
        // Copy to GPU
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        cudaMemcpyAsync(gpuDst, hostSrc, stateSize, cudaMemcpyHostToDevice, stream);
    },
    nb::arg("host_state"), nb::arg("gpu_slots"), nb::arg("slot_index"), nb::arg("max_seq_len"),
    "Copy a host-built suffix automaton state to a GPU slot (async)");
```

### Changes to `suffix_automaton.py`

1. Update `SuffixAutomatonManager.__init__`:

```python
def __init__(self, config: SAConfig, max_num_requests: int):
    self.config = config
    self.max_num_requests = max_num_requests
    self.max_seq_len = config.max_seq_len  # Use from config
    
    # Calculate per-state size based on max_seq_len
    self.state_size = _sa_native.get_state_size(self.max_seq_len)
    
    # Log memory usage
    total_memory_mb = max_num_requests * self.state_size / 1024 / 1024
    logger.info(
        f"Allocating {max_num_requests} SA slots with max_seq_len={self.max_seq_len} "
        f"({self.state_size / 1024 / 1024:.1f} MB/slot, {total_memory_mb:.1f} MB total)"
    )
```

2. Update `_ensure_workspace`:

```python
def _ensure_workspace(self, max_draft_len: int):
    if not self._workspace_allocated:
        self._gpu_slots = _sa_native.allocate_workspace(
            self.max_num_requests, 
            self.max_seq_len  # Pass max_seq_len
        )
        # ...
```

3. Update extend calls:

```python
def extend(self, ...):
    _sa_native.invoke_extend(
        batch_size,
        max_draft_len,
        self.max_num_requests,
        self.max_seq_len,  # NEW parameter
        self._gpu_slots,
        ...
    )
```

### Changes to `suffix_automaton.pyi`

```python
# REMOVE:
# MAX_SEQUENCE_LENGTH: int
# STATE_SIZE_BYTES: int

# UPDATE signatures:
def allocate_workspace(max_slots: int, max_seq_len: int) -> torch.Tensor:
    """Allocate GPU workspace for suffix automaton states"""

def get_state_size(max_seq_len: int) -> int:
    """Get state size in bytes for given max sequence length"""

def invoke_extend(
    batch_size: int, draft_length: int, max_slots: int, max_seq_len: int,
    slots: torch.Tensor, batch_indices: torch.Tensor, ...
) -> None:
    """Invoke suffix automaton extend CUDA kernel"""
```

---

## Memory Savings Comparison

| max_seq_len | State Size | 256 slots | 2048 slots |
|-------------|------------|-----------|------------|
| 256K (current) | ~61 MB | 15.6 GB | 125 GB |
| 128K | ~30 MB | 7.8 GB | 62 GB |
| 32K | ~7.5 MB | 1.9 GB | 15 GB |
| 8K | ~1.9 MB | 0.5 GB | 3.8 GB |

---

## Implementation Order

1. **Phase 1** (saBuffer.h) - Convert buffer types, no dependencies
2. **Phase 2** (saFlatGraph.h, saFlatHashMap.h) - Convert graph types, depends on Phase 1
3. **Phase 3** (suffixAutomaton.h) - Convert main struct, depends on Phase 2
4. **Phase 4** (kernels) - Update kernel params, depends on Phase 3
5. **Phase 5** (bindings + Python) - Update API, depends on Phase 4

### Notes

- Each phase modifies existing code **in place** (no parallel implementations)
- Tests will break temporarily during each phase until the full stack is updated
- Consider doing Phases 1-3 together as an atomic change, then 4-5 together
- Remove `saConfig.h` after Phase 5 (no longer needed)

---

## Migration Notes

Since we're replacing the static implementation entirely:

1. **No backward compatibility layer** — cleaner code, single code path
2. **All callers must provide `max_seq_len`** — no default fallback
3. **Remove `saConfig.h` constants** — `SA_MAX_SEQUENCE_LENGTH` and `SA_MAX_SLOTS` no longer needed
4. **Binary size reduction** — no more 256K-sized embedded arrays in the binary

### Breaking Changes

| Before | After |
|--------|-------|
| `allocate_workspace(max_slots)` | `allocate_workspace(max_slots, max_seq_len)` |
| `STATE_SIZE_BYTES` constant | `get_state_size(max_seq_len)` function |
| `MAX_SEQUENCE_LENGTH` constant | Removed (passed at runtime) |
| `sizeof(SuffixAutomaton)` fixed | Computed from `max_seq_len` |

---

## Testing Strategy

1. **Unit tests**: Test buffer/graph types with various capacities
2. **Integration tests**: Test SA with various `max_seq_len` values (8K, 32K, 128K, 256K)
3. **Memory tests**: Verify `get_state_size()` matches actual allocation
4. **Boundary tests**: Test behavior at capacity limits (sequence reaches max_seq_len)
5. **E2E tests**: Run `ngram_perf.py` with different `max_seq_len` configs

### Test Cases to Add

```python
@pytest.mark.parametrize("max_seq_len", [8192, 32768, 131072, 262144])
def test_dynamic_max_seq_len(max_seq_len):
    config = SAConfig(max_seq_len=max_seq_len)
    manager = SuffixAutomatonManager(config, max_num_requests=16)
    
    # Verify state size scales correctly
    expected_size = _sa_native.get_state_size(max_seq_len)
    assert manager.state_size == expected_size
    
    # Test basic operations work
    manager.add_request(0, list(range(100)))
    # ...
```
