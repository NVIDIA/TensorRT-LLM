(kv-cache-management)=

# KV Cache Management: Pools, Blocks, and Events

This document provides an overview of the internal hierarchy and event system for paged KV cache management, as implemented in the TensorRT-LLM codebase.

For more information on KV cache reuse see [KV cache reuse](kv-cache-reuse.md).

---

## Hierarchy: Pool, Block, and Page

### **Block**
- **Definition:** The smallest unit of KV cache allocation. A `KVCacheBlock` holds metadata (not the actual data) for a chunk of KV cache.
- **Purpose:** Each block represents a fixed number of tokens' worth of KV data (can be specified by `tokens_per_block` parameter).
- **Usage:** Blocks are allocated, reused, or evicted as sequences are processed.

### **Page**
- **Definition:** In this codebase, "page" is often used interchangeably with "block" (as in "paged KV cache"), but technically, a page could refer to a memory page (hardware-level), while a block is a logical unit for the cache.
- **In Practice:** The code uses "block" as the main unit; "page" is not a distinct class or struct.

### **Pool**
- **Definition:** A pool is a contiguous memory buffer (or set of buffers) that holds the actual KV data for one or more layers.
- **Types:** There are primary pools (fast GPU memory) and secondary pools (slower, e.g., CPU or offload memory).
- **Organization:** Each pool can serve multiple layers that share the same KV head configuration. Pools are managed by `KVCacheBlockPool` and tracked in vectors in `WindowBlockManager`.
- **Block ↔ Pool:** Each block is an index into a pool; the pool provides the actual storage, while the block is the metadata handle.

### **WindowBlockManager/BlockManager**

TRT-LLM supports 2 complex features related to KV cache management:
1. **Variable Group-Query Attention (VGQA)** - i.e. a different `num_kv_heads` value for different layers.
2. **Variable Sliding Window Attention (VSWA)** - i.e. a different `attention_window_size` value for different layers.

In order to support both of these features, the pool management works as described below.

But in the simple, *most common case*, for most models, where
1. [MHA/MQA/Non-variable GQA](gpt-attention.md#multi-head-multi-query-and-group-query-attention), i.e., same `num_kv_heads` value for all layers,
2. Global attention/[SWA](gpt-attention.md#sliding-window-attention-cyclic-rolling-buffer-kv-cache), i.e., same `attention_window_size` value for all layers,

only a *single* pool will be created within the structure described below.

#### KV Cache Pool Management

- **WindowBlockManager:** Manages blocks and pools for a specific attention window size. Within a `WindowBlockManager`, there can be multiple pools - each corresponding a unique number of KV heads - i.e., to support VGQA.
- **BlockManager:** Manages all `WindowBlockManager` instances, one per unique window size.

**Hierarchy Summary:**
- **Pool** (memory buffer for KV data)
  - Contains many blocks.
- **Blocks** (metadata for a chunk of the pool, each block = `tokens_per_block` tokens)
    - (Optionally, blocks can be swapped between primary/secondary pools.)
- **BlockManager/WindowBlockManager**: Manage pools and blocks, handle allocation, reuse, and eviction.

---

## Events in `KVCacheEventManager`

The `KVCacheEventManager` is responsible for tracking and reporting significant changes in the state of the KV cache. Events are used for logging, debugging, or possibly for external monitoring.

### **Types of Events**
- **Created Event:** When pools or blocks are created/allocated.
- **Updated Event:** When a block's state changes (e.g., moved between primary/secondary, priority updated).
- **Removed Event:** When a block is removed from the cache (evicted or released).
- **Stored Event:** When blocks are stored for potential reuse (e.g., after a sequence finishes and its blocks are reusable).

### **What Triggers an Event?**
- **Allocation/Deallocation:** Creating or freeing memory pools or blocks.
- **Eviction/Reuse:** When a block is evicted, reused, or its priority changes.
- **Block Movement:** When a block is moved between memory levels (primary ↔ secondary).
- **Block Storage:** When blocks are stored for future reuse (e.g., after a sequence completes).

**In summary:**
An "event" is any significant change in the lifecycle or state of a KV cache block or pool, tracked for monitoring, debugging, or optimization purposes.

---
