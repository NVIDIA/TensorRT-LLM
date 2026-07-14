# Optimization Patterns — Index

Reference catalog of optimization patterns by hotspot type. Each file covers a thematic group of patterns. Read only the file relevant to your classified hotspot type.

| Hotspot Types | File | Description |
|---------------|------|-------------|
| SYNC, ALLOC | [patterns/sync-alloc.md](patterns/sync-alloc.md) | GPU synchronization removal and buffer pre-allocation |
| REDUNDANT_ITER, DEAD_WORK, PYLOOP, HOIST | [patterns/loop-iteration.md](patterns/loop-iteration.md) | Loop merging, dead work elimination, vectorization, invariant hoisting |
| FUNCALL, CONTAINER, BOUNDARY, ENUM_CONSTRUCT | [patterns/python-overhead.md](patterns/python-overhead.md) | Method caching, data structure access, Python-C++ crossings, enum deferral |
| CUSTOM_OP, GRAPH_SPLIT, GRAPH_EXPAND | [patterns/gpu-graph.md](patterns/gpu-graph.md) | C++ custom ops, CUDA graph granularity, graph capture expansion |
| GC_MGMT, GIL_CONTENTION, SERIALIZE | [patterns/system.md](patterns/system.md) | Garbage collection, threading overhead, serialization optimization |
| Compound, CPython Pitfalls | [patterns/compound-pitfalls.md](patterns/compound-pitfalls.md) | Multi-type hotspots and CPython micro-optimization traps |
