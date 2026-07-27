# Compound Patterns & CPython Pitfalls

---

## Compound Patterns

Some hotspots combine multiple types. Address them in the listed order (outer -> inner).

### FUNCALL + BOUNDARY
A Python function call wraps a C++ boundary crossing. Eliminate the function call first (inline), then gate or shadow the boundary crossing.
```python
# BEFORE: function call + boundary crossing per iteration
def get_request_id(req):           # FUNCALL: ~100ns call overhead
    return req.py_request_id       # BOUNDARY: ~250ns nanobind crossing

for req in requests:
    rid = get_request_id(req)      # ~350ns total per call

# AFTER: inline + shadow
for req in requests:
    rid = req.request_id           # Shadowed Python attr: ~50ns
```

### DEAD_WORK + REDUNDANT_ITER
Unnecessary object construction combined with a redundant pass. Eliminate the dead work first (guard with cheap check), then merge the surviving iteration into an existing loop.
```python
# BEFORE: construct always + iterate twice
for req in requests:
    params = Params(req.data)      # DEAD_WORK: always constructed
    if params.has_content():
        use(params)
ids = [r.id for r in requests]     # REDUNDANT_ITER: second pass

# AFTER: guard + merge
ids = []
for req in requests:
    ids.append(req.id)             # Merged into single pass
    if req.data:                   # Guard: skip when empty
        params = Params(req.data)
        if params.has_content():
            use(params)
```

### PYLOOP + BOUNDARY
A Python loop body whose dominant cost is boundary crossings (C++ property access per iteration). Gate or shadow the boundary crossings first, then consider vectorizing the remaining loop body.
```python
# BEFORE: loop with boundary crossings
for req in requests:                          # PYLOOP: N iterations
    rid = req.py_request_id                   # BOUNDARY: ~250ns
    rtype = req.request_type                  # BOUNDARY: ~250ns
    process(rid, rtype)                       # ~500ns/iter from crossings alone

# AFTER: shadow at creation + pure Python loop
# (At request creation time, cache properties on a Python wrapper)
for req in py_requests:
    process(req.request_id, req.request_type) # ~50ns/iter (Python attrs)
```

### ENUM_CONSTRUCT + PYLOOP
Expensive enum construction inside a Python loop. Replace enum construction with raw int comparison first, then consider vectorizing the remaining loop body if possible.
```python
# BEFORE: enum + loop
for req in requests:                         # PYLOOP
    mode = RequestMode(req.raw_mode)         # ENUM_CONSTRUCT: ~1.7us
    if mode == RequestMode.GENERATE:
        handle(req)

# AFTER: raw int compare (eliminates enum cost within loop)
_GEN = RequestMode.GENERATE.value
for req in requests:
    if req.raw_mode == _GEN:                 # ~0.05us
        handle(req)
```

---

## CPython Pitfalls

Micro-optimizations in CPython can have counterintuitive results. **Always validate with re-profiling.**

### Pitfall 1: Ternary expressions can be slower than attribute access

```python
# SEEMS FASTER but is actually SLOWER:
beam_width = request.sampling_config.beam_width if _use_beam_search else 1
# CPython ternary bytecode: LOAD, POP_JUMP, LOAD_CONST, JUMP — ~14us/hit

# ACTUALLY FASTER (simple attribute chain):
beam_width = request.sampling_config.beam_width
# CPython attribute chain: LOAD_ATTR, LOAD_ATTR — ~5us/hit
```

**Why:** The conditional expression generates branch bytecode that CPython evaluates less efficiently than a straightforward attribute access chain, even when the condition is a local variable. The branch prediction overhead in the interpreter loop outweighs the saved attribute access.

### Pitfall 2: Local variable caching only helps for method calls, not simple attributes

```python
# HELPS (~1.5s savings for 279K iterations):
_has_cp_helix = self.mapping.has_cp_helix()  # Method call -> cache result
for req in requests:
    if _has_cp_helix: ...

# MARGINAL (< 0.1s savings):
_is_dummy = self.is_draft_model  # Simple attribute -> already fast
for req in requests:
    if _is_dummy: ...
```

**Why:** `self.attr` is a single `LOAD_ATTR` instruction (~5us). Caching it as a local saves ~1-2us per access. Only worthwhile for **method calls** (which have call frame overhead) or **chained attributes** (`self.a.b.c`).

### Pitfall 3: Adding conditionals to "skip" loop iterations can cost more than the iterations

```python
# SLOWER: Added check costs more than the skipped work
for req in requests:
    if req.needs_processing:  # Extra check on every iteration
        do_work(req)          # Work is cheap

# FASTER: Just do the work unconditionally
for req in requests:
    do_work(req)
```

**Rule of thumb:** Only add a guard condition if the skipped work costs >10us per iteration AND the condition itself is <2us (simple attribute/truthiness check).

### Pitfall 4: `isinstance()` in hot loops (~200-300ns MRO traversal)

```python
# SLOWER: isinstance() walks the MRO chain each call
for req in requests:
    if isinstance(req, GenerationRequest):  # ~200-300ns MRO traversal
        handle_generation(req)

# FASTER: use a type tag or duck-typing attribute
for req in requests:
    if req.is_generation:                   # ~50ns attribute check
        handle_generation(req)
```

**Why:** `isinstance()` must traverse the Method Resolution Order (MRO) and check each base class. For deep inheritance hierarchies or ABCs, this can be 200-300ns per call. In a hot loop with 1000+ iterations, this adds up. Prefer a boolean attribute or type tag set at construction time.

### Pitfall 5: `in` on list vs set (O(n) vs O(1))

```python
# SLOWER: O(n) membership test on list
blocked_ids = [1, 2, 3, ..., 100]          # list
for req in requests:
    if req.id in blocked_ids:               # O(n) linear scan

# FASTER: O(1) membership test on set
blocked_ids = {1, 2, 3, ..., 100}          # set
for req in requests:
    if req.id in blocked_ids:               # O(1) hash lookup
```

**Why:** `in` on a `list` is O(n) — it performs a linear scan. For lists with >10 elements in a hot loop, convert to a `set` or `frozenset` before the loop. The set construction cost is amortized after ~2-3 loop iterations for a 100-element collection.

### Pitfall 6: Generator expression vs list comprehension in `sum`/`min`/`max`

```python
# SLOWER: list comprehension materializes the full list
total = sum([req.token_count for req in requests])  # Allocates list

# FASTER: generator expression (no intermediate allocation)
total = sum(req.token_count for req in requests)     # Lazy evaluation
```

**Why:** A list comprehension `[expr for x in iter]` allocates a full list in memory before passing it to `sum()`. A generator expression `(expr for x in iter)` yields values lazily with no intermediate allocation. For large collections, this saves both memory and time. The difference is most pronounced with >1000 elements.

### Pitfall 7: String formatting in hot paths (use tuple keys instead)

```python
# SLOWER: f-string formatting every iteration
cache = {}
for req in requests:
    key = f"{req.model_id}_{req.batch_size}_{req.dtype}"  # ~500ns string alloc
    if key not in cache:
        cache[key] = compute(req)

# FASTER: tuple key (no string allocation, faster hashing)
for req in requests:
    key = (req.model_id, req.batch_size, req.dtype)       # ~50ns tuple creation
    if key not in cache:
        cache[key] = compute(req)
```

**Why:** f-string formatting allocates a new string object each time, involving concatenation and `__format__` calls. Tuples of ints/strings are cheaper to create and hash faster than equivalent string keys. Use tuple keys for dictionary lookups in hot loops.
