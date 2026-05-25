# Python Overhead Patterns

---

## FUNCALL: Cache Invariant Method Results

**Pattern 1: Hoist invariant method calls out of loops**
```python
# BEFORE: Method call inside hot loop (result never changes)
for request in generation_requests:  # 279K iterations
    if self.mapping.has_cp_helix():  # Same result every time!
        handle_helix(request)

# AFTER: Cache before loop
_has_cp_helix = self.mapping.has_cp_helix()
for request in generation_requests:
    if _has_cp_helix:
        handle_helix(request)
```

**Pattern 2: Eliminate redundant assignments inside loops**
```python
# BEFORE: Assign constant inside loop
for request in requests:
    for beam in range(beam_width):
        first_beam = 0                   # Assigned every iteration!
        if beam == first_beam:
            process_first_beam(request)

# AFTER: Compare directly with literal
for request in requests:
    for beam in range(beam_width):
        if beam == 0:                    # No assignment needed
            process_first_beam(request)
```

**Pattern 3: Inline hot functions**
```python
# BEFORE: function call in tight loop
def is_valid(req):
    return req.status == Status.ACTIVE and req.tokens > 0

for req in requests:
    if is_valid(req):             # Function call overhead
        process(req)

# AFTER: inline the logic
for req in requests:
    if req.status == Status.ACTIVE and req.tokens > 0:
        process(req)
```

---

## CONTAINER: Optimize Data Structure Access

**Pattern 1: Cache dictionary lookups**
```python
# BEFORE: repeated dict access
for req in requests:
    strategy = self.strategies[req.strategy_type]  # Hash + lookup each time
    strategy.process(req)

# AFTER: group by key first
from collections import defaultdict
by_strategy = defaultdict(list)
for req in requests:
    by_strategy[req.strategy_type].append(req)
for strategy_type, reqs in by_strategy.items():
    strategy = self.strategies[strategy_type]      # One lookup per group
    for req in reqs:
        strategy.process(req)
```

**Pattern 2: Use local variable for repeated attribute access**
```python
# BEFORE: repeated attribute lookup
for req in requests:
    self.config.param1             # __getattribute__ each time
    self.config.param2

# AFTER: cache in local
config = self.config               # One lookup
param1, param2 = config.param1, config.param2
for req in requests:
    use(param1, param2)
```

**Pattern 3: Use slots for frequently instantiated classes**
```python
# BEFORE: regular class (dict-based attributes)
class Request:
    def __init__(self, id, data):
        self.id = id
        self.data = data

# AFTER: slots (faster attribute access, less memory)
class Request:
    __slots__ = ['id', 'data']
    def __init__(self, id, data):
        self.id = id
        self.data = data
```

---

## BOUNDARY: Reduce Python-C++ Crossings

> **Scoping note**: BOUNDARY optimizations (feature-gating, shadowing) only help when the boundary crossing is inside a **hot loop** (high Hits count in line_profiler). A single boundary crossing outside a loop (~250ns) is negligible. Focus on crossings where `Hits * Per Hit` exceeds ~100us total.

**Pattern 1: Feature-gate to skip entirely when feature unused**
```python
# BEFORE: Always access C++ property even when feature is disabled
for req in requests:
    multimodal_data = req.py_multimodal_data  # nanobind crossing ~250ns
    if multimodal_data:
        process_multimodal(multimodal_data)

# AFTER: Gate on model-level feature flag (checked once)
_has_multimodal = self.model_config.has_multimodal()  # One-time check
for req in requests:
    if _has_multimodal:
        multimodal_data = req.py_multimodal_data  # Only cross when needed
        if multimodal_data:
            process_multimodal(multimodal_data)
```

**Pattern 2: Shadow with Python attribute at construction time**
```python
# BEFORE: Access C++ property every iteration
for req in requests:
    req_id = req.py_request_id      # nanobind crossing ~250ns
    req_type = req.request_type     # nanobind crossing ~250ns
    process(req_id, req_type)

# AFTER: Shadow at request creation time (once per request lifetime)
class PyRequest:
    """Python wrapper that caches frequently-accessed C++ properties."""
    __slots__ = ['_cpp_req', 'request_id', 'request_type']

    def __init__(self, cpp_req):
        self._cpp_req = cpp_req
        self.request_id = cpp_req.py_request_id   # One crossing at init
        self.request_type = cpp_req.request_type   # One crossing at init

# In hot loop: pure Python attribute access (~0.05us)
for req in py_requests:
    process(req.request_id, req.request_type)
```

---

## ENUM_CONSTRUCT: Defer Expensive Enum Construction

**Pattern 1: Raw int comparison, construct only on rare path**
```python
# BEFORE: Construct Enum every iteration (~1.7us each)
for req in requests:
    state = RequestState(req.raw_state)       # ~1.7us per call
    if state == RequestState.GENERATION:
        handle_generation(req)
    elif state == RequestState.CONTEXT:
        handle_context(req)

# AFTER: Compare raw ints for dominant case, construct only on rare path
_GENERATION_INT = RequestState.GENERATION.value
_CONTEXT_INT = RequestState.CONTEXT.value
for req in requests:
    raw = req.raw_state
    if raw == _GENERATION_INT:                # ~0.05us int compare
        handle_generation(req)
    elif raw == _CONTEXT_INT:                 # ~0.05us int compare
        handle_context(req)
    else:
        state = RequestState(raw)             # Rare: construct for unknown
        handle_other(req, state)
```

**Pattern 2: Batch-collect raw ints, then batch-convert non-default values**
```python
# BEFORE: Construct enum for every request to categorize
categories = {}
for req in requests:
    mode = RequestMode(req.raw_mode)          # ~1.7us * N
    categories.setdefault(mode, []).append(req)

# AFTER: Categorize by raw int, only construct enums for rare categories
_DEFAULT_MODE_INT = RequestMode.GENERATE.value
default_reqs = []
other_by_int = {}
for req in requests:
    raw = req.raw_mode
    if raw == _DEFAULT_MODE_INT:
        default_reqs.append(req)              # No enum construction
    else:
        other_by_int.setdefault(raw, []).append(req)

# Construct enums only for the non-default categories (typically 0-2 entries)
categories = {RequestMode.GENERATE: default_reqs}
for raw_int, reqs in other_by_int.items():
    categories[RequestMode(raw_int)] = reqs   # Rare: few constructions
```
