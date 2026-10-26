# Locality Domains

A locality domain is one half of a Rubin GPU: a set of SMs and the HBM attached to them. With locality
domains enabled, each request's KV cache lives in one domain and its attention runs on that domain's SMs,
so KV cache reads do not cross the die-to-die link. This is a prototype and is off by default.

Consider enabling it for dense attention variants such as GQA and MLA at high batch sizes or high
concurrency, where decode attention is bandwidth-bound. Leave it off for low concurrency or small decode
batches, where splitting each step across the two domains costs more than it saves. Sparse attention models
are not supported.

```yaml
enable_locality_domains: true
kv_cache_config:
  use_kv_cache_manager_v2: true
```

Prefix reuse happens only within a domain.
