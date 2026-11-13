# Locality Domains

A locality domain is one half of a Rubin GPU: a set of SMs and the HBM attached to them. With locality
domains enabled, each request is placed on one domain: its new KV cache pages are allocated there and its
attention runs on that domain's SMs, so most KV cache reads stay off the die-to-die link. This is a prototype
and is off by default.

Consider enabling it for dense attention variants such as GQA and MLA at high batch sizes or high
concurrency, where decode attention is bandwidth-bound. Leave it off for low concurrency or small decode
batches, where splitting each step across the two domains costs more than it saves. Sparse attention models
are not supported.

```yaml
enable_locality_domains: true
kv_cache_config:
  use_kv_cache_manager_v2: true
```

Cached prefixes are shared by both domains: a request reuses a prefix wherever its pages are, and a new
request is placed on the domain holding most of its cached prefix. New pages spill to the other domain only
when the request's own domain is full of live requests; pages on the other domain are read over the
die-to-die link.
