# Wan2.2 unified inference optimization bundle

## A/B switches (All optimizations are disabled by default)

Baseline:

```bash
unset VISUAL_GEN_WAN_OPTIMIZATIONS
```


```bash
export VISUAL_GEN_WAN_OPTIMIZATIONS=1
export VISUAL_GEN_WAN_QK_ROPE_THREADS=512
```

Restart the worker process after changing environment variables because flags
are read when `wan_transformer.py` is imported.
