# Launch Scripts for CuTe DSL Kernels

## MoE Workload Generator

```bash
# Generate workload using a balanced random method
# Per-rank token number 128, EP size 32 (a typical workload for large EP gen phase)
python moe_workload_generator.py --num_tokens 128 --ep_size 32 --tile_size 128
# Per-rank token number 8192, EP size 4 (a typical workload for ctx phase)
python moe_workload_generator.py --num_tokens 8192 --ep_size 4 --tile_size 256
```
