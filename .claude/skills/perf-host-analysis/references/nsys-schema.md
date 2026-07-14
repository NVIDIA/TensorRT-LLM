# nsys SQLite Schema Reference

## Key Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `CUPTI_ACTIVITY_KIND_KERNEL` | GPU kernel executions | `start`, `end`, `shortName` (-> StringIds) |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | CUDA API calls | `start`, `end`, `nameId` (-> StringIds) |
| `NVTX_EVENTS` | NVTX ranged events | `start`, `end`, `textId` (-> StringIds), `text` |
| `StringIds` | String lookup table | `id`, `value` |

## Important: String ID Joins

`shortName`, `nameId`, and `textId` are integer foreign keys into `StringIds`. **Always join** to get the actual string value:

```sql
-- Kernel names
SELECT s.value, k.start, k.end
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id;

-- NVTX event text
SELECT s.value, n.start, n.end
FROM NVTX_EVENTS n
JOIN StringIds s ON n.textId = s.id;
```

## Useful Queries

### Analysis Window (Total Time)

```sql
SELECT MIN(start) AS window_start, MAX(end) AS window_end,
       (MAX(end) - MIN(start)) / 1000.0 AS total_time_us
FROM CUPTI_ACTIVITY_KIND_KERNEL;
```

### M1: GPU Idle Ratio / M4: GPU Utilization

```sql
-- Approximate GPU active time (accurate when kernel overlap is minimal)
SELECT SUM(end - start) / 1000.0 AS approx_gpu_active_us
FROM CUPTI_ACTIVITY_KIND_KERNEL;
-- gpu_idle_us = total_time_us - gpu_active_us
-- gpu_idle_ratio = gpu_idle_us / total_time_us  (threshold >0.30)
-- gpu_utilization = 1 - gpu_idle_ratio           (threshold <0.60)
```

For precise GPU active time (merging overlapping ranges), export kernel `(start, end)` pairs and merge in Python -- see [metrics.md](metrics.md) for the full approach.

### M2: Launch Overhead Ratio

```sql
SELECT SUM(r.end - r.start) / 1000.0 AS cudaLaunchKernel_us
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value = 'cudaLaunchKernel';
-- launch_overhead_ratio = cudaLaunchKernel_us / total_time_us  (threshold >0.10)
```

### M5: NCCL Ratio

```sql
SELECT SUM(k.end - k.start) / 1000.0 AS nccl_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE s.value LIKE '%nccl%';
-- nccl_ratio = nccl_us / gpu_active_us  (threshold >0.20)
```

### Find All NVTX Range Names

```sql
SELECT DISTINCT s.value, COUNT(*)
FROM NVTX_EVENTS n
JOIN StringIds s ON n.textId = s.id
WHERE n.end > 0
GROUP BY s.value
ORDER BY COUNT(*) DESC;
```

### Allreduce Kernels Timeline

```sql
SELECT (k.start - (SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL))/1e9 AS t_sec,
       (k.end - k.start)/1000.0 AS dur_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE s.value LIKE '%allreduce%'
ORDER BY k.start;
```

### CUDA API Breakdown (Time Window)

```sql
SELECT s.value, COUNT(*), SUM(r.end - r.start)/1000.0 AS total_us
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE r.start >= ? AND r.start < ?
GROUP BY s.value ORDER BY total_us DESC;
```

### GPU Kernel Breakdown (Time Window)

```sql
SELECT s.value, COUNT(*), SUM(k.end - k.start)/1000.0 AS total_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE k.start >= ? AND k.start < ?
GROUP BY s.value ORDER BY total_us DESC;
```
