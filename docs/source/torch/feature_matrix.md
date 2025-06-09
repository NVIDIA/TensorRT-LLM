# Feature Matrix


## Feature Combination

| Feature                    | Overlap Scheduler | CUDA Graph | Attention Data Parallelism | Disaggregated Serving | Chunked Prefill | MTP | EAGLE-3(One Model Engine) | EAGLE-3(Two Model Engine) | Torch Sampler | TLLM C++ Sampler |
| -------------------------- | ----------------- | ---------- | -------------------------- | --------------------- | --------------- | --- | ------------------------- | ------------------------- | ------------- | ---------------- |
| Overlap Scheduler          | Yes               |            |                            |                       |                 |     |                           |                           |               |                  |
| CUDA Graph                 | Yes               | Yes        |                            |                       |                 |     |                           |                           |               |                  |
| Attention Data Parallelism | Yes               | Yes        | Yes                        |                       |                 |     |                           |                           |               |                  |
| Disaggregated Serving      | Yes               | Yes        | Yes                        | Yes                   |                 |     |                           |                           |               |                  |
| Chunked Prefill            | WIP               | Yes        | Yes                        | Yes                   | Yes             |     |                           |                           |               |                  |
| MTP                        | Yes               | Yes        | Yes                        | Yes                   | Yes             | Yes |                           |                           |               |                  |
| EAGLE-3(One Model Engine)  | Yes               | Yes        | Yes                        | Yes                   | Yes             | No  | Yes                       |                           |               |                  |
| EAGLE-3(Two Model Engine)  | WIP               | Yes        | Yes                        | No                    | Yes             | No  | Yes                       | Yes                       |               |                  |
| Torch Sampler              | Yes               | Yes        | Yes                        | Yes                   | Yes             | Yes | Yes                       | Yes                       | Yes           |                  |
| TLLM C++ Sampler           | Yes               | Yes        | Yes                        | No                    | Yes             | No  | No                        | No                        | No            | Yes              |
