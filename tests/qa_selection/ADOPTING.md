# Selecting tests by what they need, in any pytest project

Multi-GPU suites often keep static lists per machine and GPU count, edited by hand whenever a test
is added or grows:

```yaml
# h100_4gpu.yml, b200_4gpu.yml, b200_8gpu.yml, ...
- test_model.py::test_tp4
- test_model.py::test_tp4_fp8
```

With the plugin, the lists are a query instead.

**The test author states what a test needs**, next to the test:

```python
@pytest.mark.skip_less_device(4)              # 4 GPUs; no mark is 1
@pytest.mark.skip_less_device_memory(80000)   # 80000 MiB on each
@skip_pre_hopper                              # sm 90 or newer
def test_tp4_fp8(): ...
```

| states | with |
|---|---|
| GPU count | `skip_less_device(N)`, `skip_less_mpi_world_size(N)` |
| GPU memory | `skip_less_device_memory(MiB)` |
| GPU or CPU architecture, device | a `skipif` whose reason [`core/rules.json`](core/rules.json) lists, such as `skip_pre_hopper` or `skip_arm`; `skip_device_not_contain(name)` |

**The test infra engineer asks which tests a machine runs**, split by allocation size. The machine
is a [profile](README.md#profiles): measured on the machine, or saved as `out/local.json` to ask
from anywhere.

```bash
PYTHONPATH=<TensorRT-LLM>/tests pytest --collect-only -q -p qa_selection.plugin \
    --ladder=1,4,8 --selection-out-dir=out/ tests/
```

```text
target        local, ladder 1,4,8
candidates    120
selected      104
  1gpu        61
  4gpu        30
  8gpu        13
deselected    16
  9  This test is not supported in pre-Hopper architecture
  7  skip_less_device(16): needs 16 GPUs, largest rung is 8
written       out/: local.json local-1gpu.ids local-4gpu.ids local-8gpu.ids
```

Each list runs on an allocation of its size, `pytest $(cat out/local-4gpu.ids)` on 4 GPUs. A new
test, or a changed mark, moves at the next collection; nobody edits a list.

The plugin declares these marks when the suite does not. It only selects: at run time, the marks
do what the suite's own conftest makes them do. A `skipif` reason that `core/rules.json` does not
list keeps its test. [The tests](README.md#tests) are what the plugin promises.
