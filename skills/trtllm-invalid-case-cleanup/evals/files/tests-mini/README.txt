Trimmed copy of parts of a TensorRT-LLM source tree prepared for this exercise.

Only these parts are present:
  tests/integration/test_lists/test-db/  - three trimmed test-db lists (H100, B200, DGX B200)
  tests/integration/test_lists/waives.txt
  tests/unittest/                        - two trimmed test files (stored as .py.txt)
  tensorrt_llm_src/                      - one trimmed source file (stored as .py.txt)
  jenkins/stages.txt                     - the stage map excerpt from jenkins/L0_Test.groovy
  changes.diff                           - the change under review for the skip audit

Python files carry a .py.txt suffix so they are read as text. Treat each as the
checked-in file without the suffix. Everything else (the rest of the package,
other lists, conftest.py, scripts) is absent. Marker semantics follow
tests/integration/defs/conftest.py: skip_pre_blackwell skips on GPUs older than
SM100, skip_less_device(N) skips when fewer than N GPUs are visible.
Some entries were shortened or adjusted for the exercise; treat the files that
are present as the checked-in sources for this task.
