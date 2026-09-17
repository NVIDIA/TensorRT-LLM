---
receipts:
  sm_103: {status: passed, tests: 9}
---

# flashinfer_rmsnorm

The contract is the wrapper: `flashinfer_rmsnorm.py` carries what the op
computes (`reference`), what it refuses (`is_valid`), the range it is
certified over (`CELLS`, each with the reason it exists), and what is left
over (`note`). All but the last is executed by
`tests/unittest/_torch/modeling_v2/norm/test_modeling_v2_flashinfer_rmsnorm.py`
on every commit, which is the property this file never had.

Only the receipt stays here, because that is what `catalog/index.yaml`
indexes and what the freshness rule reads.
