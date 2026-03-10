# No Universal Templates

The earlier architecture-class templates were removed because current checked-in configs vary materially by model family, GPU, and serving mode, and some earlier family assignments were wrong.

Use a nearby checked-in config instead:

1. Exact in-scope database match in `examples/configs/database/lookup.yaml`
2. Same-model in-scope database config for a nearby scenario
3. Same-model curated config from `examples/configs/curated/lookup.yaml` that stays in this skill's scope
4. Model-specific deployment guide / README

If none of those exist, stop and label any draft config as unverified.

This file intentionally does not provide synthetic YAML templates.
