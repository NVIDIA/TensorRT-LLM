Sanity Perf Check Introduction

# Background
The sanity perf check mechanism is the way of perf regression detection for L0 testing. We create the base_perf.csv which consists of the several models' perf baseline and use the sanity_perf_check.py to detect the perf regression.
# Usage
There're four typical scenarios for sanity perf check feature.

1. The newly added MR doesn't impact the models' perf, the perf check will pass w/o exception.
2. The newly added MR introduces the new model into perf model list. The sanity check will trigger the exception and the author of this MR needs to add the perf into base_perf.csv.
3. The newly added MR improves the existed models' perf and the MR author need to refresh the base_perf.csv data w/ new baseline.
4. The newly added MR introduces the perf regression and the MR author needs to fix the issue and rerun the pipeline.
