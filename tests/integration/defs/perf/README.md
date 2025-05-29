# Sanity Perf Check Introduction

## Background
"Sanity perf check" is a mechanism to detect performance regressions in the L0 pipeline.
The tests defined in `l0_perf.yml` are the ones that are required to pass for every PR before merge.

### `base_perf.csv`
The baseline for performance benchmarking is defined at `base_perf.csv` - this file contains the metrics that we verify regression on between CI runs.

This file contains records in the following format:
```
perf_case_name metric_type	perf_metric	threshold absolute_threshold
```

To allow for some machine dependent variance in performance benchmarking we also define a `threshold` and an `absolute_threshold`. This ensures we do not fail on results that reside within legitimate variance thresholds.

`threshold` is relative.

## CI
As part of our CI, the `test_perf.py` collects performance metrics for configurations defined in `l0_perf.yml`. This step outputs a `perf_script_test_results.csv` containing the metrics collected for all configurations.

After this step completes, the CI will run `sanity_perf_check.py`. This script will make sure that all differences in metrics from the run on this branch is within a designated threshold of the baseline (`base_perf.csv`).

There're 4 possible results for this:
1. The current HEAD impact on the performance for our setups is within accepted threshold - the perf check will **pass** w/o exception.
2. The current HEAD introduces a new setup/metric in `l0_perf.yml` or removes some of them. This will result in new metrics collected by `test_perf.py` which will **fail** `sanity_perf_check.py`. This requires an update for `base_perf.csv`.
3. The current HEAD improves performance for at least one metric by more than the accepted threshold, which will **fail** `sanity_perf_check.py`. This requires an update for `base_perf.csv`
4. The current HEAD introduces a regression to one of the metrics that is over the accepted threshold, which will **fail** `sanity_perf_check.py`. This will require to fix the current branch and rerun the pipeline.

### Updating `base_perf.csv`
If a CI run fails `sanity_perf_check.py`, it will upload a patch file as an artifact. This file can be applied to current branch using `git apply <patch_file>`.

This patch will only update the metrics that had a difference which was over the accepted threshold. The patch will also remove/add metrics according to the removed or added tests.

## Running locally
Given a `target_perf_csv_path` you can compare it to another perf csv file.
First make sure you install the dependencies:
```
pip install -r tests/integration/defs/perf/requirements.txt
```
Then, you can run it with:
```
sanity_perf_check.py <target_perf_csv_path> <base_perf_csv_path>
```
** In the CI, `<base_perf_csv_path>` is the `base_perf.csv` file path mentioned above.

Running this print the diffs between both performance results. It presents only:
1. Metrics that have a diff bigger than the accepted threshold.
2. Metrics missing in `base_perf_csv`.
3. Metrics missing in `target_perf_csv`.

If any diffs were found it will also generate a patch file to change `base_perf_csv`  with the new metrics, it will be written to the same directory as <target_perf_csv_path> resides in.


## Generating diff report
To view the difference between performance reports, it is possible to generate a pdf report containing Bar graphs comparing the perf metric value per-metric.
Each metric will contain comparison bars per configuration.

For example: If we run the script with 3 files and test 2 configurations per metric, we will have 2 groups of 3 bars - A group per-configuration, each group containing the 3 performance metrics reported in the 3 files.

To generate this report:
```
python tests/integration/defs/perf/create_perf_comparison_report.py --output_path=<output_path_for_report> --files <csv file paths separated by spaces>
```

This will create a pdf file at <output_path_for_report>.
