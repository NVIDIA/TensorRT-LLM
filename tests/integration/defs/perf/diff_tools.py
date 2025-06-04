from io import StringIO

import numpy as np
import pandas as pd

PERF_CASE_NAME = 'perf_case_name'
PERF_METRIC = 'perf_metric'
THRESHOLD = 'threshold'
ABSOLUTE_THRESHOLD = 'absolute_threshold'
METRIC_TYPE = 'metric_type'
IGNORED_METRICS = {'BUILD_TIME'}


def load_file(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file)


def get_intersecting_metrics(
    base: pd.DataFrame, target: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    missing_from_target = base.index.difference(target.index)
    missing_from_base = target.index.difference(base.index)

    cleaned_base = base.drop(missing_from_target).sort_index()
    cleaned_target = target.drop(missing_from_base).sort_index()
    return cleaned_base, cleaned_target, base.loc[
        missing_from_target], target.loc[missing_from_base]


def get_diff_exceeding_threshold(
        base: pd.DataFrame,
        target: pd.DataFrame) -> tuple[np.array, pd.DataFrame]:
    diff_exceeding_threshold = ~np.isclose(base[PERF_METRIC],
                                           target[PERF_METRIC],
                                           rtol=abs(base[THRESHOLD]),
                                           atol=abs(base[ABSOLUTE_THRESHOLD]))
    diff_exceeding_threshold = np.array([
        diff and base[METRIC_TYPE][i] not in IGNORED_METRICS
        for i, diff in enumerate(diff_exceeding_threshold)
    ])
    diff_mask = np.tile(diff_exceeding_threshold[:, None],
                        (1, target.shape[-1]))
    return diff_exceeding_threshold, target.where(diff_mask, base)


def get_full_diff(base: pd.DataFrame, target: pd.DataFrame,
                  missing_from_base: pd.Series, missing_from_target: pd.Series,
                  diff_over_threshold: np.array) -> pd.DataFrame:
    PERF_METRIC_BASE = f'{PERF_METRIC}_base'
    PERF_METRIC_TARGET = f'{PERF_METRIC}_target'
    thershold_diff = pd.merge(base,
                              target,
                              on=PERF_CASE_NAME,
                              how='outer',
                              suffixes=['_base', '_target'])
    if not thershold_diff.empty:
        thershold_diff = thershold_diff[diff_over_threshold][[
            PERF_METRIC_BASE, PERF_METRIC_TARGET
        ]]
    missing_from_base = missing_from_base.rename(
        columns={PERF_METRIC: PERF_METRIC_TARGET})[[PERF_METRIC_TARGET]]
    missing_from_target = missing_from_target.rename(
        columns={PERF_METRIC: PERF_METRIC_BASE})[[PERF_METRIC_BASE]]
    return pd.concat([thershold_diff, missing_from_base, missing_from_target])


def get_diff(base: pd.DataFrame,
             target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    relevant_columns = base.columns
    base = base.set_index(PERF_CASE_NAME)
    target = target.set_index(PERF_CASE_NAME)
    cleaned_base, cleaned_target, missing_from_target, missing_from_base = get_intersecting_metrics(
        base, target)
    diff_over_threshold, new_df = get_diff_exceeding_threshold(
        cleaned_base, cleaned_target)
    full_diff = get_full_diff(cleaned_base, cleaned_target, missing_from_base,
                              missing_from_target, diff_over_threshold)
    return full_diff, pd.concat([new_df, missing_from_base
                                 ]).reset_index()[relevant_columns]


def get_csv_lines(df: pd.DataFrame) -> list[str]:
    string_buffer = StringIO()
    df.to_csv(string_buffer, index=False)
    string_buffer.seek(0)
    return string_buffer.readlines()
