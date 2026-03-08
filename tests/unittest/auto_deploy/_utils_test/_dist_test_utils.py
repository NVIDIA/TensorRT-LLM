import pytest
import torch


def get_device_counts(num_gpu_list=None):
    num_gpu_list = [1, 2] if num_gpu_list is None else num_gpu_list
    return [param_with_device_count(n) for n in num_gpu_list]


def param_with_device_count(n: int, *args, marks_extra=None):
    gpu_count = torch.cuda.device_count()
    marks = [pytest.mark.skipif(gpu_count < n, reason=f"need {n} GPUs!")]
    marks.extend(marks_extra or [])
    return pytest.param(n, *args, marks=marks)
