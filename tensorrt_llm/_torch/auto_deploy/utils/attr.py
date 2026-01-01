# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


def get_attr_by_name(obj, name):
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def set_attr_by_name(obj, name, value):
    parts = name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def del_attr_by_name(obj, name):
    parts = name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    delattr(obj, parts[-1])
