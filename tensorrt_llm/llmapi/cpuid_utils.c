/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Minimal x86 CPUID Python extension for querying CPU feature flags.
 *
 * Exposes has_pct_support() to Python, which checks whether the CPU
 * supports Intel Priority Core Turbo (PCT) via CPUID.
 */

#include <Python.h>
#include <cpuid.h>
#include <stdint.h>

/*
 * Check whether the CPU supports Intel Priority Core Turbo (PCT).
 *
 * PCT requires both:
 *   1. Intel Granite Rapids CPU (Family 6, Model 0xAD or 0xAE)
 *   2. SST (Speed Select Technology) support: CPUID leaf 0x06, EAX bit 10
 *
 * Returns 1 if supported, 0 otherwise.
 */
static int has_pct_support(void)
{
    unsigned int eax, ebx, ecx, edx;

    /* Leaf 1: CPU family and model identification */
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return 0;

    uint32_t family = (eax >> 8) & 0xF;
    uint32_t model = (eax >> 4) & 0xF;
    uint32_t extended_family = (eax >> 20) & 0xFF;
    uint32_t extended_model = (eax >> 16) & 0xF;

    uint32_t display_family = (family == 0xF) ? family + extended_family : family;
    uint32_t display_model = (family == 0x6 || family == 0xF)
                                 ? (extended_model << 4) | model
                                 : model;

    if (display_family != 6 || (display_model != 0xAD && display_model != 0xAE))
        return 0;

    /* Leaf 0x06: Thermal and Power Management — bit 10 = SST support */
    if (!__get_cpuid(0x06, &eax, &ebx, &ecx, &edx))
        return 0;

    return (eax >> 10) & 1;
}

static PyObject* py_has_pct_support(PyObject* self, PyObject* args)
{
    return PyBool_FromLong(has_pct_support());
}

static PyMethodDef cpuid_methods[] = {{"has_pct_support", py_has_pct_support, METH_NOARGS,
                                          "Return True if the CPU supports Intel Priority Core Turbo (PCT)."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef cpuid_module = {PyModuleDef_HEAD_INIT, "_cpuid_utils",
    "x86 CPUID utilities for detecting CPU features such as PCT.", -1, cpuid_methods};

PyMODINIT_FUNC PyInit__cpuid_utils(void)
{
    return PyModule_Create(&cpuid_module);
}
