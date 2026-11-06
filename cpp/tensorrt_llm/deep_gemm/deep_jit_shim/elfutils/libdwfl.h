/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Minimal declarations of the libdwfl (elfutils) entry points that DeepJIT's
 * exception backtraces resolve through dlopen("libdw.so.1"). DeepJIT only
 * includes <elfutils/libdwfl.h> for these types and prototypes and never links
 * libdw, so build hosts without the elfutils development headers use this
 * shim instead; when the real header is installed it takes precedence.
 */

#pragma once

#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef uint64_t Dwarf_Addr;
    typedef uint64_t Dwarf_Word;
    typedef uint64_t GElf_Addr;
    typedef uint32_t GElf_Word;

    typedef struct Elf Elf;
    typedef struct Dwfl Dwfl;
    typedef struct Dwfl_Module Dwfl_Module;
    typedef struct Dwfl_Line Dwfl_Line;
    typedef struct Dwfl_Shim_Elf64_Shdr GElf_Shdr;

    typedef struct
    {
        int (*find_elf)(
            Dwfl_Module* mod, void** userdata, char const* modname, Dwarf_Addr base, char** file_name, Elf** elfp);
        int (*find_debuginfo)(Dwfl_Module* mod, void** userdata, char const* modname, Dwarf_Addr base,
            char const* file_name, char const* debuglink_file, GElf_Word debuglink_crc, char** debuginfo_file_name);
        int (*section_address)(Dwfl_Module* mod, void** userdata, char const* modname, Dwarf_Addr base,
            char const* secname, GElf_Word shndx, GElf_Shdr const* shdr, Dwarf_Addr* addr);
        char** debuginfo_path;
    } Dwfl_Callbacks;

    Dwfl* dwfl_begin(Dwfl_Callbacks const* callbacks);
    int dwfl_linux_proc_report(Dwfl* dwfl, pid_t pid);
    int dwfl_report_end(Dwfl* dwfl,
        int (*removed)(Dwfl_Module* mod, void* userdata, char const* name, Dwarf_Addr low_addr, void* arg), void* arg);
    Dwfl_Module* dwfl_addrmodule(Dwfl* dwfl, Dwarf_Addr address);
    char const* dwfl_module_addrname(Dwfl_Module* mod, GElf_Addr address);
    Dwfl_Line* dwfl_module_getsrc(Dwfl_Module* mod, Dwarf_Addr address);
    char const* dwfl_lineinfo(
        Dwfl_Line* line, Dwarf_Addr* addr, int* linep, int* colp, Dwarf_Word* mtime, Dwarf_Word* length);
    int dwfl_linux_proc_find_elf(
        Dwfl_Module* mod, void** userdata, char const* modname, Dwarf_Addr base, char** file_name, Elf** elfp);

#ifdef __cplusplus
}
#endif
