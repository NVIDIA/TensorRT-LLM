/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

/* Augment the default ELF layout. CUDA payloads can exceed 2 GiB, so place
 * them after host code, unwind tables, and data. Keep .nvFatBinSegment (the
 * registration wrappers containing pointers to the payloads) near host data.
 */
SECTIONS
{
    .nv_fatbin : { *(.nv_fatbin) }
    __nv_relfatbin : { *(__nv_relfatbin) }
}
INSERT AFTER .bss;
