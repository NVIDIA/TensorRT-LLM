# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resource definitions for the FMHA decode TS kernel.

--------------------------------

SMEM resources
~~~~~~~~~~~~~~
- SmemQResource           : Q tile in SMEM, q_stages-deep TMA pipeline.
                            Producer (LoadTask): TMA Q GMEM -> SMEM (one or two
                            head-dim halves for 16-bit Q).  Consumer (MmaTask):
                            builds a tcgen05 SMEM descriptor per stage for the
                            BMM1 (QK) A operand.  Q is loaded once per work
                            tile in HEAD and reused across all BMM1 calls.

- SmemPageOffsetsKvResource : SMEM-cached paged-KV page table entries.
                            Producer (dedicated prefetch warp): coalesced load
                            of a 32-page window from ``page_idx_kv`` into SMEM
                            (one stage per HEAD/LOOP K0/K1/V0/V1 cadence).
                            Consumer (TMA load warp): reads the ``pages_per_tile``
                            slice for the current tile when issuing page-sized
                            TMA copies.  Multiple consecutive tiles whose page
                            IDs share a 32-page window reuse the same stage.

- SmemBlockSparseKvMetadataResource : Pipeline-free prepared route metadata
                            retained from one K load through the matching V.

- SmemBlockSparseSoftmaxMetadataResource : Staged prepared route/token metadata
                            copied to Softmax task-local registers before the
                            corresponding pipeline stage is released.

- SmemKvResource          : Shared SMEM ring for K and V tiles.  K and V
                            alternate in one allocation/pipeline; consumer
                            descriptors target the same SMEM but may use
                            different leading-byte offsets for K vs. V.
                            Producers (LoadTask): TMA K0/K1/V0/V1 GMEM -> SMEM
                            (paged or contiguous, depending on cfg).  Consumers
                            (MmaTask): build tcgen05 K descriptors (BMM1 B
                            operand) and V descriptors (BMM2 B operand).

- SmemKvTileResource      : Dedicated SMEM tile used by split-head-dimension
                            profiles for one K or V producer instance.

- SmemPResource           : P operand for BMM2.  The validated one-instance
                            staged-D256 Keeps profile places P in two TMEM views
                            aliased with the matching S stages; other profiles
                            use an SMEM tile. Producer
                            (Softmax): converts S in registers to P, writes the
                            selected operand layout, and publishes per-lane
                            local sums back through TmemS. Consumer (MmaTask):
                            publishes the matching TMEM address or SMEM
                            descriptor used by BMM2.

TMEM resources
~~~~~~~~~~~~~~
- TmemSResource           : TMEM score buffer (BMM1 accumulator / softmax
                            input).  Producer (MmaTask): QK MMA -> S in TMEM.
                            Consumer (Softmax): loads S to registers, maintains
                            running row max/sum, applies optional causal /
                            sliding-window / attention-sink masks, and feeds the
                            P producer.  Also owns the SMEM softmax scratch
                            buffer used for the cross-warp atomic-max reduction.

- TmemOResource           : TMEM O accumulator for BMM2, o_stages deep.
                            Producer (MmaTask): P x V MMA -> O.  Consumer
                            (Correction): tracks which O stage is ready
                            (``o_stage_idx`` plus tail stage indices) so the
                            in-place rescale path can find the correct columns.

- TmemSoftmaxLocalResource : TMEM-local softmax stats exchanged with the
                            correction warps.  Producer (Softmax): writes the
                            per-loop ``old_max``/``new_max``/``sum`` arrays and
                            the tail-visible per-instance copies.  Consumer
                            (Correction): loads the stats to drive O rescaling
                            (LOOP) and final normalization (TAIL).

- TmemSoftmaxOrderResource : Barrier-only ordering resource for softmax-stat
                             publication and correction consumption.

- TmemStatsDoneResource   : Barrier-only lifetime credit for TMEM columns
                            shared by S and local stats. MMA acquires it before
                            overwriting S; Correction returns it after loading
                            the matching stats into registers.

- TmemSoftmaxGlobalResource : FP8 sum-correction helper.  Producer-only
                            resource that, after P quantization, reapplies the
                            running-max correction to the running denominator
                            (using the TmemS local-sum array) and publishes the
                            corrected sums back through TmemS.  Inactive when
                            non-FP8 Q/K/V.

- TmemCorrResource        : Correction and output resource.  LOOP stages
                            rescale an in-flight O tile when the running max
                            changes; TAIL stages combine the two BMM2 instances,
                            normalize by the final denominator, and either
                            store the final O tile or write partial O for a
                            split-KV reduction.

All resource classes derive from ``DecodeGenResourceBase`` (a thin
``MemoryResource`` subclass that marks ``consumer_vars`` / ``producer_vars``
as Constexpr so the @cute.jit tracer does not traverse them during
dynamic-if serialization).
"""

from .helpers_common import DecodeGenResourceBase
from .smem_resources import (
    SmemKvTileResource,
    SmemKvResource,
    SmemPageOffsetsKvResource,
    SmemQResource,
)
from .smem_block_sparse_metadata import (
    SmemBlockSparseKvMetadataResource,
    SmemBlockSparseSoftmaxMetadataResource,
)
from .tmem_corr import TmemCorrResource
from .tmem_o import TmemOResource
from .smem_p import SmemPResource
from .tmem_s import TmemSResource
from .tmem_softmax_stats import (
    TmemStatsDoneResource,
    TmemSoftmaxGlobalResource,
    TmemSoftmaxLocalResource,
    TmemSoftmaxOrderResource,
)

__all__ = [
    "DecodeGenResourceBase",
    "SmemKvResource",
    "SmemKvTileResource",
    "SmemPageOffsetsKvResource",
    "SmemPResource",
    "SmemQResource",
    "SmemBlockSparseKvMetadataResource",
    "SmemBlockSparseSoftmaxMetadataResource",
    "TmemCorrResource",
    "TmemOResource",
    "TmemSResource",
    "TmemStatsDoneResource",
    "TmemSoftmaxGlobalResource",
    "TmemSoftmaxLocalResource",
    "TmemSoftmaxOrderResource",
]
