# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy sparse metadata views shared by attention and split reduction."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64


class HeadIndexedMetadataView:
    """Read one KV head's values from interleaved [route, pattern_head] rows."""

    def __init__(self, values, heads, head):
        self.values = (values, heads, head)

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return HeadIndexedMetadataView(
            *cutlass.new_from_mlir_values(self.values, values)
        )

    @cute.jit
    def __getitem__(self, route):
        values, heads, head = self.values
        return values[Int64(route) * Int64(heads) + Int64(head)]


class DirectSparseMetadataView:
    """Lazy sparse metadata resolved from raw inputs for one query per route.

    The combined G1 path skips metadata construction: attention resolves block
    IDs, causal tails, physical locators and lengths directly from the inputs.
    G1 means one token per route, not one token per request; packed and fixed
    requests may each contain several routes. Split reduction can still run.
    """

    def __init__(
        self,
        inputs,
        qo_indptr,
        *,
        packed,
        model_len,
        page_size,
        page_capacity,
        lengths,
        sparse_block_size=4,
        fragment_size=4,
        head=0,
    ):
        self.values = (*inputs, qo_indptr, head)
        self.packed = packed
        self.model_len = model_len
        self.page_size = page_size
        self.page_capacity = page_capacity
        self.lengths = lengths
        self.sparse_block_size = sparse_block_size
        self.fragment_size = fragment_size
        self.fragments_per_block = sparse_block_size // fragment_size

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        new = cutlass.new_from_mlir_values(self.values, values)
        return DirectSparseMetadataView(
            new[:4],
            new[4],
            packed=self.packed,
            model_len=self.model_len,
            page_size=self.page_size,
            page_capacity=self.page_capacity,
            lengths=self.lengths,
            sparse_block_size=self.sparse_block_size,
            fragment_size=self.fragment_size,
            head=new[5],
        )

    def with_head(self, head):
        return DirectSparseMetadataView(
            self.values[:4],
            self.values[4],
            packed=self.packed,
            model_len=self.model_len,
            page_size=self.page_size,
            page_capacity=self.page_capacity,
            lengths=self.lengths,
            sparse_block_size=self.sparse_block_size,
            fragment_size=self.fragment_size,
            head=head,
        )

    @cute.jit
    def resolve_route(self, route):
        blocks, table, requests, positions, offsets, _ = self.values
        row = route
        valid = (row >= Int32(0)) & (row < Int32(blocks.shape[0]))
        if cutlass.const_expr(self.packed):
            row = Int32(offsets[route])
            end = Int32(offsets[route + Int32(1)])
            valid = (
                (row >= Int32(0))
                & (row < Int32(blocks.shape[0]))
                & (end == row + Int32(1))
            )
        request = Int32(-1)
        visible = Int32(0)
        if valid:
            request = Int32(requests[row])
            position = Int64(positions[row])
            valid = (
                (request >= Int32(0))
                & (request < Int32(table.shape[0]))
                & (position >= Int64(0))
                & (position < Int64(self.model_len))
            )
            if valid:
                visible = Int32(position + Int64(1))
        return row, request, visible

    @cute.jit
    def load_page(self, route_state, slot):
        blocks = self.values[0]
        row, _, visible = route_state
        complete = cute.math.min(
            visible // Int32(self.sparse_block_size), Int32(blocks.shape[-1])
        )
        logical = Int32(-1)
        candidate = slot // Int32(self.fragments_per_block)
        if candidate < complete:
            if cutlass.const_expr(len(blocks.shape) == 3):
                logical = Int32(blocks[row, self.values[5], candidate])
            else:
                logical = Int32(blocks[row, candidate])
        return self.map_page(route_state, slot, logical)

    @cute.jit
    def map_page(self, route_state, slot, logical):
        blocks, table, _, _, _, _ = self.values
        _, request, visible = route_state
        complete = cute.math.min(
            visible // Int32(self.sparse_block_size), Int32(blocks.shape[-1])
        )
        tail = visible % Int32(self.sparse_block_size)
        candidate = slot // Int32(self.fragments_per_block)
        result = Int32(-1)
        if candidate < complete:
            if (logical < Int32(0)) | (
                logical >= visible // Int32(self.sparse_block_size)
            ):
                logical = Int32(-1)
        elif (candidate == complete) & (
            slot % Int32(self.fragments_per_block) * Int32(self.fragment_size) < tail
        ):
            logical = visible // Int32(self.sparse_block_size)
        else:
            logical = Int32(-1)
        if logical >= Int32(0):
            fragment = logical * Int32(self.fragments_per_block) + slot % Int32(
                self.fragments_per_block
            )
            fragments_per_page = Int32(self.page_size // self.fragment_size)
            storage_page = fragment // fragments_per_page
            subpage = fragment % fragments_per_page
            if storage_page < Int32(table.shape[1]):
                physical = Int32(table[request, storage_page])
                if physical >= Int32(0):
                    result = physical * fragments_per_page + subpage
        return result

    @cute.jit
    def __getitem__(self, index):
        if cutlass.const_expr(self.lengths):
            _, _, visible = self.resolve_route(Int32(index))
            complete = cute.math.min(
                visible // Int32(self.sparse_block_size),
                Int32(self.values[0].shape[-1]),
            )
            tail = visible % Int32(self.sparse_block_size)
            result = cute.math.max(
                complete * Int32(self.sparse_block_size) + tail, Int32(1)
            )
        else:
            route = Int32(index // Int64(self.page_capacity))
            slot = Int32(index % Int64(self.page_capacity))
            result = self.load_page(self.resolve_route(route), slot)
        return result
