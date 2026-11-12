# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import msgspec

from tensorrt_llm.serve.kv_hints import KvHintsRequest, KvHintsRoute

_msgpack_decoder = msgspec.msgpack.Decoder()


class _MsgspecRequest(KvHintsRequest):
    """Request that decodes X-TRTLLM-Msgpack bodies with msgspec."""

    async def json(self):
        if not hasattr(self, "_json_body"):
            body = await self.body()
            if not body:
                self._json_body = {}
            elif self.headers.get("x-trtllm-msgpack") == "1":
                try:
                    self._json_body = _msgpack_decoder.decode(body)
                except msgspec.DecodeError as e:
                    raise json.JSONDecodeError(f"msgpack: {e}", "", 0) from e
            else:
                self._json_body = json.loads(body)
        return self.filter_kv_hints(self._json_body)


class _MsgspecRoute(KvHintsRoute):
    """Parse msgpack/JSON and apply hint trust policy before model validation."""

    request_type = _MsgspecRequest
