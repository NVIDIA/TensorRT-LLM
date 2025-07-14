import asyncio
import json
from typing import List

import aiohttp

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import *
from ..pyexecutor.scheduler import ScheduledRequests
from .drafter import Drafter


class APIDrafter(Drafter):

    def __init__(
        self,
        spec_config: "ExternalAPIConfig",
    ):
        super().__init__()
        self.max_draft_len = spec_config.max_draft_len
        self.endpoint = spec_config.endpoint
        assert self.endpoint is not None, "API endpoint is required for external API speculative decoding."
        self.template = spec_config.template if spec_config.template is not None else {}
        self.response_field = spec_config.response_field if spec_config.response_field is not None else "draft_tokens"

    def single_draft_call(self):
        return True

    def get_nested_field_from_response(self, response: dict) -> List[int]:
        # Allows for nested fields in the response.
        # Example: "choices.0.message.content"
        # Returns the value of the nested field: response["choices"][0]["message"]["content"]
        keys = self.response_field.split(".")
        current = response

        for key in keys:
            try:
                if key.isdigit():
                    key = int(key)
                    if isinstance(current, list) and 0 <= key < len(current):
                        current = current[key]
                    else:
                        logger.warning(
                            f"Response field {self.response_field} is invalid for response {response}. Index {key} is invalid."
                        )
                        return []
                else:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        logger.warning(
                            f"Response field {self.response_field} is invalid for response {response}. Index {key} is invalid."
                        )
                        return []

            except (KeyError, ValueError, IndexError):
                logger.warning(
                    f"Response field path is invalid: {self.response_field}")
                return []

        if not isinstance(current, list):
            logger.warning(
                f"API response '{self.response_field}' must be a list. Got type: {type(current)}"
            )
            return []
        return current

    async def get_draft_tokens(
        self,
        prefix: list[int],
        request_id: int,
        end_id: int,
        max_sequence_length: int,
    ) -> List[int]:
        try:
            request_data = {
                "prefix": prefix,
                "request_id": request_id,
                "end_id": end_id,
                "max_sequence_length": max_sequence_length,
            }
            if self.template:
                request_data.update(self.template)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url=self.endpoint,
                        json=request_data,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=10),
                ) as response:

                    # check for unsuccessful response
                    if response.status != 200:
                        logger.error(
                            f"Failed to get draft tokens. API call failed for request {request_id} with status code {response.status}"
                        )
                        return []

                    result = await response.json()
                    draft_tokens = self.get_nested_field_from_response(result)
                    if len(draft_tokens) > self.max_draft_len:
                        draft_tokens = draft_tokens[:self.max_draft_len]
                    logger.debug(
                        f"Retrieved draft tokens for request {request_id}: {draft_tokens}"
                    )
                    return draft_tokens

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON response for request {request_id}: {e}")
            return []

        except Exception as e:
            logger.error(
                f"Failed to get draft tokens. API call failed for request {request_id} with the following error: {e}"
            )
            return []

    async def async_prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: None,
    ) -> None:
        # Sort by request_id when py_batch_idx is None as a fallback.
        # This happens in the disagg case: for a set of new requests, we draft
        # before forward_step, so py_batch_idx is not assigned.
        sorted_requests = sorted(
            scheduled_requests.generation_requests,
            key=lambda r:
            (r.py_batch_idx is None, r.py_batch_idx or r.request_id),
        )

        tasks = []
        for request in sorted_requests:
            # Add new token to a copy of the generated tokens to find new draft tokens
            prefix = list(request.get_tokens()[0])  # Get a copy
            task = self.get_draft_tokens(
                prefix,
                request.request_id,
                request.py_end_id,
                request.py_orig_prompt_len + request.py_max_new_tokens,
            )
            tasks.append(task)

        try:
            all_draft_tokens = await asyncio.wait_for(asyncio.gather(
                *tasks, return_exceptions=True),
                                                      timeout=10.0)
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout occurred while getting draft tokens for batch of requests"
            )
            all_draft_tokens = [[] for _ in tasks]

        for request, draft_tokens in zip(sorted_requests, all_draft_tokens):
            if isinstance(draft_tokens, Exception):
                logger.error(
                    f"An exception occurred while getting draft tokens for request {request.request_id}. Set TLLM_LOG_LEVEL for more details."
                )
                draft_tokens = []
            elif len(draft_tokens) == 0:
                logger.error(
                    f"Draft tokens could not be generated for request {request.request_id}. Set TLLM_LOG_LEVEL for more details."
                )
            else:
                # Pad length to `self.max_draft_len`
                if len(draft_tokens) > 0:
                    pad_length = self.max_draft_len - len(draft_tokens)
                    draft_tokens.extend([request.py_end_id] * pad_length)

            request.py_draft_tokens = draft_tokens

    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: None,
    ) -> None:
        asyncio.run(
            self.async_prepare_draft_tokens(scheduled_requests,
                                            resource_manager))
