# Copyright (c) 2025-2026, NVIDIA CORPORATION.
#
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

import asyncio
import os
from typing import Awaitable, Callable, Optional, TypeVar
from uuid import UUID, uuid4

from tensorrt_llm._torch.disaggregation.capability_negotiation import (
    negotiate_generation_safe_lifecycle,
    validate_generation_safe_advertisement,
)
from tensorrt_llm.disaggregated_params import TransceiverLifecycleAdvertisement
from tensorrt_llm.llmapi.disagg_utils import ConditionalDisaggConfig, DisaggServerConfig, ServerRole
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinator
from tensorrt_llm.serve.disagg_lifecycle_control import (
    ContextArtifactAbortRequest,
    GenerationGrantAbortRequest,
    GenerationGrantDecisionResponse,
    GenerationGrantRenewRequest,
    GenerationGrantRequest,
)
from tensorrt_llm.serve.openai_client import OpenAIClient, _OwnedAsyncGenerator
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    DisaggregatedParams,
    DisaggScheduleStyle,
    UCompletionRequest,
    UCompletionResponse,
)
from tensorrt_llm.serve.openai_service import OpenAIService
from tensorrt_llm.serve.responses_utils import (
    ResponseHooks,
    UCompletionResponseOrGenerator,
    done_generator,
)
from tensorrt_llm.serve.router import CoordinatorDelegatingRouter, KvCacheAwareRouter, Router

# Finish reasons for which a GEN handoff is still pending; any other reason means
# the CTX request already completed and the disagg KV-cache handoff was never set up.
_GEN_PENDING_FINISH_REASONS = ("length", "not_finished")
_T = TypeVar("_T")


class _GenerationAdmissionRejected(RuntimeError):
    """Raised when the selected GEN endpoint cannot own this request."""


class OpenAIDisaggregatedService(OpenAIService):
    def __init__(
        self,
        config: DisaggServerConfig,
        coordinator: "DisaggCoordinator",
        client_factory: Callable[[Router, ServerRole], OpenAIClient],
        req_timeout_secs: int = 180,
    ):
        self._config = config
        # The service drives the coordinator's ctx/gen routers uniformly, so serving
        # is identical whether the router is the real one (single-process) or a
        # delegating one that forwards placement to a remote coordinator (worker).
        self._coordinator = coordinator
        self._ctx_router = coordinator.ctx_router
        self._gen_router = coordinator.gen_router
        self._client_factory = client_factory
        self._req_timeout_secs = req_timeout_secs
        # Opt-in body-shrink for generation_only requests; see _get_gen_request.
        self._strip_gen_message_history = config.gen_strip_message_history
        # Opt-in: ask context workers to return prompt_token_ids as base64 int32.
        self._tokids_ctxbytes = config.gen_tokids_ctxbytes

        self._ctx_client = None
        self._gen_client = None
        self._schedule_style = DisaggScheduleStyle.CONTEXT_FIRST
        self._lifecycle_protocol_version = int(
            os.getenv("TRTLLM_DISAGG_LIFECYCLE_PROTOCOL_VERSION", "0")
        )
        if self._lifecycle_protocol_version not in (0, 1):
            raise ValueError("TRTLLM_DISAGG_LIFECYCLE_PROTOCOL_VERSION must be 0 or 1")
        self._generation_grant_ttl_s = float(os.getenv("TRTLLM_DISAGG_GEN_GRANT_TTL_S", "600"))
        self._generation_grant_renew_interval_s = float(
            os.getenv("TRTLLM_DISAGG_GEN_GRANT_RENEW_INTERVAL_S", "200")
        )
        artifact_ttl_s = float(os.getenv("TRTLLM_DISAGG_ARTIFACT_TTL_S", "60"))
        default_admission_timeout_s = min(req_timeout_secs, artifact_ttl_s / 2.0)
        self._lifecycle_admission_timeout_s = float(
            os.getenv(
                "TRTLLM_DISAGG_LIFECYCLE_ADMISSION_TIMEOUT_S",
                str(default_admission_timeout_s),
            )
        )
        if self._generation_grant_ttl_s <= 0:
            raise ValueError("generation grant TTL must be positive")
        if not (0 < self._generation_grant_renew_interval_s < self._generation_grant_ttl_s):
            raise ValueError("generation grant renewal interval must be below its TTL")
        if not 0 < self._lifecycle_admission_timeout_s < artifact_ttl_s:
            raise ValueError(
                "generation lifecycle admission timeout must be positive and "
                "below the context artifact TTL"
            )

        match self._config.schedule_style:
            case "generation_first":
                self._send_disagg_request = self._send_disagg_request_gen_first
                self._schedule_style = DisaggScheduleStyle.GENERATION_FIRST
                logger.info(
                    f"Using generation first disagg schedule style, schedule_style: {self._config.schedule_style}"
                )
            case _:
                self._send_disagg_request = self._send_disagg_request_ctx_first
                self._schedule_style = DisaggScheduleStyle.CONTEXT_FIRST
                logger.info(
                    f"Using context first disagg schedule style, schedule_style: {self._config.schedule_style}"
                )

    async def openai_completion(
        self, request: UCompletionRequest, hooks: Optional[ResponseHooks] = None
    ) -> UCompletionResponseOrGenerator:
        if not await self.is_ready():
            raise RuntimeError("Cluster is not ready")
        if not isinstance(request.prompt, str):
            # Reject empty prompt lists explicitly so the router does not
            # index prompt[0] on an empty list.
            if isinstance(request.prompt, list) and len(request.prompt) == 0:
                raise ValueError("Disaggregated server does not support empty prompt list")
            # Check if it's a list and contains integers
            if type(request.prompt) is list and len(request.prompt) == 1:
                request.prompt = request.prompt[0]
            elif not isinstance(request.prompt, list) or not all(
                isinstance(x, int) for x in request.prompt
            ):
                raise ValueError(
                    "Disaggregated server currently only supports single string prompt or list of integers in request"
                )

        return await self._send_disagg_request(request, hooks)

    async def openai_chat_completion(
        self, request: UCompletionRequest, hooks: Optional[ResponseHooks] = None
    ) -> UCompletionResponseOrGenerator:
        if not await self.is_ready():
            raise RuntimeError("Cluster is not ready")
        return await self._send_disagg_request(request, hooks)

    @property
    def _generation_safe_lifecycle_enabled(self) -> bool:
        return self._lifecycle_protocol_version == 1

    async def _wait_for_lifecycle_admission(self, operation: Awaitable[_T]) -> _T:
        try:
            return await asyncio.wait_for(
                operation,
                timeout=self._lifecycle_admission_timeout_s,
            )
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                "generation lifecycle admission timed out after "
                f"{self._lifecycle_admission_timeout_s:g}s"
            ) from error

    @staticmethod
    def _new_lifecycle_identity(logical_request_id: int) -> dict:
        return {
            "logical_request_id": logical_request_id,
            "prefill_artifact_id": str(uuid4()),
            "artifact_version": 0,
            "handoff_attempt_uuid": str(uuid4()),
            "consumer_grant_id": str(uuid4()),
            "transfer_session_id": str(uuid4()),
        }

    @staticmethod
    def _copy_disaggregated_params(
        params: DisaggregatedParams,
        **updates,
    ) -> DisaggregatedParams:
        values = params.model_dump()
        values.update(updates)
        return DisaggregatedParams.model_validate(values)

    @staticmethod
    def _lifecycle_fields(params: DisaggregatedParams) -> dict:
        return {
            name: getattr(params, name)
            for name in (
                "logical_request_id",
                "prefill_artifact_id",
                "artifact_version",
                "handoff_attempt_uuid",
                "consumer_grant_id",
                "transfer_session_id",
            )
        }

    @classmethod
    def _lifecycle_contract_fields(
        cls,
        params: DisaggregatedParams,
    ) -> dict:
        return {
            **cls._lifecycle_fields(params),
            "generation_endpoint_name": params.generation_endpoint_name,
            "generation_endpoint_rank": params.generation_endpoint_rank,
            "generation_endpoint_incarnation": params.generation_endpoint_incarnation,
            "context_control_endpoint": params.context_control_endpoint,
            "context_transceiver_lifecycle": params.context_transceiver_lifecycle,
            "schedule_style": params.schedule_style,
            "ctx_dp_rank": params.ctx_dp_rank,
        }

    def _verify_lifecycle_echo(
        self,
        expected: DisaggregatedParams,
        actual: DisaggregatedParams,
    ) -> None:
        if not self._generation_safe_lifecycle_enabled:
            return
        if self._lifecycle_contract_fields(expected) != self._lifecycle_contract_fields(actual):
            raise RuntimeError("context worker returned conflicting lifecycle contract")

    @staticmethod
    def _context_lifecycle_from_server_info(
        server_info: dict,
    ) -> TransceiverLifecycleAdvertisement:
        try:
            value = server_info["server_info"]["disaggregated_params"][
                "context_transceiver_lifecycle"
            ]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "selected context endpoint does not advertise a generation-safe "
                "transceiver lifecycle contract"
            ) from error
        advertisement = TransceiverLifecycleAdvertisement.from_value(value)
        validate_generation_safe_advertisement(
            advertisement,
            role="context",
        )
        return advertisement

    async def _admit_generation(
        self,
        request: UCompletionRequest,
        *,
        server: str,
    ) -> tuple[UCompletionRequest, GenerationGrantDecisionResponse]:
        params = request.disaggregated_params
        if params is None:
            raise RuntimeError("generation-safe request is missing disaggregated params")
        decision = await self._gen_client.issue_generation_grant(
            GenerationGrantRequest(
                lifecycle_protocol_version=1,
                **self._lifecycle_fields(params),
                ttl_s=self._generation_grant_ttl_s,
                context_control_endpoint=params.context_control_endpoint,
                context_transceiver_lifecycle=params.context_transceiver_lifecycle,
                schedule_style=params.schedule_style,
                ctx_dp_rank=params.ctx_dp_rank,
            ),
            server=server,
        )
        if not decision.accepted:
            raise _GenerationAdmissionRejected(
                decision.reason or "generation endpoint rejected admission"
            )
        if (
            decision.generation_endpoint_name is None
            or decision.generation_endpoint_rank is None
            or decision.generation_endpoint_incarnation is None
            or decision.generation_transceiver_lifecycle is None
            or decision.ttl_s is None
        ):
            raise RuntimeError(
                "generation endpoint accepted admission without an exact "
                "identity and transceiver lifecycle contract"
            )
        negotiate_generation_safe_lifecycle(
            params.context_transceiver_lifecycle,
            decision.generation_transceiver_lifecycle,
            schedule_style=params.schedule_style,
            ctx_dp_rank=params.ctx_dp_rank,
        )
        admitted_params = self._copy_disaggregated_params(
            params,
            generation_endpoint_name=decision.generation_endpoint_name,
            generation_endpoint_rank=decision.generation_endpoint_rank,
            generation_endpoint_incarnation=str(decision.generation_endpoint_incarnation),
        )
        request.disaggregated_params = admitted_params
        return request, decision

    def _generation_grant_renew_request(
        self,
        params: DisaggregatedParams,
        *,
        sequence: int,
    ) -> GenerationGrantRenewRequest:
        return GenerationGrantRenewRequest(
            lifecycle_protocol_version=1,
            **self._lifecycle_fields(params),
            generation_endpoint_name=params.generation_endpoint_name,
            generation_endpoint_rank=params.generation_endpoint_rank,
            generation_endpoint_incarnation=params.generation_endpoint_incarnation,
            ttl_s=self._generation_grant_ttl_s,
            sequence=sequence,
        )

    def _generation_grant_abort_request(
        self,
        params: DisaggregatedParams,
    ) -> GenerationGrantAbortRequest:
        return GenerationGrantAbortRequest(
            lifecycle_protocol_version=1,
            **self._lifecycle_fields(params),
            generation_endpoint_name=params.generation_endpoint_name,
            generation_endpoint_rank=params.generation_endpoint_rank,
            generation_endpoint_incarnation=params.generation_endpoint_incarnation,
        )

    async def _abort_context_artifact(
        self,
        params: DisaggregatedParams,
        *,
        server: str,
        reason: str = "context artifact abandoned",
    ) -> None:
        response = await self._ctx_client.abort_context_artifact(
            ContextArtifactAbortRequest(
                lifecycle_protocol_version=1,
                **self._lifecycle_fields(params),
                context_endpoint_incarnation=UUID(params.context_transceiver_lifecycle.instance_id),
                reason=reason,
            ),
            server=server,
        )
        if not response.accepted:
            raise RuntimeError(
                "context endpoint rejected artifact abort: "
                f"{response.state} {response.reason}".rstrip()
            )

    async def _best_effort_abort_context_artifact(
        self,
        params: Optional[DisaggregatedParams],
        *,
        server: Optional[str],
        reason: str,
    ) -> None:
        """Independently retire CTX compute/artifacts during coordinator cleanup."""
        if params is None or server is None or params.context_transceiver_lifecycle is None:
            return
        try:
            await asyncio.wait_for(
                self._abort_context_artifact(
                    params,
                    server=server,
                    reason=reason,
                ),
                timeout=self._lifecycle_admission_timeout_s,
            )
        except Exception as error:
            logger.warning(
                "Failed to abort context attempt %s during coordinator cleanup: %s",
                params.consumer_grant_id,
                error,
            )

    async def _abort_generation_grant(
        self,
        params: DisaggregatedParams,
        *,
        server: str,
    ) -> None:
        response = await self._gen_client.abort_generation_grant(
            self._generation_grant_abort_request(params),
            server=server,
        )
        if not response.accepted:
            raise RuntimeError(
                "generation endpoint rejected grant abort: "
                f"{response.state} {response.reason}".rstrip()
            )

    async def _renew_generation_grant(
        self,
        params: DisaggregatedParams,
        *,
        server: str,
        initial_ttl_s: float,
    ) -> None:
        remaining_ttl_s = initial_ttl_s
        sequence = 0
        while True:
            await asyncio.sleep(
                min(
                    self._generation_grant_renew_interval_s,
                    remaining_ttl_s / 3.0,
                )
            )
            request = self._generation_grant_renew_request(
                params,
                sequence=sequence,
            )
            sent_at_s = asyncio.get_running_loop().time()
            response = await self._gen_client.renew_generation_grant(
                request,
                server=server,
            )
            if not response.accepted:
                raise RuntimeError(
                    "generation endpoint no longer owns the admitted request: "
                    f"{response.state} {response.reason}".rstrip()
                )
            if response.ttl_s is None:
                raise RuntimeError(
                    "generation endpoint renewed a grant without its endpoint-owned remaining TTL"
                )
            received_at_s = asyncio.get_running_loop().time()
            deadline_s = sent_at_s + response.ttl_s
            if received_at_s >= deadline_s:
                raise RuntimeError(
                    "generation grant renewal acknowledgement arrived after "
                    "its endpoint-owned deadline"
                )
            remaining_ttl_s = deadline_s - received_at_s
            sequence += 1

    async def _stop_generation_grant_renewal(
        self,
        task: Optional[asyncio.Task],
        *,
        propagate_failure: bool = True,
    ) -> None:
        if task is None:
            return
        task.cancel()
        results = await asyncio.gather(task, return_exceptions=True)
        result = results[0]
        if (
            propagate_failure
            and isinstance(result, Exception)
            and not isinstance(result, asyncio.CancelledError)
        ):
            raise result

    async def _settle_abandoned_generation_stream(
        self,
        params: DisaggregatedParams,
        *,
        server: str,
    ) -> None:
        """Revoke the exact grant after the HTTP client closes its stream."""
        try:
            await self._abort_generation_grant(
                params,
                server=server,
            )
        except Exception as error:
            logger.warning(
                "Failed to abort abandoned generation grant %s: %s",
                params.consumer_grant_id,
                error,
            )

    @staticmethod
    async def _close_generation_stream(
        stream,
        *,
        grant_id: str,
    ) -> None:
        close = getattr(stream, "aclose", None)
        if close is None:
            return
        try:
            await close()
        except Exception as error:
            logger.warning(
                "Failed to close abandoned generation stream for grant %s: %s",
                grant_id,
                error,
            )

    async def _resolve_ambiguous_generation_admission(
        self,
        request: UCompletionRequest,
        *,
        server: str,
    ) -> None:
        """Replay an exact admission request, then revoke any accepted grant.

        Admission is idempotent by ``consumer_grant_id`` and transfer session,
        so replay resolves the accept-vs-lost-response ambiguity without
        creating a second grant.
        """
        params = request.disaggregated_params
        if params is None:
            return
        try:
            admitted, _ = await asyncio.wait_for(
                self._admit_generation(
                    request,
                    server=server,
                ),
                timeout=self._lifecycle_admission_timeout_s,
            )
        except _GenerationAdmissionRejected:
            return
        except Exception as error:
            logger.warning(
                "Could not resolve ambiguous generation admission %s at %s; "
                "endpoint-owned expiry remains the backstop: %s",
                params.consumer_grant_id,
                server,
                error,
            )
            return
        admitted_params = admitted.disaggregated_params
        if admitted_params is None:
            return
        try:
            await asyncio.wait_for(
                self._abort_generation_grant(
                    admitted_params,
                    server=server,
                ),
                timeout=self._lifecycle_admission_timeout_s,
            )
        except Exception as error:
            logger.warning(
                "Failed to revoke resolved generation admission %s at %s; "
                "endpoint-owned expiry remains the backstop: %s",
                admitted_params.consumer_grant_id,
                server,
                error,
            )

    async def _await_context_with_generation_supervision(
        self,
        context_task: asyncio.Task,
        generation_task: asyncio.Task,
        renewal_task: asyncio.Task,
    ):
        """Do not let a failed GEN/grant age invisibly behind long prefill."""
        while True:
            done, _ = await asyncio.wait(
                (context_task, generation_task, renewal_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if renewal_task in done:
                await renewal_task
                raise RuntimeError("generation grant renewal stopped before context completion")
            if context_task in done:
                return await context_task
            if generation_task in done:
                await generation_task
                raise RuntimeError("generation request ended before context completion")

    async def _send_generation_only_without_handoff(
        self,
        request: UCompletionRequest,
        *,
        server: Optional[str],
        req_id: int,
        hooks: Optional[ResponseHooks],
    ) -> UCompletionResponseOrGenerator:
        """Dispatch a local-only GEN request without inventing a CTX contract.

        Conditional-disaggregation bypass and the GEN-only benchmark create no
        cross-side artifact, publication, or transfer obligation. They must
        therefore remain outside lifecycle protocol v1 instead of fabricating
        a source advertisement solely to obtain a generation grant.
        """
        params = request.disaggregated_params
        if params is not None and params.request_type == "context_only":
            request.disaggregated_params = None
        if server is None:
            server, _ = await self._gen_router.get_next_server(
                request,
                req_id=req_id,
            )
        return await self._gen_client.send_request(
            request,
            server=server,
            hooks=hooks,
            req_id=req_id,
        )

    async def _select_and_admit_generation(
        self,
        request: UCompletionRequest,
        *,
        server: Optional[str],
        exclude_server: Optional[str],
        req_id: int,
    ) -> tuple[
        UCompletionRequest,
        str,
        GenerationGrantDecisionResponse,
    ]:
        selected_server = server
        reservation_live = selected_server is not None
        admission_may_be_ambiguous = False

        async def _release_reservation() -> None:
            nonlocal reservation_live
            if not reservation_live:
                return
            await self._gen_router.finish_request(
                request,
                success=False,
                req_id=req_id,
            )
            reservation_live = False

        async def _select_and_admit() -> tuple[
            UCompletionRequest,
            str,
            GenerationGrantDecisionResponse,
        ]:
            nonlocal admission_may_be_ambiguous, reservation_live, selected_server
            if selected_server is None:
                selected_server, _ = await self._gen_router.get_next_server(
                    request,
                    exclude_server=exclude_server,
                    req_id=req_id,
                )
                reservation_live = True
            try:
                admission_may_be_ambiguous = True
                admitted, decision = await self._admit_generation(
                    request,
                    server=selected_server,
                )
                admission_may_be_ambiguous = False
            except _GenerationAdmissionRejected:
                admission_may_be_ambiguous = False
                await _release_reservation()
                if len(self._gen_router.servers) < 2:
                    raise
                first_server = selected_server
                selected_server, _ = await self._gen_router.get_next_server(
                    request,
                    exclude_server=first_server,
                    req_id=req_id,
                )
                reservation_live = True
                admission_may_be_ambiguous = True
                admitted, decision = await self._admit_generation(
                    request,
                    server=selected_server,
                )
                admission_may_be_ambiguous = False
            reservation_live = False
            return admitted, selected_server, decision

        try:
            return await self._wait_for_lifecycle_admission(_select_and_admit())
        except TimeoutError:
            if admission_may_be_ambiguous and selected_server is not None:
                await self._resolve_ambiguous_generation_admission(
                    request,
                    server=selected_server,
                )
            await _release_reservation()
            raise
        except asyncio.CancelledError:
            if admission_may_be_ambiguous and selected_server is not None:
                await self._resolve_ambiguous_generation_admission(
                    request,
                    server=selected_server,
                )
            await _release_reservation()
            raise
        except Exception:
            if admission_may_be_ambiguous and selected_server is not None:
                await self._resolve_ambiguous_generation_admission(
                    request,
                    server=selected_server,
                )
            await _release_reservation()
            raise

    async def _send_disagg_request_ctx_first_v1(
        self,
        request: UCompletionRequest,
        hooks: Optional[ResponseHooks],
    ) -> UCompletionResponseOrGenerator:
        """Produce an immutable CTX artifact, then bind it to an exact GEN."""
        if hooks:
            hooks.on_req_begin(request)
        disagg_request_id = await self._coordinator.get_disagg_request_id()
        if hooks:
            hooks.on_disagg_request_id(disagg_request_id)
        lifecycle_identity = self._new_lifecycle_identity(disagg_request_id)
        gen_server, need_ctx = await self._wait_for_lifecycle_admission(
            self._check_conditional_disagg(request, disagg_request_id)
        )
        gen_reservation_id = disagg_request_id if gen_server else None
        preselected_gen_reservation_live = gen_server is not None
        need_ctx = need_ctx and not await self._check_gen_only_disagg(request)
        if not need_ctx:
            return await self._send_generation_only_without_handoff(
                request,
                server=gen_server,
                req_id=disagg_request_id,
                hooks=hooks,
            )
        ctx_server = None
        ctx_req = None
        ctx_reservation_live = False
        gen_req = request
        gen_reservation_live = False
        grant_params = None
        grant_admitted = False
        renewal_task = None

        async def _cleanup_failed_request() -> None:
            await self._stop_generation_grant_renewal(
                renewal_task,
                propagate_failure=False,
            )
            await self._best_effort_abort_context_artifact(
                None if ctx_req is None else ctx_req.disaggregated_params,
                server=ctx_server,
                reason="context-first coordinator request failed",
            )
            if grant_admitted and grant_params is not None:
                try:
                    await self._abort_generation_grant(
                        grant_params,
                        server=gen_server,
                    )
                except Exception as error:
                    logger.warning(
                        "Failed to abort generation grant %s after coordinator failure: %s",
                        grant_params.consumer_grant_id,
                        error,
                    )
            if gen_reservation_live:
                await self._gen_router.finish_request(
                    gen_req,
                    success=False,
                    req_id=gen_reservation_id,
                )
            elif preselected_gen_reservation_live:
                await self._gen_router.finish_request(
                    request,
                    success=False,
                    req_id=gen_reservation_id,
                )
            if ctx_reservation_live:
                await self._ctx_router.finish_request(
                    ctx_req,
                    success=False,
                    req_id=disagg_request_id,
                )

        try:
            if need_ctx:
                if hooks:
                    hooks.on_ctx_dispatch(request)
                ctx_req = self._get_ctx_request(
                    request,
                    disagg_request_id,
                    lifecycle_identity=lifecycle_identity,
                )
                ctx_server, ctx_server_info = await self._ctx_router.get_next_server(
                    ctx_req,
                    exclude_server=gen_server,
                    req_id=disagg_request_id,
                )
                ctx_reservation_live = True
                ctx_req.disaggregated_params = self._copy_disaggregated_params(
                    ctx_req.disaggregated_params,
                    context_control_endpoint=ctx_server,
                    context_transceiver_lifecycle=self._context_lifecycle_from_server_info(
                        ctx_server_info
                    ),
                )
                # From this point the HTTP client exclusively owns router
                # finalization, including cancellation and transport failure.
                ctx_reservation_live = False
                ctx_response = await self._ctx_client.send_request(
                    ctx_req,
                    server=ctx_server,
                    hooks=hooks,
                    req_id=disagg_request_id,
                )
                await self._verify_ctx_response(ctx_response)
                ctx_response_params = ctx_response.choices[0].disaggregated_params
                self._verify_lifecycle_echo(
                    ctx_req.disaggregated_params,
                    ctx_response_params,
                )
                if ctx_response_params.disagg_request_id != disagg_request_id:
                    raise RuntimeError(
                        "generation-safe CTX response changed the admitted disaggregated request id"
                    )
                if not self._need_gen(ctx_response):
                    if preselected_gen_reservation_live:
                        await self._gen_router.finish_request(
                            request,
                            success=False,
                            req_id=gen_reservation_id,
                        )
                    if request.stream:
                        return done_generator()
                    return ctx_response
                gen_req = self._get_gen_request(
                    request,
                    ctx_response,
                    disagg_request_id,
                )
            else:
                if (
                    gen_req.disaggregated_params is not None
                    and gen_req.disaggregated_params.request_type == "context_only"
                ):
                    gen_req.disaggregated_params = None
                if gen_req.disaggregated_params is None:
                    gen_req = self._get_gen_request(
                        gen_req,
                        ctx_response=None,
                        disagg_request_id=disagg_request_id,
                        lifecycle_identity=lifecycle_identity,
                    )
                else:
                    gen_req.disaggregated_params = self._copy_disaggregated_params(
                        gen_req.disaggregated_params,
                        disagg_request_id=disagg_request_id,
                        ctx_request_id=disagg_request_id,
                        schedule_style=self._schedule_style,
                        **lifecycle_identity,
                    )

            preselected_gen_reservation_live = False
            gen_req, gen_server, grant_decision = await self._select_and_admit_generation(
                gen_req,
                server=gen_server,
                exclude_server=ctx_server,
                req_id=disagg_request_id,
            )
            gen_reservation_id = disagg_request_id
            grant_params = gen_req.disaggregated_params
            grant_admitted = True
            gen_reservation_live = True
            renewal_task = asyncio.create_task(
                self._renew_generation_grant(
                    grant_params,
                    server=gen_server,
                    initial_ttl_s=grant_decision.ttl_s,
                )
            )

            gen_reservation_live = False
            gen_response = await self._gen_client.send_request(
                gen_req,
                server=gen_server,
                hooks=hooks,
                req_id=gen_reservation_id,
            )
            if request.stream:

                async def _stream_with_generation_grant():
                    async for chunk in gen_response:
                        yield chunk

                async def _finish_generation_stream(success: bool) -> None:
                    renewal_error = None
                    try:
                        await self._stop_generation_grant_renewal(
                            renewal_task,
                            propagate_failure=success,
                        )
                    except Exception as error:
                        renewal_error = error
                        success = False
                    if success:
                        return
                    await self._close_generation_stream(
                        gen_response,
                        grant_id=grant_params.consumer_grant_id,
                    )
                    await self._best_effort_abort_context_artifact(
                        ctx_req.disaggregated_params,
                        server=ctx_server,
                        reason="generation stream was abandoned",
                    )
                    await self._settle_abandoned_generation_stream(
                        grant_params,
                        server=gen_server,
                    )
                    if renewal_error is not None:
                        raise renewal_error

                return _OwnedAsyncGenerator(
                    _stream_with_generation_grant(),
                    _finish_generation_stream,
                )
            await self._stop_generation_grant_renewal(renewal_task)
            return gen_response
        except asyncio.CancelledError:
            await _cleanup_failed_request()
            raise
        except Exception:
            await _cleanup_failed_request()
            raise

    async def _send_disagg_request_ctx_first(
        self, request: UCompletionRequest, hooks: Optional[ResponseHooks] = None
    ) -> UCompletionResponseOrGenerator:
        # ctx_response contains a http response with ContextPhaseParams attached after prefill compute is done

        if self._generation_safe_lifecycle_enabled:
            return await self._send_disagg_request_ctx_first_v1(request, hooks)

        if hooks:
            hooks.on_req_begin(request)
        # empty server means client decides which server to use
        ctx_server = None
        disagg_request_id = await self._coordinator.get_disagg_request_id()
        if hooks:
            hooks.on_disagg_request_id(disagg_request_id)
        lifecycle_identity = (
            self._new_lifecycle_identity(disagg_request_id)
            if self._generation_safe_lifecycle_enabled
            else None
        )
        # reserve a gen_server if conditional disagg is needed
        gen_server, need_ctx = await self._check_conditional_disagg(request, disagg_request_id)
        # Context retries may replace disagg_request_id for the KV-transfer
        # handshake. Keep the ID used to reserve the generation server separate
        # so its coordinator-side load is released under the original key.
        gen_reservation_id = disagg_request_id if gen_server else None
        benchmark_gen_only = await self._check_gen_only_disagg(request)
        need_ctx = need_ctx and not benchmark_gen_only
        ctx_response = None
        gen_req = request
        if need_ctx:
            try:
                # Mark ctx-dispatch start: arrival->here is the pre-ctx wait in the
                # orchestrator/fleet (accept queue + event loop + pipeline).
                if hooks:
                    hooks.on_ctx_dispatch(request)
                ctx_req = self._get_ctx_request(
                    request,
                    disagg_request_id,
                    lifecycle_identity=lifecycle_identity,
                )
                # ctx generator is empty
                ctx_server, _ = await self._ctx_router.get_next_server(
                    ctx_req, exclude_server=gen_server, req_id=disagg_request_id
                )
                if self._generation_safe_lifecycle_enabled:
                    ctx_req.disaggregated_params = self._copy_disaggregated_params(
                        ctx_req.disaggregated_params,
                        context_control_endpoint=ctx_server,
                    )
                ctx_response = await self._ctx_client.send_request(
                    ctx_req, server=ctx_server, hooks=hooks, req_id=disagg_request_id
                )
                await self._verify_ctx_response(ctx_response)
                ctx_response_disagg_params = ctx_response.choices[0].disaggregated_params
                self._verify_lifecycle_echo(
                    ctx_req.disaggregated_params,
                    ctx_response_disagg_params,
                )
                if ctx_response_disagg_params.disagg_request_id is not None:
                    disagg_request_id = ctx_response_disagg_params.disagg_request_id
                    if hooks:
                        hooks.on_disagg_request_id(disagg_request_id)
                gen_req = self._get_gen_request(request, ctx_response, disagg_request_id)
            except Exception:
                if gen_server:
                    await self._gen_router.finish_request(
                        request, success=False, req_id=gen_reservation_id
                    )
                raise
        else:
            # When need_ctx=False the gen server handles full generation and
            # must not see client-supplied disaggregated handoff params. The
            # benchmark-only path above is the only trusted source here.
            if not benchmark_gen_only:
                gen_req = request.model_copy(update={"disaggregated_params": None})
            if self._generation_safe_lifecycle_enabled:
                if gen_req.disaggregated_params is None:
                    gen_req = self._get_gen_request(
                        gen_req,
                        ctx_response=None,
                        disagg_request_id=disagg_request_id,
                        lifecycle_identity=lifecycle_identity,
                    )
                else:
                    gen_req.disaggregated_params = self._copy_disaggregated_params(
                        gen_req.disaggregated_params,
                        disagg_request_id=disagg_request_id,
                        ctx_request_id=disagg_request_id,
                        schedule_style=self._schedule_style,
                        **lifecycle_identity,
                    )
        if ctx_response is None or self._need_gen(ctx_response):
            if self._generation_safe_lifecycle_enabled:
                gen_req, gen_server, grant_decision = await self._select_and_admit_generation(
                    gen_req,
                    server=gen_server,
                    exclude_server=ctx_server,
                    req_id=disagg_request_id,
                )
                gen_reservation_id = disagg_request_id
            elif not gen_server:
                gen_server, _ = await self._gen_router.get_next_server(
                    gen_req, exclude_server=ctx_server, req_id=disagg_request_id
                )
                gen_reservation_id = disagg_request_id
            renewal_task = (
                asyncio.create_task(
                    self._renew_generation_grant(
                        gen_req.disaggregated_params,
                        server=gen_server,
                        initial_ttl_s=grant_decision.ttl_s,
                    )
                )
                if self._generation_safe_lifecycle_enabled
                else None
            )
            try:
                gen_response = await self._gen_client.send_request(
                    gen_req,
                    server=gen_server,
                    hooks=hooks,
                    req_id=gen_reservation_id,
                )
            except Exception:
                if renewal_task is not None:
                    renewal_task.cancel()
                    await asyncio.gather(renewal_task, return_exceptions=True)
                raise
            if request.stream and renewal_task is not None:

                async def _stream_with_generation_grant():
                    try:
                        async for chunk in gen_response:
                            yield chunk
                    finally:
                        await self._stop_generation_grant_renewal(renewal_task)

                return _stream_with_generation_grant()
            await self._stop_generation_grant_renewal(renewal_task)
            return gen_response
        else:
            if gen_server:
                await self._gen_router.finish_request(request, req_id=gen_reservation_id)
            if hooks:
                hooks.on_resp_done("", request, ctx_response)
            if request.stream:
                # ctx client will never return a generator when streaming is requested
                # make up for this by returning a done generator
                return done_generator()
            return ctx_response

    def _need_gen(self, response: UCompletionResponse) -> bool:
        if response and response.choices[0].finish_reason not in _GEN_PENDING_FINISH_REASONS:
            del response.choices[0].disaggregated_params
            return False
        return True

    @staticmethod
    def _get_conversation_id(request: UCompletionRequest) -> Optional[str]:
        if request.conversation_params is not None:
            return request.conversation_params.conversation_id
        if request.disaggregated_params is not None:
            return request.disaggregated_params.conversation_id
        return None

    def _get_ctx_request(
        self,
        request: UCompletionRequest,
        disagg_request_id: Optional[int],
        *,
        lifecycle_identity: Optional[dict] = None,
        context_control_endpoint: Optional[str] = None,
    ) -> UCompletionRequest:
        conversation_id = self._get_conversation_id(request)
        lifecycle_identity = lifecycle_identity or {}
        ctx_request = request.model_copy(
            update={
                "disaggregated_params": DisaggregatedParams(
                    request_type="context_only",
                    disagg_request_id=disagg_request_id,
                    schedule_style=self._schedule_style,
                    conversation_id=conversation_id,
                    return_prompt_token_ids_b64=self._tokids_ctxbytes,
                    context_control_endpoint=context_control_endpoint,
                    **lifecycle_identity,
                ),
                "stream": False,
                "stream_options": None,
            }
        )
        return ctx_request

    def _get_gen_request(
        self,
        request: UCompletionRequest,
        ctx_response: Optional[UCompletionResponse],
        disagg_request_id: Optional[int],
        ctx_server_info: Optional[dict] = None,
        lifecycle_identity: Optional[dict] = None,
        context_control_endpoint: Optional[str] = None,
    ) -> UCompletionRequest:
        conversation_id = self._get_conversation_id(request)
        if ctx_response:
            request.disaggregated_params = ctx_response.choices[0].disaggregated_params
            request.disaggregated_params.request_type = "generation_only"
            request.disaggregated_params.schedule_style = self._schedule_style
            request.disaggregated_params.conversation_id = conversation_id
            request.disaggregated_params.ctx_usage = ctx_response.usage
            # Replace the string prompt with prompt_tokens_ids
            if isinstance(request, CompletionRequest):
                request.prompt = ctx_response.prompt_token_ids
            elif isinstance(request, ChatCompletionRequest):
                # Relay the base64 token-id string verbatim (no int-list
                # materialization on the orchestrator loop), else the int array.
                if ctx_response.prompt_token_ids_b64 is not None:
                    request.prompt_token_ids_b64 = ctx_response.prompt_token_ids_b64
                else:
                    request.prompt_token_ids = ctx_response.prompt_token_ids
                # Opt-in: drop conversation history so the gen worker doesn't
                # re-parse the full conversation JSON (dominates its GIL at high
                # concurrency). It uses prompt_token_ids and only reads the last
                # message; tools are preserved. Config-gated because it's unsafe
                # for harmony/multimodal workers (model type is fixed per deploy).
                if (
                    self._strip_gen_message_history
                    and request.messages
                    and len(request.messages) > 1
                ):
                    request.messages = request.messages[-1:]
        else:
            # no ctx response, it's either a generation-only request or a generation-first disagg request
            lifecycle_identity = lifecycle_identity or {}
            request.disaggregated_params = DisaggregatedParams(
                request_type="generation_only",
                ctx_request_id=disagg_request_id,
                disagg_request_id=disagg_request_id,
                schedule_style=self._schedule_style,
                conversation_id=conversation_id,
                context_control_endpoint=context_control_endpoint,
                **lifecycle_identity,
            )
        if ctx_server_info and "server_info" in ctx_server_info:
            disaggregated_params = ctx_server_info["server_info"].get("disaggregated_params", {})
            if disaggregated_params:
                context_lifecycle = disaggregated_params.get("context_transceiver_lifecycle")
                if context_lifecycle is not None:
                    context_lifecycle = TransceiverLifecycleAdvertisement.from_value(
                        context_lifecycle
                    )
                    existing_lifecycle = request.disaggregated_params.context_transceiver_lifecycle
                    if existing_lifecycle is not None and existing_lifecycle != context_lifecycle:
                        raise RuntimeError(
                            "selected context endpoint changed its transceiver "
                            "lifecycle advertisement"
                        )
                    request.disaggregated_params = self._copy_disaggregated_params(
                        request.disaggregated_params,
                        context_transceiver_lifecycle=context_lifecycle,
                    )
                # ctx_info_endpoint from get_disaggregated_params() is a list;
                # the Pydantic model expects a single str.
                ep = disaggregated_params.get("ctx_info_endpoint")
                if isinstance(ep, list) and ep:
                    disaggregated_params = {**disaggregated_params, "ctx_info_endpoint": ep[0]}
                protected_fields = {
                    *self._lifecycle_fields(request.disaggregated_params).keys(),
                    "generation_endpoint_name",
                    "generation_endpoint_rank",
                    "generation_endpoint_incarnation",
                    "context_control_endpoint",
                    "context_transceiver_lifecycle",
                }
                disaggregated_params = {
                    key: value
                    for key, value in disaggregated_params.items()
                    if key not in protected_fields
                }
                request.disaggregated_params = self._copy_disaggregated_params(
                    request.disaggregated_params,
                    **disaggregated_params,
                )

        request.disaggregated_params.disagg_request_id = disagg_request_id
        return request

    async def _check_conditional_disagg(self, request: UCompletionRequest, req_id: int) -> bool:
        if self.conditional_disagg_config:
            local_gen_router = (
                self._gen_router._local
                if isinstance(self._gen_router, CoordinatorDelegatingRouter)
                else self._gen_router
            )
            if not isinstance(local_gen_router, KvCacheAwareRouter):
                raise TypeError(
                    "conditional disaggregation requires a KV-cache-aware generation router"
                )
            # Query kv cache status and select a best gen_server.
            # The server is reserved for generation request
            gen_server, info = await self._gen_router.get_next_server(request, req_id=req_id)
            match_length = info["match_length"]
            total_length = info["num_tokens"]
            need_ctx_decision = (
                match_length == 0
                or total_length - match_length
                > self.conditional_disagg_config.max_local_prefill_length
            )
            # Visibility hook for verifying bypass triggers in disagg deployments.
            logger.debug(
                f"[conditional_disagg] gen={gen_server} match={match_length} "
                f"total={total_length} residual={total_length - match_length} "
                f"max_local_prefill_length="
                f"{self.conditional_disagg_config.max_local_prefill_length} "
                f"→ need_ctx={need_ctx_decision}"
            )
            return gen_server, need_ctx_decision
        return None, True

    async def _check_gen_only_disagg(self, request: UCompletionRequest) -> bool:
        if os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1":
            # Hard-code first token, ctx_request_id for testing
            request.disaggregated_params = DisaggregatedParams(
                request_type="generation_only",
                first_gen_tokens=[7],
                ctx_request_id=1,
                encoded_opaque_state=None,
                draft_tokens=None,
            )
            request.ignore_eos = True
            return True
        return False

    async def is_ready(self) -> bool:
        # Per-request readiness gate for the /v1/ handlers (the server's /health
        # and /cluster_info hook the coordinator directly). Cluster topology
        # (cluster_info) is the coordinator's concern, not the request service's.
        return await self._coordinator.is_ready()

    @property
    def conditional_disagg_config(self) -> Optional[ConditionalDisaggConfig]:
        return self._config.conditional_disagg_config

    async def setup(self) -> None:
        # Build the request-sending clients from the coordinator's routers and share
        # them with the coordinator service so its readiness checks use the same pool.
        self._ctx_client = self._client_factory(
            self._ctx_router, ServerRole.CONTEXT, self._config.max_retries
        )
        self._gen_client = self._client_factory(
            self._gen_router, ServerRole.GENERATION, self._config.max_retries
        )
        if hasattr(self._coordinator, "set_clients"):
            self._coordinator.set_clients(self._ctx_client, self._gen_client)
        await self._coordinator.start()

    async def teardown(self) -> None:
        await self._ctx_client.shutdown()
        await self._gen_client.shutdown()
        await self._coordinator.stop()

    async def _verify_ctx_response(self, ctx_response: UCompletionResponse) -> None:
        if ctx_response:
            for idx, choice in enumerate(ctx_response.choices):
                if choice.disaggregated_params is None:
                    raise ValueError(
                        f"Context server choice {idx} did not return disaggregated params."
                        f" finish_reason={choice.finish_reason!r}"
                    )
                # A CTX request that finished early (e.g. EOS during prefill) never
                # sets up the KV-cache handoff, so ctx_request_id/disagg_request_id
                # stay None. Only enforce them when a GEN handoff is still pending --
                # mirroring _need_gen, which skips the handoff for these responses.
                if choice.finish_reason in _GEN_PENDING_FINISH_REASONS:
                    if choice.disaggregated_params.ctx_request_id is None:
                        raise ValueError(
                            f"Invalid disaggregated params: ctx_request_id is None for choice {idx}."
                            f" finish_reason={choice.finish_reason!r},"
                            f" disagg_request_id={choice.disaggregated_params.disagg_request_id!r}"
                        )
                    if choice.disaggregated_params.disagg_request_id is None:
                        raise ValueError(
                            f"Invalid disaggregated params: disagg_request_id is None for choice {idx}."
                            f" finish_reason={choice.finish_reason!r},"
                            f" ctx_request_id={choice.disaggregated_params.ctx_request_id!r}"
                        )
            return ctx_response

    async def _send_disagg_request_gen_first_v1(
        self,
        request: UCompletionRequest,
        hooks: Optional[ResponseHooks],
    ) -> UCompletionResponseOrGenerator:
        """Run generation-first scheduling with one exact GEN-owned grant."""
        if hooks:
            hooks.on_req_begin(request)
        need_ctx = not await self._check_gen_only_disagg(request)
        disagg_request_id = await self._coordinator.get_disagg_request_id()
        if hooks:
            hooks.on_disagg_request_id(disagg_request_id)
        if not need_ctx:
            return await self._send_generation_only_without_handoff(
                request,
                server=None,
                req_id=disagg_request_id,
                hooks=hooks,
            )
        lifecycle_identity = self._new_lifecycle_identity(disagg_request_id)
        ctx_server = None
        ctx_req = None
        ctx_server_info = None
        ctx_reservation_live = False
        gen_req = None
        gen_server = None
        gen_reservation_live = False
        grant_params = None
        renewal_task = None
        grant_admitted = False
        gen_task = None
        consume_task = None
        ctx_task = None
        ctx_attempt_live = False

        async def _cleanup_failed_request() -> None:
            for task in (ctx_task, consume_task, gen_task):
                if task is not None and not task.done():
                    task.cancel()
            pending_tasks = tuple(
                task for task in (ctx_task, consume_task, gen_task) if task is not None
            )
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await self._stop_generation_grant_renewal(
                renewal_task,
                propagate_failure=False,
            )
            if ctx_attempt_live:
                await self._best_effort_abort_context_artifact(
                    None if ctx_req is None else ctx_req.disaggregated_params,
                    server=ctx_server,
                    reason="generation-first coordinator request failed",
                )
            if grant_admitted and grant_params is not None:
                try:
                    await self._abort_generation_grant(
                        grant_params,
                        server=gen_server,
                    )
                except Exception as error:
                    logger.warning(
                        "Failed to abort generation grant %s after coordinator failure: %s",
                        grant_params.consumer_grant_id,
                        error,
                    )
            if gen_reservation_live:
                await self._gen_router.finish_request(
                    gen_req,
                    success=False,
                    req_id=disagg_request_id,
                )
            if ctx_reservation_live:
                await self._ctx_router.finish_request(
                    ctx_req,
                    success=False,
                    req_id=disagg_request_id,
                )

        try:
            if need_ctx:
                if hooks:
                    hooks.on_ctx_dispatch(request)
                ctx_req = self._get_ctx_request(
                    request,
                    disagg_request_id,
                    lifecycle_identity=lifecycle_identity,
                )
                ctx_server, ctx_server_info = await self._ctx_router.get_next_server(
                    ctx_req,
                    req_id=disagg_request_id,
                )
                ctx_reservation_live = True
                ctx_req.disaggregated_params = self._copy_disaggregated_params(
                    ctx_req.disaggregated_params,
                    context_control_endpoint=ctx_server,
                    context_transceiver_lifecycle=self._context_lifecycle_from_server_info(
                        ctx_server_info
                    ),
                )
            gen_req = self._get_gen_request(
                request,
                ctx_response=None,
                disagg_request_id=disagg_request_id,
                ctx_server_info=ctx_server_info,
                lifecycle_identity=lifecycle_identity,
                context_control_endpoint=ctx_server,
            )
            gen_req, gen_server, grant_decision = await self._select_and_admit_generation(
                gen_req,
                server=None,
                exclude_server=ctx_server,
                req_id=disagg_request_id,
            )
            gen_reservation_live = True
            grant_params = gen_req.disaggregated_params
            grant_admitted = True
            if ctx_req is not None:
                ctx_req.disaggregated_params = self._copy_disaggregated_params(
                    ctx_req.disaggregated_params,
                    generation_endpoint_name=grant_params.generation_endpoint_name,
                    generation_endpoint_rank=grant_params.generation_endpoint_rank,
                    generation_endpoint_incarnation=grant_params.generation_endpoint_incarnation,
                )
            renewal_task = asyncio.create_task(
                self._renew_generation_grant(
                    grant_params,
                    server=gen_server,
                    initial_ttl_s=grant_decision.ttl_s,
                )
            )

            if request.stream:
                gen_reservation_live = False
                gen_response = await self._gen_client.send_request(
                    gen_req,
                    server=gen_server,
                    hooks=hooks,
                    req_id=disagg_request_id,
                )
                if not need_ctx:

                    async def _yield_generation_only():
                        async for chunk in gen_response:
                            yield chunk

                    async def _finish_generation_only(success: bool) -> None:
                        await self._stop_generation_grant_renewal(
                            renewal_task,
                            propagate_failure=success,
                        )
                        if success:
                            return
                        await self._close_generation_stream(
                            gen_response,
                            grant_id=grant_params.consumer_grant_id,
                        )
                        await self._settle_abandoned_generation_stream(
                            grant_params,
                            server=gen_server,
                        )

                    return _OwnedAsyncGenerator(
                        _yield_generation_only(),
                        _finish_generation_only,
                    )

                queue: asyncio.Queue = asyncio.Queue()

                async def _consume_gen():
                    try:
                        async for chunk in gen_response:
                            await queue.put(chunk)
                    except Exception as error:
                        await queue.put(error)
                        raise
                    finally:
                        await queue.put(None)

                consume_task = asyncio.create_task(_consume_gen())
                await asyncio.sleep(0)
                ctx_task = asyncio.create_task(
                    self._ctx_client.send_request(
                        ctx_req,
                        server=ctx_server,
                        hooks=hooks,
                        req_id=disagg_request_id,
                    )
                )
                ctx_reservation_live = False
                ctx_attempt_live = True
                ctx_response = await self._await_context_with_generation_supervision(
                    ctx_task,
                    consume_task,
                    renewal_task,
                )
                ctx_task = None
                await self._verify_ctx_response(ctx_response)
                self._verify_lifecycle_echo(
                    ctx_req.disaggregated_params,
                    ctx_response.choices[0].disaggregated_params,
                )
                if (
                    ctx_response.choices[0].disaggregated_params.disagg_request_id
                    != disagg_request_id
                ):
                    raise RuntimeError(
                        "generation-safe CTX response changed the admitted disaggregated request id"
                    )
                if not self._need_gen(ctx_response):
                    await self._abort_generation_grant(
                        grant_params,
                        server=gen_server,
                    )
                    grant_admitted = False
                    consume_task.cancel()
                    await asyncio.gather(consume_task, return_exceptions=True)
                    await self._stop_generation_grant_renewal(
                        renewal_task,
                        propagate_failure=False,
                    )
                    renewal_task = None
                    return done_generator()

                async def _yield_from_queue():
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        yield item

                async def _finish_queued_generation(success: bool) -> None:
                    renewal_error = None
                    try:
                        await self._stop_generation_grant_renewal(
                            renewal_task,
                            propagate_failure=success,
                        )
                    except Exception as error:
                        renewal_error = error
                        success = False
                    if not consume_task.done():
                        consume_task.cancel()
                    await asyncio.gather(consume_task, return_exceptions=True)
                    if success:
                        return
                    await self._close_generation_stream(
                        gen_response,
                        grant_id=grant_params.consumer_grant_id,
                    )
                    await self._best_effort_abort_context_artifact(
                        ctx_req.disaggregated_params,
                        server=ctx_server,
                        reason="generation stream was abandoned",
                    )
                    await self._settle_abandoned_generation_stream(
                        grant_params,
                        server=gen_server,
                    )
                    if renewal_error is not None:
                        raise renewal_error

                return _OwnedAsyncGenerator(
                    _yield_from_queue(),
                    _finish_queued_generation,
                )

            gen_task = asyncio.create_task(
                self._gen_client.send_request(
                    gen_req,
                    server=gen_server,
                    hooks=hooks,
                    req_id=disagg_request_id,
                )
            )
            gen_reservation_live = False
            if need_ctx:
                await asyncio.sleep(0)
                ctx_task = asyncio.create_task(
                    self._ctx_client.send_request(
                        ctx_req,
                        server=ctx_server,
                        hooks=hooks,
                        req_id=disagg_request_id,
                    )
                )
                ctx_reservation_live = False
                ctx_attempt_live = True
                ctx_response = await self._await_context_with_generation_supervision(
                    ctx_task,
                    gen_task,
                    renewal_task,
                )
                ctx_task = None
                await self._verify_ctx_response(ctx_response)
                self._verify_lifecycle_echo(
                    ctx_req.disaggregated_params,
                    ctx_response.choices[0].disaggregated_params,
                )
                if (
                    ctx_response.choices[0].disaggregated_params.disagg_request_id
                    != disagg_request_id
                ):
                    raise RuntimeError(
                        "generation-safe CTX response changed the admitted disaggregated request id"
                    )
                if not self._need_gen(ctx_response):
                    await self._abort_generation_grant(
                        grant_params,
                        server=gen_server,
                    )
                    grant_admitted = False
                    gen_task.cancel()
                    await asyncio.gather(gen_task, return_exceptions=True)
                    await self._stop_generation_grant_renewal(
                        renewal_task,
                        propagate_failure=False,
                    )
                    renewal_task = None
                    return ctx_response
            gen_response = await gen_task
            await self._stop_generation_grant_renewal(renewal_task)
            renewal_task = None
            return gen_response
        except asyncio.CancelledError:
            await _cleanup_failed_request()
            raise
        except Exception:
            await _cleanup_failed_request()
            raise

    async def _send_disagg_request_gen_first(
        self, request: UCompletionRequest, hooks: Optional[ResponseHooks] = None
    ) -> UCompletionResponse:
        if self._generation_safe_lifecycle_enabled:
            return await self._send_disagg_request_gen_first_v1(request, hooks)

        if hooks:
            hooks.on_req_begin(request)
        need_ctx = not (await self._check_gen_only_disagg(request))
        ctx_server, gen_server = None, None
        ctx_server_info = None
        ctx_req, gen_req = None, None
        # Single-issuer disagg id (see _send_disagg_request_ctx_first): fetch from
        # the coordinator so fleet workers never mint colliding ids.
        disagg_request_id = await self._coordinator.get_disagg_request_id()
        if hooks:
            hooks.on_disagg_request_id(disagg_request_id)
        lifecycle_identity = (
            self._new_lifecycle_identity(disagg_request_id)
            if self._generation_safe_lifecycle_enabled
            else None
        )
        if need_ctx:
            # arrival->here = pre-ctx wait in the orchestrator/fleet.
            if hooks:
                hooks.on_ctx_dispatch(request)
            ctx_req = self._get_ctx_request(
                request,
                disagg_request_id,
                lifecycle_identity=lifecycle_identity,
            )
            ctx_server, ctx_server_info = await self._ctx_router.get_next_server(
                ctx_req, req_id=disagg_request_id
            )
            if self._generation_safe_lifecycle_enabled:
                ctx_req.disaggregated_params = self._copy_disaggregated_params(
                    ctx_req.disaggregated_params,
                    context_control_endpoint=ctx_server,
                )
        gen_req = self._get_gen_request(
            request,
            ctx_response=None,
            disagg_request_id=disagg_request_id,
            ctx_server_info=ctx_server_info,
            lifecycle_identity=lifecycle_identity,
            context_control_endpoint=ctx_server,
        )
        if self._generation_safe_lifecycle_enabled:
            gen_req, gen_server, grant_decision = await self._select_and_admit_generation(
                gen_req,
                server=None,
                exclude_server=ctx_server,
                req_id=disagg_request_id,
            )
            if ctx_req is not None:
                gen_params = gen_req.disaggregated_params
                ctx_req.disaggregated_params = self._copy_disaggregated_params(
                    ctx_req.disaggregated_params,
                    generation_endpoint_name=gen_params.generation_endpoint_name,
                    generation_endpoint_rank=gen_params.generation_endpoint_rank,
                    generation_endpoint_incarnation=gen_params.generation_endpoint_incarnation,
                )
        renewal_task = (
            asyncio.create_task(
                self._renew_generation_grant(
                    gen_req.disaggregated_params,
                    server=gen_server,
                    initial_ttl_s=grant_decision.ttl_s,
                )
            )
            if self._generation_safe_lifecycle_enabled
            else None
        )

        if request.stream and need_ctx:
            # For streaming gen_first requests, the gen client returns a lazy
            # async generator whose HTTP POST only fires when iterated. The ctx
            # server blocks waiting for the gen server's rx session (gen_first
            # protocol). Using asyncio.gather would deadlock: ctx waits for gen
            # server, but gen POST is deferred until the generator is consumed,
            # and the generator isn't consumed until gather returns.
            #
            # Fix: eagerly start consuming the gen generator in a background
            # task so the HTTP POST fires, then pipe chunks through a queue.
            gen_response = await self._gen_client.send_request(
                gen_req, server=gen_server, hooks=hooks, req_id=disagg_request_id
            )

            queue: asyncio.Queue = asyncio.Queue()

            async def _consume_gen():
                try:
                    async for chunk in gen_response:
                        await queue.put(chunk)
                except Exception as e:
                    await queue.put(e)
                await queue.put(None)  # sentinel

            consume_task: asyncio.Task = asyncio.create_task(_consume_gen())

            # Now send ctx request — gen server has received its request
            try:
                ctx_response = await self._ctx_client.send_request(
                    ctx_req,
                    server=ctx_server,
                    hooks=hooks,
                    req_id=disagg_request_id,
                )
                await self._verify_ctx_response(ctx_response)
                self._verify_lifecycle_echo(
                    ctx_req.disaggregated_params,
                    ctx_response.choices[0].disaggregated_params,
                )
            except Exception:
                consume_task.cancel()
                await asyncio.gather(consume_task, return_exceptions=True)
                if renewal_task is not None:
                    renewal_task.cancel()
                    await asyncio.gather(renewal_task, return_exceptions=True)
                raise

            async def _yield_from_queue():
                try:
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        yield item
                finally:
                    if not consume_task.done():
                        consume_task.cancel()
                    try:
                        await consume_task
                    except asyncio.CancelledError:
                        pass
                    await self._stop_generation_grant_renewal(renewal_task)

            return _yield_from_queue()
        elif request.stream:
            gen_response = await self._gen_client.send_request(
                gen_req,
                server=gen_server,
                hooks=hooks,
                req_id=disagg_request_id,
            )

            async def _yield_generation_only():
                try:
                    async for chunk in gen_response:
                        yield chunk
                finally:
                    await self._stop_generation_grant_renewal(renewal_task)

            return _yield_generation_only()
        else:
            # Non-streaming or no ctx needed: both HTTP POSTs fire eagerly
            # through generator consumption, so asyncio.gather works fine.
            tasks = []
            if need_ctx:
                tasks.append(
                    asyncio.create_task(
                        self._ctx_client.send_request(
                            ctx_req,
                            server=ctx_server,
                            hooks=hooks,
                            req_id=disagg_request_id,
                        )
                    )
                )
            tasks.append(
                asyncio.create_task(
                    self._gen_client.send_request(
                        gen_req,
                        server=gen_server,
                        hooks=hooks,
                        req_id=disagg_request_id,
                    )
                )
            )
            try:
                responses = await asyncio.gather(*tasks)
                if need_ctx:
                    ctx_response = responses[0]
                    await self._verify_ctx_response(ctx_response)
                    self._verify_lifecycle_echo(
                        ctx_req.disaggregated_params,
                        ctx_response.choices[0].disaggregated_params,
                    )
                return responses[-1]
            finally:
                await self._stop_generation_grant_renewal(renewal_task)
