# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unary encoding adapter; native output storage owns tensor transfer and lifetime."""

import asyncio
from functools import partial

import grpc

from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams

from .bindings import encoding_pb2, generation_pb2
from .coordinator import CoordinationError, DuplicateRequestError, InvalidRequestIdError
from .encoder import encoder_input, portable_reference, release_encoder_result
from .errors import MediaUnavailableError, UnsupportedFeatureError
from .metrics import RequestMetrics
from .request_mapping import (
    _load_prepared_input,
    _metadata_from_context,
    _prepare_input_from_request,
    _validate_media_placeholders,
)
from .servicer import (
    OpenEngineInferenceServicer,
    _LocalRequest,
    _PreparationLease,
    _PreprocessingWork,
)
from .streaming import ActiveRequest


async def _prepare(owner, request, sampling, local_request):
    prepared = _prepare_input_from_request(request, owner._llm, owner._media_capabilities)
    await owner._media_admission.acquire()
    lease = _PreparationLease(owner._media_admission)
    try:
        expands_ids = getattr(owner._llm.input_processor, "supports_token_id_mm_expansion", False)
        ids = await lease.run(
            owner._input_executor,
            partial(
                _validate_media_placeholders,
                add_special_tokens=sampling.add_special_tokens and expands_ids,
            ),
            prepared,
            owner._llm,
        )
        with BaseMediaIO.use_executor(owner._media_executor, on_submit=lease.track):
            inputs = await _load_prepared_input(prepared, owner._llm)
        if ids is not None and isinstance(prepared.inputs, str) and expands_ids:
            inputs.pop("prompt")
            inputs["prompt_token_ids"] = ids
        work = local_request.preprocessing = _PreprocessingWork()
        return await lease.run(
            owner._input_executor, work.run, partial(owner._llm.preprocess, inputs, sampling, None)
        )
    finally:
        lease.close()


async def encode(
    owner: OpenEngineInferenceServicer,
    request: encoding_pb2.EncodeRequest,
    context: grpc.aio.ServicerContext,
) -> encoding_pb2.EncodeResponse:
    """Run native encoding and return one framework-owned transferable reference."""
    local_request = _LocalRequest()
    registration = ActiveRequest(owner._active_requests, request.request_id, local_request)
    reservation = None
    handle = None
    completed = False
    metrics = RequestMetrics(owner._metrics, phase="encode") if owner._metrics else None
    try:
        if not request.request_id or not request.model:
            raise ValueError("request_id and model must be non-empty")
        if owner.encoder_role != "encode":
            raise UnsupportedFeatureError("loaded engine is not an encoder-only worker")
        if owner._encoder_cleanup_failed:
            await context.abort(
                grpc.StatusCode.UNAVAILABLE, "encoder cleanup failed; restart the engine"
            )
        if not registration.register():
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, "request_id is already active")
        if owner._frontend is not None:
            reservation = await owner._frontend.reserve(request.request_id)
            reservation.bind(local_request)
        if not request.media or any(
            item.modality != generation_pb2.MODALITY_IMAGE for item in request.media
        ):
            raise UnsupportedFeatureError("native encoder handoffs currently support images only")
        native_input = encoder_input(request)
        _, target_dp_rank, affinity_headers = _metadata_from_context(context)
        if target_dp_rank is not None or affinity_headers:
            raise UnsupportedFeatureError("encoder workers do not support DP affinity metadata")
        sampling = SamplingParams(max_tokens=1)
        task = asyncio.create_task(_prepare(owner, request, sampling, local_request))
        local_request.bind_mapping(task)
        try:
            await asyncio.wait((task,))
            if task.cancelled() and local_request.aborted:
                await context.abort(grpc.StatusCode.CANCELLED, "Encode aborted during preparation")
            inputs = task.result()
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            local_request.clear_mapping(task)
        if local_request.aborted or (reservation is not None and reservation.aborted):
            await context.abort(grpc.StatusCode.CANCELLED, "Encode aborted before submission")
        handle = owner._llm.generate_async(inputs=inputs, sampling_params=sampling)
        if metrics is not None:
            metrics.accepted()
        local_request.bind_engine(handle)
        final = None
        async for result in handle:
            if result.error:
                raise RuntimeError(str(result.error))
            if result.finished:
                final = result
        if final is None:
            raise RuntimeError("native encoder ended without a completed output")
        completed = True
        if (
            context.cancelled()
            or local_request.aborted
            or (reservation is not None and reservation.aborted)
        ):
            await context.abort(grpc.StatusCode.CANCELLED, "Encode aborted before publication")
        await owner._media_admission.acquire()
        lease = _PreparationLease(owner._media_admission)
        publication = _PreprocessingWork(partial(_discard_publication, owner))
        try:
            reference = await lease.run(
                owner._input_executor,
                publication.run,
                partial(owner._encoder_cache.publish, final, native_input, owner._llm),
            )
            if (
                context.cancelled()
                or local_request.aborted
                or (reservation is not None and reservation.aborted)
            ):
                await context.abort(grpc.StatusCode.CANCELLED, "Encode aborted before publication")
            response = encoding_pb2.EncodeResponse(
                request_id=request.request_id, encoded_media=portable_reference(reference)
            )
            publication.transfer()
            if metrics is not None:
                metrics.complete_encoding()
            return response
        finally:
            publication.abandon()
            lease.close()
    except (DuplicateRequestError, InvalidRequestIdError, CoordinationError) as error:
        status = (
            grpc.StatusCode.UNAVAILABLE
            if isinstance(error, CoordinationError)
            else grpc.StatusCode.INVALID_ARGUMENT
        )
        if isinstance(error, DuplicateRequestError):
            status = grpc.StatusCode.ALREADY_EXISTS
        await context.abort(status, str(error))
    except UnsupportedFeatureError as error:
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(error))
    except MediaUnavailableError as error:
        await context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
    except ValueError as error:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
    except RuntimeError as error:
        if metrics is not None:
            metrics.engine_failed()
        await context.abort(grpc.StatusCode.INTERNAL, str(error))
    except asyncio.CancelledError as error:
        if metrics is not None:
            metrics.observe_exception(error)
        raise
    except Exception as error:
        # Preserve the gRPC status selected by context.abort.
        if context.code() not in (None, grpc.StatusCode.OK):
            raise
        if metrics is not None:
            metrics.engine_failed()
        await context.abort(grpc.StatusCode.INTERNAL, str(error))
    finally:
        if local_request.preprocessing is not None:
            local_request.preprocessing.abandon()
        try:
            if handle is not None:
                await _reclaim(owner, handle, completed)
        finally:
            registration.release()
            try:
                if reservation is not None:
                    await asyncio.shield(reservation.release())
            finally:
                if metrics is not None:
                    metrics.finish(context.code())


async def _reclaim(owner, handle, completed):
    if not completed:
        try:
            handle.abort()
        except Exception:
            owner._encoder_cleanup_failed = True

    async def reclaim():
        try:
            if not completed:
                await asyncio.wait_for(handle.aresult(), timeout=30)
        except asyncio.TimeoutError:
            owner._encoder_cleanup_failed = True
        except Exception:
            pass
        finally:
            if not await asyncio.to_thread(release_encoder_result, handle):
                owner._encoder_cleanup_failed = True

    await owner._run_encoder_cleanup(reclaim())


def _discard_publication(owner, reference):
    if reference is None:
        return
    try:
        owner._encoder_cache.discard(reference)
    except Exception as error:
        owner._encoder_cleanup_failed = True
        logger.error(f"Failed to reclaim abandoned encoder output: {error}")
