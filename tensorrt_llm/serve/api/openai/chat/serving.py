# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/chat/completions``: chat, and gpt-oss through ``chat_harmony``."""

import asyncio
import base64
import functools
import json
import os
import signal
import traceback
from http import HTTPStatus
from typing import Any, AsyncIterator, List, Optional, Union

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.inputs import prompt_inputs
from tensorrt_llm.inputs.utils import ConversationMessage
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi import tracing
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.llmapi.thinking_budget import \
    add_thinking_budget_logits_processor
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams
from tensorrt_llm.serve._base.errors import (EngineFatalError, ServingError,
                                             serving_error)
from tensorrt_llm.serve._base.request_context import RequestContext
from tensorrt_llm.serve._base.serving import (
    GenerateBaseServing, _configure_parser_special_token_decoding,
    _record_generator_termination)
from tensorrt_llm.serve._base.sse import chat_stream_generator
from tensorrt_llm.serve.chat_tokenization import tokenize_harmony_chat_request
from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines
from tensorrt_llm.serve.conversation_id import resolve_request_conversation_id
from tensorrt_llm.serve.harmony_adapter import (HarmonyAdapter,
                                                get_harmony_adapter)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionNamedToolChoiceParam, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionToolsParam, ResponseFormat,
    ensure_request_chat_template_allowed, to_llm_conversation_params,
    to_llm_disaggregated_params)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatCompletionPostprocArgs, ChatPostprocArgs, chat_harmony_post_processor,
    chat_harmony_streaming_post_processor, chat_response_post_processor,
    chat_stream_post_processor)
from tensorrt_llm.serve.render import (RenderResources, prepare_chat_request,
                                       render_conversation)
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory


class ChatServing(GenerateBaseServing):
    """``POST /v1/chat/completions``: chat, and gpt-oss through ``chat_harmony``.

    No harmony subclass: ``chat_harmony`` is a method here until the temporary
    switches are removed, chosen by the router exactly as the server chose it
    before this class existed. The class is also the chat backend the Anthropic
    Messages serving class runs on (``stream_model`` and ``error_message``).
    """

    def __init__(self, *, tool_parser: Optional[str], tool_call_id_type: str,
                 multimodal_server_config: Any,
                 allow_request_chat_template: bool, use_harmony: bool,
                 harmony_adapter: Optional[HarmonyAdapter],
                 render_resources: RenderResources, **base_kwargs: Any) -> None:
        super().__init__(**base_kwargs)
        self.tool_parser = tool_parser
        self.tool_call_id_type = tool_call_id_type
        self.multimodal_server_config = multimodal_server_config
        self.allow_request_chat_template = allow_request_chat_template
        self.use_harmony = use_harmony
        # The gpt-oss adapter, resolved lazily by chat_harmony on first use,
        # as the server did.
        self.harmony_adapter = harmony_adapter
        # Built once by the composition root; never RenderResources.from_server
        # on this object, which would drop the server chat template.
        self._render_resources = render_resources

    async def create_chat_completion(
        self, request: ChatCompletionRequest, ctx: RequestContext
    ) -> Union[ChatCompletionResponse, AsyncIterator[str]]:
        """Run one chat completion and return a typed result.

        Returns the ChatCompletionResponse for a non-streaming request, and
        for a streaming one the SSE stream as strings, ending with
        "data: [DONE]".

        Raises:
            ServingError: The request is rejected; the HTTP status travels
                apart from the body.
            EngineFatalError: The engine failed; SIGINT has been sent.
        """

        def get_role() -> str:
            if request.add_generation_prompt:
                role = "assistant"
            else:
                role = request.messages[-1]["role"]
            return role

        try:
            # Extension preprocessing, the required-tool check and tool dumping
            # are the first stage of the shared prompt-preparation pipeline.
            render_resources = self._render_resources
            extension = render_resources.extension
            tool_dicts = prepare_chat_request(request,
                                              render_resources).tool_dicts
            required_mode = extension.required_tool_choice_mode()
            required_tool_grammar = (request.tool_choice == "required"
                                     and required_mode == "grammar")
            conversation: List[ConversationMessage] = []
            # Pass the model vocabulary size so ``logit_bias`` can be
            # expanded into an embedding bias tensor in the sampler.
            vocab_size = getattr(self.tokenizer.tokenizer,
                                 "vocab_size", None) or getattr(
                                     self.tokenizer, "vocab_size", None)
            sampling_params = request.to_sampling_params(
                vocab_size=self._logit_bias_vocab_size(),
                gather_generation_logits=self.generator.args.
                gather_generation_logits,
                reasoning_parser=self.generator.args.reasoning_parser,
                backend=self.generator.args.backend)
            if required_tool_grammar:
                _validate_required_tool_grammar_request(
                    request,
                    sampling_params,
                    guided_decoding_backend=self.generator.args.
                    guided_decoding_backend,
                    tool_parser=self.tool_parser,
                    reasoning_parser=self.generator.args.reasoning_parser)
            # Pre-render, so the mode may be unresolved. Safe because the
            # boundary lookup reads the parser class attributes, not the
            # branch `__init__` picked.
            add_thinking_budget_logits_processor(
                sampling_params,
                reasoning_parser=self.generator.args.reasoning_parser,
                tokenizer=self.tokenizer,
                chat_template_kwargs=request.chat_template_kwargs,
            )
            # Resolve a named tool_choice (OpenAI spec / Azure compat):
            #   tool_choice = {"type": "function", "function": {"name": "X"}}
            # forces the model to call exactly function X. We honor this by
            # (a) prefix-injecting the tool parser's ``begin`` string into the
            # rendered chat prompt so the model starts generation already inside
            # a tool call, and (b) constraining the generated arguments with
            # plain JSON-schema guided decoding. The post-processor then
            # synthesizes the resulting ``tool_calls`` entry. Unlike the
            # ``triggered_tags`` structural-tag path used for ``strict`` tools,
            # this path is mandatory by construction — the model cannot escape
            # the tool call.
            forced_tool_name = None
            forced_tool_begin_prefix = None
            if isinstance(request.tool_choice,
                          ChatCompletionNamedToolChoiceParam):
                if not request.tools:
                    raise serving_error(
                        message=("tool_choice with a named function requires "
                                 "`tools` to be set."),
                        err_type="BadRequestError",
                        status_code=HTTPStatus.BAD_REQUEST)
                if (request.prompt_token_ids is not None
                        or request.prompt_token_ids_b64):
                    # Tokenizing and concatenating the ``begin_prefix`` onto a
                    # pre-tokenized prompt correctly across all tokenizers is
                    # fragile (token-boundary issues with special tokens).
                    # Reject the combination explicitly until there is a
                    # demonstrated use case.
                    #
                    # ``prompt_token_ids_b64`` must be checked here too: it is
                    # only decoded into ``prompt_token_ids`` further below, so
                    # testing ``prompt_token_ids`` alone lets a base64 request
                    # through to the pre-tokenized branch, where the begin
                    # prefix is never appended.
                    raise serving_error(
                        message=("tool_choice with a named function is not "
                                 "supported alongside `prompt_token_ids`; pass "
                                 "`messages` instead."),
                        err_type="BadRequestError",
                        status_code=HTTPStatus.BAD_REQUEST)
                if sampling_params.guided_decoding is not None:
                    # A named tool_choice is mandatory by construction: the
                    # begin prefix plus a JSON-schema constraint force the
                    # model into exactly this call. That cannot coexist with a
                    # client-supplied guided-decoding mode (``response_format``
                    # and friends), which constrains the assistant *content*
                    # instead. Honouring only one of them silently would either
                    # ignore an explicit instruction or emit a tool call whose
                    # arguments are unconstrained text, so reject instead.
                    raise serving_error(
                        message=("tool_choice with a named function cannot be "
                                 "combined with `response_format` or other "
                                 "guided-decoding settings."),
                        err_type="BadRequestError",
                        status_code=HTTPStatus.BAD_REQUEST)
                if (self.generator.args.guided_decoding_backend is None
                        and not _parser_extracts_forced_tool_calls(
                            self.tool_parser)):
                    # Parsers that extract forced calls from their own markup
                    # need no grammar, so this only applies to the constrained
                    # path below.
                    #
                    # The forced path is only mandatory because a JSON-schema
                    # grammar constrains what follows the begin prefix. With no
                    # guided-decoding backend configured that grammar is never
                    # built and the per-request params are silently dropped, so
                    # the prefix alone would be left "forcing" the call: the
                    # model completes the arguments, closes the tool-call
                    # object itself and keeps generating, and the whole tail
                    # ends up reported as ``function.arguments``. Fail loudly
                    # instead of returning a tool call whose arguments are
                    # unconstrained text.
                    raise serving_error(message=(
                        "tool_choice with a named function requires the "
                        "server to be started with a guided decoding "
                        "backend (set `guided_decoding_backend` to "
                        "'xgrammar' or 'llguidance')."),
                                        err_type="BadRequestError",
                                        status_code=HTTPStatus.BAD_REQUEST)
                forced_tool_name = request.tool_choice.function.name

            reasoning_parser_name = self.generator.args.reasoning_parser
            # Message-level (dynamic) tools are a model extension (Kimi API);
            # only models whose extension declares them render them, so other
            # models keep ignoring the key entirely.
            dynamic_tools = extension.dynamic_tools(request.messages)
            dynamic_tool_params: List[ChatCompletionToolsParam] = []
            if dynamic_tools:
                try:
                    dynamic_tool_params = [
                        ChatCompletionToolsParam.model_validate(tool)
                        for tool in dynamic_tools
                    ]
                except ValidationError as e:
                    raise ValueError(
                        f"Invalid message-level tool declaration: {e}") from e
            _configure_parser_special_token_decoding(
                sampling_params,
                reasoning_parser_name=reasoning_parser_name,
                tool_parser_name=self.tool_parser,
                has_tools=bool(request.tools) or bool(dynamic_tools))
            all_tools = (request.tools or []) + dynamic_tool_params
            if self.tool_parser and all_tools:
                # When strict=True on any tool, apply constrained decoding
                # via structural tags (only if response_format doesn't already
                # set guided decoding). Dynamic tools must be in the grammar
                # too, or their calls would be masked out.
                if sampling_params.guided_decoding is None and not required_tool_grammar:
                    if (forced_tool_name is not None
                            and _parser_extracts_forced_tool_calls(
                                self.tool_parser)):
                        # This parser handles forced calls by extracting them
                        # from its own native markup (see
                        # ``BaseToolParser.extracts_forced_tool_calls``), so
                        # there is no grammar to build and none is needed. Leave
                        # the prompt and sampling params alone; the
                        # post-processor runs the parser and takes the call from
                        # there. Falling through to the grammar path below would
                        # reject these parsers outright, because none of them
                        # supports structural tags.
                        pass
                    elif forced_tool_name is not None:
                        # Forced-call path: build the begin-prefix + JSON-schema
                        # guided decoding. The prefix is injected into the
                        # prompt after ``apply_chat_template`` below.
                        try:
                            forced_tool_begin_prefix, forced_guided = (
                                _build_forced_tool_call_decoding(
                                    request.tools, self.tool_parser,
                                    forced_tool_name))
                        except ValueError as e:
                            raise serving_error(
                                message=str(e),
                                err_type="BadRequestError",
                                status_code=HTTPStatus.BAD_REQUEST)
                        sampling_params.guided_decoding = forced_guided
                    else:
                        # Strict-tools path: structural tags with the existing
                        # ``triggered_tags`` semantics. Engages only when at
                        # least one tool has ``strict=True``.
                        strict_guided = _build_tool_strict_guided_decoding_params(
                            all_tools, self.tool_parser)
                        if strict_guided is not None:
                            sampling_params.guided_decoding = strict_guided
            elif forced_tool_name is not None:
                # tools were provided but the server has no tool_parser configured.
                raise serving_error(
                    message=("tool_choice with a named function requires the "
                             "server to be started with --tool_parser."),
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST)
            postproc_args = ChatPostprocArgs.from_request(request)
            self._apply_spec_decode_stats_opt_in(postproc_args)
            # Some models' prompt-token accounting excludes trailing rendered
            # tokens (kimi_k3: the generation channel opener); the model still
            # sees the full prompt.
            postproc_args.num_prompt_tokens_offset = (
                render_resources.prompt_tokens_excluded_from_usage(
                    add_generation_prompt=request.add_generation_prompt,
                    pre_tokenized=request.prompt_token_ids is not None
                    or request.prompt_token_ids_b64 is not None,
                    request_chat_template=request.chat_template,
                ))
            if dynamic_tool_params:
                # The tool parser must see dynamic tools to recognize their
                # calls in the model output.
                postproc_args.tools = (postproc_args.tools
                                       or []) + dynamic_tool_params
            self._validate_internal_disagg_request(request, ctx)
            disaggregated_params = to_llm_disaggregated_params(
                request.disaggregated_params)

            # A generation-only worker already has prompt_token_ids with the
            # placeholders expanded and the KV behind them over the
            # transceiver, so the media still on the relayed messages is not
            # its to resolve.
            #
            # Decided before the fetch rather than after it: resolving
            # downloads and decodes every item a second time, and fails
            # outright on a reference only the context worker could read --
            # a node-local path, or a single-use or expired URL.
            #
            # prompt_token_ids_b64 counts too; it is decoded into
            # prompt_token_ids further below.
            resolve_media = not (disaggregated_params is not None
                                 and disaggregated_params.request_type
                                 == "generation_only" and
                                 (request.prompt_token_ids is not None
                                  or request.prompt_token_ids_b64))

            try:
                conversation, mm_coroutines, mm_placeholder_counts, mm_item_order = parse_chat_messages_coroutines(
                    request.messages,
                    self.model_config,
                    self.multimodal_server_config,
                    request_media_io_kwargs=request.media_io_kwargs,
                    resolve_media=resolve_media,
                )
            except ValidationError:
                # ValidatorIterator rejects extra fields; fall back to raw JSON.
                raw_body = await ctx.json()
                raw_messages = raw_body.get("messages", [])
                conversation, mm_coroutines, mm_placeholder_counts, mm_item_order = parse_chat_messages_coroutines(
                    raw_messages,
                    self.model_config,
                    self.multimodal_server_config,
                    request_media_io_kwargs=request.media_io_kwargs,
                    resolve_media=resolve_media,
                )

            # Decode base64 int32 prompt_token_ids relayed by the orchestrator.
            if request.prompt_token_ids is None and request.prompt_token_ids_b64:
                import numpy as np
                request.prompt_token_ids = np.frombuffer(
                    base64.b64decode(request.prompt_token_ids_b64),
                    dtype=np.int32).tolist()

            rendered_prompt = None
            if request.prompt_token_ids is not None:
                prompt = request.prompt_token_ids
            else:
                # Render (and append the forced named-tool prefix, see
                # ``_build_forced_tool_call_decoding``) while the media loads.
                # Tokenization stays with the engine's input processor.
                render_task = render_conversation(
                    render_resources,
                    conversation=conversation,
                    mm_placeholder_counts=mm_placeholder_counts,
                    add_generation_prompt=request.add_generation_prompt,
                    tools=tool_dicts,
                    documents=request.documents,
                    chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs,
                    injected_chat_template_kwargs=request.
                    injected_chat_template_kwargs,
                    forced_prefix=forced_tool_begin_prefix,
                )
                rendered, (mm_data, mm_embeddings) = await asyncio.gather(
                    render_task, mm_coroutines)
                if rendered.text is None:
                    prompt = rendered.token_ids
                else:
                    prompt = rendered.text
                    # Captured before the forced-tool prefix is appended:
                    # resolve_prefilled_thinking() inspects what the chat
                    # template rendered, and the prefix is not part of it.
                    rendered_prompt = rendered.template_text
            prompt = prompt_inputs(prompt)

            if request.prompt_token_ids is not None:
                mm_data, mm_embeddings = await mm_coroutines
            if mm_data:
                self._attach_multimodal_data(prompt, request, mm_data,
                                             mm_item_order)
            if mm_embeddings:
                prompt["multi_modal_embeddings"] = mm_embeddings
            if mm_data and mm_embeddings:
                raise ValueError(
                    "Passing 'multi_modal_data' and 'multi_modal_embeddings' at the same time is not supported."
                )

            if request.mm_processor_kwargs:
                prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

            thinking = None
            postproc_args.reasoning_parser = self.generator.args.reasoning_parser
            # Templates that prefill <think>/</think> leave the marker in the
            # prompt, so the request kwargs alone cannot tell the parser which
            # mode was rendered. Take it from the prompt instead.
            if postproc_args.reasoning_parser and (
                    ReasoningParserFactory.resolves_thinking_from_prompt(
                        postproc_args.reasoning_parser)):
                if rendered_prompt and request.add_generation_prompt:
                    thinking = ReasoningParserFactory.resolve_prefilled_thinking(
                        postproc_args.reasoning_parser, rendered_prompt)
                if thinking is None and request.disaggregated_params is not None:
                    # Generation worker: it never rendered, so use the mode the
                    # context worker resolved and relayed.
                    thinking = request.disaggregated_params.resolved_thinking
                    if thinking is None:
                        _warn_unresolvable_thinking_once(
                            postproc_args.reasoning_parser)
                if thinking is not None:
                    # Both keys, because the parser ORs them: leaving a stale
                    # `thinking` in place would override what we resolved.
                    postproc_args.chat_template_kwargs = {
                        **(request.chat_template_kwargs or {}),
                        "thinking": thinking,
                        "enable_thinking": thinking,
                    }
            if required_tool_grammar:
                if thinking is None:
                    raise ValueError(
                        "Cannot resolve required-tool reasoning state from the rendered prompt."
                    )
                parser = ToolParserFactory.create_tool_parser(self.tool_parser)
                build_required = functools.partial(
                    parser.build_required_structural_tag,
                    request.tools,
                    thinking=thinking)
                required_format = await asyncio.get_running_loop(
                ).run_in_executor(self._input_proc_executor, build_required)
                if required_format is None:
                    raise ValueError(
                        "The configured tool parser does not support required tools."
                    )
                sampling_params.guided_decoding = GuidedDecodingParams(
                    structural_tag=required_format.model_dump_json(
                        by_alias=True))
                postproc_args.tool_parser_grammar_constrained = True
            postproc_args.tool_parser = self.tool_parser
            postproc_args.tool_call_id_type = self.tool_call_id_type
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == get_role():
                postproc_args.last_message_content = conversation[-1]["content"]
            postproc_params = PostprocParams(
                post_processor=chat_stream_post_processor
                if request.stream else chat_response_post_processor,
                postproc_args=postproc_args,
            )

            trace_headers = (None if ctx is None else
                             tracing.extract_trace_headers(ctx.headers))

            resolve_request_conversation_id(
                request, None if ctx is None else ctx.headers)
            conversation_params = to_llm_conversation_params(
                request.conversation_params)
            scheduling_params = self._get_scheduling_params(request, ctx)

            generate_inputs = prompt
            preprocess_fn = getattr(self.generator, "preprocess", None)
            if preprocess_fn is not None:
                loop = asyncio.get_event_loop()
                generate_inputs = await loop.run_in_executor(
                    self._input_proc_executor,
                    functools.partial(preprocess_fn, prompt, sampling_params,
                                      disaggregated_params))

            promise = self.generator.generate_async(
                inputs=generate_inputs,
                sampling_params=sampling_params,
                _postproc_params=postproc_params
                if self.postproc_worker_enabled else None,
                streaming=request.stream,
                lora_request=request.lora_request,
                disaggregated_params=disaggregated_params,
                conversation_params=conversation_params,
                cache_salt=request.cache_salt,
                kv_hints=request.kv_hints,
                trace_headers=trace_headers,
                scheduling_params=scheduling_params,
                priority=request.priority
                if request.priority is not None else DEFAULT_REQUEST_PRIORITY,
            )
            asyncio.create_task(self.await_disconnected(ctx, promise))
            if not self.postproc_worker_enabled:
                postproc_args.tokenizer = self.tokenizer
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            if request.stream:
                return chat_stream_generator(promise,
                                             postproc_params,
                                             server=self,
                                             ctx=ctx)
            else:
                response = await self._create_chat_response(
                    promise, postproc_params, ctx, disaggregated_params)
                # Context-only: optionally return prompt_token_ids as a base64
                # int32 buffer (one string) so the disagg orchestrator relays it
                # without materializing the int list on its event loop. The encode
                # runs here on the context worker (1-of-N), not the orchestrator.
                if (request.disaggregated_params is not None and
                        request.disaggregated_params.return_prompt_token_ids_b64
                        and response.prompt_token_ids is not None):
                    import numpy as np
                    response.prompt_token_ids_b64 = base64.b64encode(
                        np.asarray(response.prompt_token_ids,
                                   dtype=np.int32).tobytes()).decode("ascii")
                    response.prompt_token_ids = None
                return response
        except ServingError:
            raise
        except CppExecutorError as e:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            _record_generator_termination(self.generator)
            signal.raise_signal(signal.SIGINT)
            raise EngineFatalError() from e
        except ValueError as e:
            raise serving_error(str(e)) from e
        except Exception as e:
            logger.error(traceback.format_exc())
            raise serving_error(str(e)) from e

    async def openai_chat(self, request: ChatCompletionRequest,
                          raw_request: Request) -> Response:
        try:
            result = await self.create_chat_completion(
                request, RequestContext.from_http(raw_request))
            if isinstance(result, ChatCompletionResponse):
                return JSONResponse(content=result.model_dump())
            return StreamingResponse(content=result,
                                     media_type="text/event-stream")
        except ServingError as e:
            return JSONResponse(content=e.body.model_dump(),
                                status_code=e.status)
        except EngineFatalError:
            # Answered as HTTP 200 with body null (FastAPI's rendering of
            # None); the in-process callers map that to their own errors.
            # Answering 5xx would be a behavior change of its own.
            return None
        except ValueError as e:
            # JSONResponse encodes in its constructor (allow_nan=False), so a
            # response that cannot be encoded fails here and keeps its 400.
            return self.create_error_response(str(e))
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))

    async def _create_chat_response(
        self,
        promise: RequestOutput,
        postproc_params: PostprocParams,
        raw_request: Request,
        disaggregated_params: Optional[LlmDisaggregatedParams] = None
    ) -> ChatCompletionResponse:
        await promise.aresult()
        if promise.error is not None:
            raise RuntimeError(f"Generation failed: {promise.error}")
        chat_response = self._postproc.apply(promise, postproc_params)

        if disaggregated_params is not None and disaggregated_params.request_type == "context_only":
            chat_response.prompt_token_ids = promise.prompt_token_ids

        if disaggregated_params is not None and chat_response.choices[
                0].disaggregated_params is None:
            raise ValueError(
                f"disaggregated_params is not set in the response for request"
                f" {disaggregated_params.disagg_request_id}")

        await self._extract_metrics(promise, raw_request)
        return chat_response

    async def chat_harmony(self, request: ChatCompletionRequest,
                           raw_request: Request) -> Response:
        """Chat Completion API with harmony format support.

        Supports both streaming and non-streaming modes.
        """

        async def create_streaming_generator(promise: RequestOutput,
                                             postproc_params: PostprocParams):
            try:
                first_response = await anext(promise)
                raw_request.state.server_first_token_time = (
                    self._adjusted_steady_clock.now())
                pp_results = self._postproc.apply(first_response,
                                                  postproc_params)
                for pp_res in pp_results:
                    yield pp_res
                res = first_response
                async for res in promise:
                    pp_results = self._postproc.apply(res, postproc_params)
                    for pp_res in pp_results:
                        yield pp_res
                yield "data: [DONE]\n\n"
                await self._extract_metrics(res, raw_request)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(traceback.format_exc())
                raise

        try:
            ensure_request_chat_template_allowed(
                request, self.allow_request_chat_template)
            # Named tool_choice (forced function call) is not yet supported on
            # the harmony / GPT-OSS path; return a clear 4xx pointing to the
            # follow-up sub-task rather than silently falling back to "auto".
            # See TRTLLM-12758 (and its harmony follow-up).
            if isinstance(request.tool_choice,
                          ChatCompletionNamedToolChoiceParam):
                return self.create_error_response(
                    message=("tool_choice with a named function is not yet "
                             "supported for harmony / GPT-OSS models. Tracking "
                             "follow-up: harmony sub-task of TRTLLM-12758."),
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST)
            if request.tool_choice == "required":
                # The harmony path treats unknown tool_choice values like
                # "auto"; reject instead of silently degrading.
                raise ValueError(
                    "tool_choice='required' is not supported for this model.")
            # Initialize HarmonyAdapter
            # NOTE: WAR for Disagg failure, may affect perf if no warmup
            if not self.harmony_adapter:
                self.harmony_adapter = get_harmony_adapter()

            # Reuse pre-tokenized harmony tokens when forwarded by an upstream
            # context worker (disaggregated serving). Otherwise, run the
            # Harmony adapter on the request messages.
            try:
                harmony_tokens = tokenize_harmony_chat_request(
                    request, harmony_adapter=self.harmony_adapter)
            except Exception:
                logger.error("messages_dict: %s", request.messages)
                logger.error("tools: %s", request.tools)
                logger.error("request: %s", request)
                raise

            # Get harmony stop tokens
            harmony_stop_tokens = self.harmony_adapter.get_stop_tokens()
            if request.stop_token_ids:
                request.stop_token_ids.extend(harmony_stop_tokens)
            else:
                request.stop_token_ids = harmony_stop_tokens

            sampling_params = request.to_sampling_params(
                vocab_size=self._logit_bias_vocab_size(),
                reasoning_parser="gpt_oss")
            if sampling_params.thinking_token_budget is not None:
                raise ValueError(
                    "thinking_token_budget is not supported by the Harmony "
                    "GPT-OSS serving path; use reasoning_effort instead")
            sampling_params.detokenize = False  # Harmony adapter handles detokenization
            # The server-level effective flag already enables engine metrics
            # for header or JSONL output.
            if len(os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "")) > 0:
                sampling_params.return_perf_metrics = True
            self._validate_internal_disagg_request(request, raw_request)
            disaggregated_params = to_llm_disaggregated_params(
                request.disaggregated_params)
            trace_headers = (None if raw_request is None else
                             tracing.extract_trace_headers(raw_request.headers))

            postproc_args = ChatCompletionPostprocArgs.from_request(request)
            # No spec-decode opt-in here on purpose. The Harmony handlers build
            # their choices in harmony_adapter, which carries no per-request
            # spec-decode data at all -- avg_decoded_tokens_per_iter is absent
            # from that path too -- so setting the flag would configure
            # something nothing reads. Extending Harmony should cover both
            # fields together; handle_non_streaming_response would need the
            # GenerationResult threaded through, as it currently receives only
            # the outputs.
            postproc_params = PostprocParams(
                post_processor=chat_harmony_streaming_post_processor
                if request.stream else chat_harmony_post_processor,
                postproc_args=postproc_args,
            )

            resolve_request_conversation_id(
                request, None if raw_request is None else raw_request.headers)
            conversation_params = to_llm_conversation_params(
                request.conversation_params)
            scheduling_params = self._get_scheduling_params(
                request, raw_request)

            # Generate
            promise = self.generator.generate_async(
                inputs=harmony_tokens,
                sampling_params=sampling_params,
                _postproc_params=postproc_params
                if self.postproc_worker_enabled else None,
                streaming=bool(request.stream),
                lora_request=request.lora_request,
                scheduling_params=scheduling_params,
                disaggregated_params=disaggregated_params,
                conversation_params=conversation_params,
                trace_headers=trace_headers,
                kv_hints=request.kv_hints,
                priority=request.priority
                if request.priority is not None else DEFAULT_REQUEST_PRIORITY,
            )
            if not self.postproc_worker_enabled:
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            # Disconnect cancellation
            asyncio.create_task(self.await_disconnected(raw_request, promise))

            # Handle streaming
            if request.stream:
                return StreamingResponse(content=create_streaming_generator(
                    promise, postproc_params),
                                         media_type="text/event-stream")
            else:
                response = await self._create_chat_response(
                    promise, postproc_params, raw_request, disaggregated_params)
                return JSONResponse(response.model_dump())

        except ValueError as e:
            return self.create_error_response(str(e))
        except Exception as e:
            logger.error("Error in harmony chat completion: %s", e)
            logger.debug("Error details: %s", traceback.format_exc())
            return self.create_error_response(message=str(e),
                                              err_type="internal_error")

    def stream_model(self, request: ChatCompletionRequest) -> str:
        # The served name, as the aggregated route always announced it.
        return self.model

    def error_message(self, error: ServingError) -> str:
        # What the rejected chat route's body said: its message, else the
        # whole body.
        return error.body.message or json.dumps(error.body.model_dump())


_warned_unresolvable_thinking = False


def _warn_unresolvable_thinking_once(reasoning_parser: str) -> None:
    """Warn when a generation worker cannot learn the reasoning mode.

    Nothing was rendered here and the context worker relayed no value, so
    every request on this deployment is parsed in a possibly wrong mode with
    nothing in the response saying so. Reachable via a rolling upgrade where
    the context worker predates `resolved_thinking`, a hand-crafted
    `generation_only` request, or an orchestration path that does not go
    through `_get_ctx_request`.
    """
    global _warned_unresolvable_thinking
    if _warned_unresolvable_thinking:
        return
    _warned_unresolvable_thinking = True
    logger.warning(
        f"Reasoning parser {reasoning_parser!r} resolves its mode from the "
        "rendered prompt, but this request had neither a rendered prompt nor "
        "a relayed mode from a context worker. Reasoning content will not be "
        "separated correctly. Check that the context worker is running a "
        "build that relays 'resolved_thinking'.")


def _validate_required_tool_grammar_request(
        request: ChatCompletionRequest, sampling_params: SamplingParams, *,
        guided_decoding_backend: Optional[str], tool_parser: Optional[str],
        reasoning_parser: Optional[str]) -> None:
    """Validate inputs for mandatory tool-call grammar construction."""
    if guided_decoding_backend != "xgrammar":
        raise ValueError("Required tool grammars need the xgrammar backend.")
    if not tool_parser:
        raise ValueError("Required tool grammars need a tool parser.")
    if not reasoning_parser or not ReasoningParserFactory.resolves_thinking_from_prompt(
            reasoning_parser):
        raise ValueError(
            "Required tool grammars need a reasoning parser that resolves its "
            "mode from the rendered prompt. Configure the server with "
            "--reasoning_parser nemotron-v3 (or a compatible parser alias).")
    if not request.tools:
        raise ValueError("tool_choice='required' requires request-level tools.")
    if (request.prompt_token_ids is not None or request.prompt_token_ids_b64
            or not request.add_generation_prompt
            or request.disaggregated_params is not None):
        raise ValueError("Required tool grammars need chat messages with "
                         "add_generation_prompt=True on a native server.")
    if sampling_params.guided_decoding is not None:
        raise ValueError(
            "tool_choice='required' cannot be combined with response_format "
            "or other guided-decoding settings.")


def _build_tool_strict_guided_decoding_params(tools, tool_parser_name):
    """Build GuidedDecodingParams with structural tags for tools with strict=True.

    When a tool has ``strict=True`` in its function definition, the server
    should use constrained decoding to guarantee that the generated tool call
    arguments exactly match the function's ``parameters`` JSON Schema.

    This function builds structural tag items from each tool parser's
    ``structure_info()`` and the tool's ``parameters`` schema, then returns
    a ``GuidedDecodingParams`` with the structural tag format.

    Returns None if no tool has strict=True or the parser doesn't support
    structural tags.
    """
    if not tools or not tool_parser_name:
        return None

    # Check if any tool has strict=True
    has_strict = any(tool.function.strict for tool in tools
                     if tool.function.strict)
    if not has_strict:
        return None

    tool_parser_cls = ToolParserFactory.parsers.get(tool_parser_name.lower())
    if tool_parser_cls is None:
        logger.warning(
            "Tool parser '%s' not found, cannot enforce strict mode for tools.",
            tool_parser_name)
        return None

    parser = tool_parser_cls()
    # Parsers whose wire format can't be expressed through structure_info
    # triples (kimi_k3 XTML) provide a complete format themselves.
    custom_format = parser.build_strict_structural_tag_format(tools)
    if custom_format is not None:
        resp_format = ResponseFormat(type="structural_tag",
                                     format=custom_format)
        return GuidedDecodingParams(structural_tag=resp_format.model_dump_json(
            by_alias=True, exclude_none=True))
    if not parser.supports_structural_tag():
        logger.warning(
            "Tool parser '%s' does not support structural tags, "
            "cannot enforce strict mode for tools.", tool_parser_name)
        return None

    get_info = parser.structure_info()

    tags = []
    triggers = set()
    for tool in tools:
        info = get_info(tool.function.name)
        triggers.add(info.trigger)

        if tool.function.strict and tool.function.parameters:
            # Strict tool: constrain arguments to match the JSON Schema
            content = {
                "type": "json_schema",
                "json_schema": tool.function.parameters,
            }
        else:
            # Non-strict tool or no parameters: allow any text
            content = {"type": "any_text"}

        tags.append({
            "begin": info.begin,
            "content": content,
            "end": info.end,
        })

    stag_format = {
        "type": "triggered_tags",
        "triggers": sorted(triggers),
        "tags": tags,
    }

    resp_format = ResponseFormat(type="structural_tag", format=stag_format)
    return GuidedDecodingParams(structural_tag=resp_format.model_dump_json(
        by_alias=True, exclude_none=True))


def _parser_extracts_forced_tool_calls(tool_parser_name) -> bool:
    """Whether this parser extracts forced calls from its own markup.

    Mirrors ``_forced_choice_uses_tool_parser`` in postprocess_handlers: those
    parsers emit native markup on a forced call and the post-processor parses
    it, so the request must not be constrained (or rejected) here.
    """
    if not tool_parser_name:
        return False
    parser_cls = ToolParserFactory.parsers.get(tool_parser_name.lower())
    return bool(parser_cls
                and getattr(parser_cls, "extracts_forced_tool_calls", False))


def _build_forced_tool_call_decoding(tools, tool_parser_name, forced_tool_name):
    r"""Build the begin-prefix and guided-decoding params for a named tool_choice.

    OpenAI / Azure spec: ``tool_choice={"type":"function","function":{"name":"X"}}``
    must force the model to emit exactly one tool call to function ``X``.

    Unlike the strict-tools path (which relies on xgrammar's *triggered*
    structural tags and is therefore non-mandatory by design), this path
    forces the call by:

    1. Asking the caller to prepend ``begin_prefix`` (e.g.
       ``<tool_call>\n{"name":"X", "arguments":``) to the rendered prompt
       so the model starts generation *already inside* the tool call.
    2. Constraining what follows with a plain JSON-schema (or json-object)
       guided-decoding mode so the arguments are guaranteed to be valid
       against ``tool.function.parameters``.

    The caller is then responsible for synthesizing the resulting
    ``tool_calls`` entry — the raw model output is the JSON arguments
    string of the forced function call.

    Raises:
        ValueError: if ``tools`` is empty, ``tool_parser_name`` is unset, the
            parser is not registered, the parser does not support structural
            tags, or ``forced_tool_name`` is not in ``tools``.
    """
    if not tools:
        raise ValueError("tool_choice requires a named function "
                         f"'{forced_tool_name}' but no tools were provided.")
    if not tool_parser_name:
        raise ValueError(
            "tool_choice with a named function requires a tool parser "
            "to be configured on the server (use --tool_parser).")

    tool_parser_cls = ToolParserFactory.parsers.get(tool_parser_name.lower())
    if tool_parser_cls is None:
        raise ValueError(
            f"Tool parser '{tool_parser_name}' is not registered; cannot "
            f"force tool_choice to function '{forced_tool_name}'.")

    parser = tool_parser_cls()
    if not parser.supports_structural_tag():
        raise ValueError(
            f"Tool parser '{tool_parser_name}' does not support structural "
            "tags, which are required to honor tool_choice with a named "
            "function.")

    forced_tool = next(
        (t for t in tools if t.function.name == forced_tool_name), None)
    if forced_tool is None:
        available = sorted(t.function.name for t in tools if t.function.name)
        raise ValueError(
            f"tool_choice requested function '{forced_tool_name}' which is "
            f"not present in `tools`. Available functions: {available}.")

    info = parser.structure_info()(forced_tool.function.name)
    begin_prefix = info.begin

    if forced_tool.function.parameters:
        guided = GuidedDecodingParams(json=forced_tool.function.parameters)
    else:
        # No parameters declared — still require a JSON object (typically ``{}``)
        # so the synthesized ``arguments`` field is well-formed JSON.
        guided = GuidedDecodingParams(json_object=True)

    return begin_prefix, guided
