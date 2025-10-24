"""A demo LLM api to for debugging and testing purposes of e2e workflows."""

import gc
from collections import defaultdict
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

from ....executor import GenerationExecutor
from ....executor.request import GenerationRequest
from ....executor.result import CompletionOutput, GenerationResult
from ....inputs.multimodal import MultimodalParams
from ....sampling_params import SamplingParams
from ...pyexecutor.sampling_utils import greedy_search_sampling_batch, top_k_sampling_batch
from ..distributed import common as dist_ad
from ..utils.logger import ad_logger
from .ad_executor import ADEngine

FusedMHACallable = Callable[..., torch.Tensor]


class DemoEngine(ADEngine):
    """The model engine is responsible for executing the model on each individual rank.

    The engine also owns the cache interface with the cache pools and makes sure that the cache
    information is passed through from the high-level runner in the parent process.

    This is a demo and debugging interface to simplify deployment to the real TRT-LLM runtime.
    """

    @torch.inference_mode()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.queue = mp.Queue()

    @torch.inference_mode()
    def __call__(
        self, requests: GenerationRequest, multimodal_params: Optional[MultimodalParams]
    ) -> mp.Queue:
        """Generate tokens and put the results in a queue and return the queue."""
        requests.multimodal_params = multimodal_params
        output = self.generate_tokens_batched([requests])[0]
        self.queue.put(output)
        return self.queue

    def stop(self):
        """Stop the engine."""
        self.queue.close()
        self.queue.join_thread()

    def _assign_pages(self, total_lens: List[int]) -> List[List[int]]:
        """A simple heuristic to assign pages based on current sequence info.

        In a nutshell, we will look at the following information to update the page assignments:
        1. Sequence lengths
        2. Input positions
        3. Existing page assignments

        The information above is extracted from the SequenceInfo object stored in the
        cache_seq_interface.

        Note that we assume that sequence lengths and input positions has been updated by the
        engine before this method is called. Moreover, we assume that the order of sequences does
        not change or has been updated by the engine before this method is called.

        Then we just look at the total length of each sequence as the sum of the input positions
        (past number of tokens) and the current sequence length. The total length corresponds to the
        total number of tokens that need to be stored in the kv-cache. We now compare this the
        currently available token slots in the assigned pages and assign a new, previously
        unassigned page if needed.
        """
        si = self.cache_seq_interface.info
        page_assignments = si.page_assignments

        free_pages = set(range(si.num_pages)) - {i for pages in page_assignments for i in pages}
        updated_assignments = []
        for t_l, pages in zip(total_lens, page_assignments):
            extra_tokens = t_l - len(pages) * si.page_size
            num_extra_pages = (extra_tokens // si.page_size) + (extra_tokens > 0)
            updated_assignments.append(pages + [free_pages.pop() for _ in range(num_extra_pages)])
        return updated_assignments

    def generate_tokens_batched(
        self, requests: List[GenerationRequest]
    ) -> List[List[CompletionOutput]]:
        if len(requests) == 0:
            return []

        sampling_params = requests[0].sampling_params
        # we don't support heterogeneous sampling params or best-of atm
        assert all(r.sampling_params == sampling_params for r in requests), (
            "Heterogeneous sampling params are not supported."
        )
        assert sampling_params.best_of == 1, "Best-of is not supported."

        # set up sequence info object for decode phase
        sequence_info = self.cache_seq_interface.info

        input_ids = []
        total_lens = []
        extra_args: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for request in requests:
            total_lens.append(len(request.prompt_token_ids))
            input_ids.append(request.prompt_token_ids)
            if request.multimodal_params is not None:
                for k, v in request.multimodal_params.multimodal_data.items():
                    extra_args[k].append(v)

        sequence_info.reset()
        sequence_info.nest_sequences(
            input_ids=input_ids,
            input_pos=0,
            page_assignments=self._assign_pages(total_lens),
            slot_idx=list(range(len(input_ids))),
            **extra_args,
        )

        # setup objects we want to track for the output
        batch_size = sequence_info.num_sequences
        new_tokens = [[] for _ in range(batch_size)]  # [batch_size][max_seq_len]
        stop_tokens = sampling_params._get_stop_words()
        # NOTE: TRTLLM has made the intentional choice to separate `end_id` from `stop_words`, and not
        # include the former in the latter's corresponding stop IDs. From a UX perspective, `stop_words`
        # are optional, and can be customized per user requests, whereas `end_id` is static per model,
        # and should always be used outside of benchmarking.
        stop_tokens.append([sampling_params.end_id])
        idxs_stop = [sampling_params.max_tokens - 1] * batch_size
        gen_logits = [] if sampling_params.return_generation_logits else None
        context_logits: Optional[List[torch.Tensor]] = None

        def _generate_single_step(idx: int):
            logits = self._compute_logits()
            logits_last = torch.stack([l_one_seq[-1] for l_one_seq in logits]).float().unsqueeze(1)

            token_ids, _ = self._decode_tokens(logits_last, sampling_params)  # [b,1]

            # update sequence info accordingly for next step (generate phase)
            input_pos_next = sequence_info.input_pos
            seq_lens_current = sequence_info.seq_len
            input_pos_next = [ip + sl for ip, sl in zip(input_pos_next, seq_lens_current)]
            total_lens_next = [ip + len(t_ids) for ip, t_ids in zip(input_pos_next, token_ids)]
            sequence_info.nest_sequences(
                token_ids,
                input_pos=input_pos_next,
                page_assignments=self._assign_pages(total_lens_next),
            )

            # nest new tokens and run stop check
            for b, (new_tokens_b, new_id) in enumerate(zip(new_tokens, token_ids)):
                # if we stopped already, skip the sequence
                if idxs_stop[b] < idx:
                    continue

                # add new token
                new_tokens_b.append(int(new_id))

                # now check the stop tokens
                for stop_seq in stop_tokens:
                    stop_len = len(stop_seq)
                    if new_tokens_b[-stop_len:] == stop_seq:
                        idxs_stop[b] = idx
                        if not sampling_params.include_stop_str_in_output:
                            idxs_stop[b] -= stop_len
                            del new_tokens_b[-stop_len:]
                        break

            if gen_logits is not None:
                # store logits_last as [b, vocab_size]
                gen_logits.append(logits_last.squeeze(1))

            if idx == 0 and sampling_params.return_context_logits:
                # store context logits as [b][seq_len, vocab_size]
                nonlocal context_logits
                context_logits = logits

        # prefill (i==0) and decode stage (i > 0)
        for i in range(sampling_params.max_tokens):
            _generate_single_step(i)
            if all(i >= i_stop for i_stop in idxs_stop):
                break

        # if existing convert generation_logits from [max_seq_len, batch_size, vocab_size] to
        # [batch_size, max_seq_len, vocab_size]
        if gen_logits is not None:
            gen_logits = torch.stack(gen_logits, dim=0).permute(1, 0, 2).cpu()

        # sanity check on produced tokens and their lengths
        num_tokens = [len(new_ids) for new_ids in new_tokens]
        assert all(i_stop + 1 == nt for i_stop, nt in zip(idxs_stop, num_tokens)), (
            f"{new_tokens=} vs {num_tokens=}"
        )

        # let's put together a completion output object here for each request
        outputs = []
        for b, new_ids in enumerate(new_tokens):
            completion_output = CompletionOutput(
                index=0,
                token_ids=new_ids,
                finish_reason="stop" if len(new_ids) < sampling_params.max_tokens else "length",
                generation_logits=None if gen_logits is None else gen_logits[b, : len(new_ids)],
            )
            completion_output._postprocess_result = {
                "context_logits": None if context_logits is None else context_logits[b].cpu()
            }
            outputs.append([completion_output])

        return outputs

    @staticmethod
    def _multinomial_sample_one_no_sync(probs_sort):
        # Does multinomial sampling without a cuda synchronization
        q = torch.randn_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=False).to(dtype=torch.int)

    @staticmethod
    def _logits_to_probs(
        logits: torch.Tensor, temperature: Optional[float] = None, top_k: Optional[int] = None
    ):
        logits = logits / max(1.0 if temperature is None else temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    @classmethod
    def _sample(
        cls, logits: torch.Tensor, sampling_params: SamplingParams
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_shape = logits.shape
        logits = logits.view(-1, logits_shape[-1])  # sampling_batch expects 2D logits
        if isinstance(sampling_params.top_k, int) and sampling_params.top_k > 1:
            idx_next, probs = top_k_sampling_batch(
                logits, top_k=sampling_params.top_k, temperature=1.0
            )
        else:
            idx_next, probs = greedy_search_sampling_batch(logits)
        idx_next = idx_next.view(logits_shape[:-1])
        return idx_next, probs

    def _decode_tokens(
        self, logits_last: torch.Tensor, sampling_params: SamplingParams
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sampled token per input sequence and associating probability."""
        # run sampling
        return self._sample(logits_last, sampling_params)


class DemoGenerationExecutor(GenerationExecutor):
    """A simple GenerationExecutor for demo and debugging purposes.

    Requests are just handled in the main loop which is helpful for tracing through the workflow
    and use debugging tools.
    """

    def __init__(self, world_size: int = 0, tokenizer: Optional[Any] = None, **engine_kwargs):
        super().__init__()
        self.tokenizer = tokenizer

        # call engine directly or use distributed executor for multi-gpu setting
        if world_size == 0:
            ad_logger.info("Initializing model executor in main process...")
            dist_ad.initialize_or_skip()
            self.engine_executor = DemoEngine.build_from_config(**engine_kwargs)
        else:
            ad_logger.info("Starting multi-process model executor...")
            self.engine_executor = dist_ad.MultiProcessExecutor(
                self._run_engine, world_size=world_size, **engine_kwargs
            )

    @classmethod
    @torch.inference_mode()
    def _run_engine(
        cls,
        rank: int,
        world_size: int,
        *,
        input_queue: mp.Queue,
        output_queue: Optional[mp.Queue] = None,
        **engine_kwargs,
    ):
        def _unpack(inputs) -> GenerationRequest:
            args, kwargs = inputs  # unpack the inputs
            request: GenerationRequest = args[0]
            request.multimodal_params: Optional[MultimodalParams] = args[1]
            return request

        engine = DemoEngine.build_from_config(**engine_kwargs)
        while inputs := input_queue.get():  # blocking wait for inputs
            # create request list
            request_list = [_unpack(inputs)]

            # check if we can quickly add on other requests from the queue
            try:
                while len(request_list) < engine.cache_seq_interface.info.max_batch_size:
                    request_list.append(_unpack(input_queue.get(block=False)))
            except Empty:
                pass

            # let's make sure that all ranks received the same number of requests. Since we use a
            # non-blocking approach to retrieve more requests there might be some mismatch between
            # the ranks.
            num_max_requests = torch.tensor(len(request_list), dtype=torch.int, device="cuda")
            dist_ad.all_reduce(num_max_requests, op=dist_ad.ReduceOp.MAX)
            num_max_requests = int(num_max_requests)
            while len(request_list) < num_max_requests:
                # NOTE: there should be at most a short delay between input_queue's across ranks.
                # Hence, the timeout should only trigger when something went wrong.
                request_list.append(_unpack(input_queue.get(timeout=1.0)))

            # call the engine with the generic args/kwargs
            ad_logger.debug(f"Running engine on {len(request_list)} requests")
            outs = engine.generate_tokens_batched(request_list)

            # put the outputs in the output queue
            if output_queue:
                for out in outs:
                    output_queue.put(out)

            # clean up
            del inputs, request_list, outs

        del engine
        gc.collect()

    def shutdown(self):
        if hasattr(self, "engine_executor"):
            self.engine_executor.stop()

    def __del__(self):
        self.shutdown()

    def submit(self, request: GenerationRequest) -> GenerationResult:
        # set request id if necessary
        client_id = request.id if request.id is not None else self._get_next_client_id()
        if request.id is None:
            request.set_id(client_id)

        # submit request to our demo engine and store results
        # NOTE: when returning from this function, the reference request.multimodal_params will
        # be cleared immediately. So we pass it in explicitly to maintain a reference even when
        # requests get submitted asynchronously.
        result = GenerationResult(request)
        result.queue = self.engine_executor(request, request.multimodal_params)

        return result

    def abort_request(self, client_id: int) -> None:
        ad_logger.warning(f"Abort request is not supported in the demo executor: {client_id=}")
