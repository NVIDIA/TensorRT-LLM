import asyncio
import traceback
from collections import deque
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple,
                    Optional)

import zmq
import zmq.asyncio

from ..llmapi.tokenizer import TransformersTokenizer, load_hf_tokenizer
from ..llmapi.utils import nvtx_range, print_traceback_on_error
from ..sampling_params import SamplingParams
from .ipc import ZeroMqQueue
from .utils import ExecutorResponse

if TYPE_CHECKING:
    from .result import (DetokenizedGenerationResultBase, GenerationResult,
                         GenerationResultBase)

__all__ = [
    "PostprocWorker",
    "PostprocWorkerConfig",
]


@dataclass(kw_only=True)
class PostprocArgs:
    first_iteration: bool = True
    num_prompt_tokens: Optional[int] = None
    tokenizer: Optional[TransformersTokenizer] = None


@dataclass(kw_only=True)
class PostprocParams:
    post_processor: Callable[["GenerationResultBase", PostprocArgs], Any] = None
    postproc_args: PostprocArgs = None


@dataclass
class PostprocWorkerConfig:
    ''' The config for the postprocess worker. '''
    num_postprocess_workers: int = 0
    postprocess_tokenizer_dir: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.num_postprocess_workers > 0


class PostprocWorker:
    '''
    The worker to postprocess the responses from the executor's await_response.
    '''

    class Input(NamedTuple):
        rsp: "ExecutorResponse"

        # The information necessary for creating a GenerationResult in the first Input for each request
        sampling_params: Optional[SamplingParams] = None
        postproc_params: Optional[PostprocParams] = None
        streaming: Optional[bool] = None

    class Output(NamedTuple):
        client_id: int
        res: Any
        is_final: bool
        error: str = ""

    def __init__(
        self,
        pull_pipe_addr: str,
        push_pipe_addr: str,
        tokenizer_dir: str,
        record_creator: Callable[
            ["PostprocWorker.Input", TransformersTokenizer], Any],
    ):
        '''
        Args:
            pull_pipe_addr (str): The address of the input IPC.
            push_pipe_addr (str): The address of the output IPC.
            tokenizer_dir (str): The directory to load tokenizer.
            record_creator (Callable[["ResponsePostprocessWorker.Input"], Any]): A creator for creating a record for a request.
            result_handler (Optional[Callable[[GenerationResultBase], Any]]): A callback handles the final result.
        '''

        self._records: Dict[int, GenerationResult] = {}
        self._record_creator = record_creator
        self._pull_pipe = ZeroMqQueue(address=pull_pipe_addr,
                                      is_async=True,
                                      is_server=False,
                                      name="postprocess_pull_pipe")
        self._push_pipe = ZeroMqQueue(address=push_pipe_addr,
                                      is_async=True,
                                      is_server=False,
                                      socket_type=zmq.PUSH,
                                      name="postprocess_push_pipe")
        self._to_stop = asyncio.Event()

        self._q = deque()

        # Load the tokenizer and share in all records
        self._tokenizer = load_hf_tokenizer(tokenizer_dir)

    @staticmethod
    def default_record_creator(
            inp: "PostprocWorker.Input", tokenizer: TransformersTokenizer
    ) -> "DetokenizedGenerationResultBase":
        from .result import DetokenizedGenerationResultBase
        assert inp.sampling_params is not None
        assert tokenizer is not None
        return DetokenizedGenerationResultBase(
            inp.rsp.client_id,
            sampling_params=inp.sampling_params,
            postproc_params=inp.postproc_params,
            streaming=inp.streaming,
            tokenizer=tokenizer)

    async def _handle_input(self, input: "PostprocWorker.Input") -> Any:
        ''' Handle a single response from await_response worker. '''
        with nvtx_range("handle_input", color="yellow", category="Postproc"):
            req_id = input.rsp.client_id
            if req_id not in self._records:
                # TODO: support variant creation later
                self._records[req_id] = self._record_creator(
                    input, self._tokenizer)

            record = self._records[req_id]
            record._handle_response(input.rsp)  # inplace
            # Left the result_handler determine the final output dtype.
            # NOTE: This will change the CompletionOutput._postprocess_result
            if postproc_params := record.postproc_params:
                result_handler, args = postproc_params.post_processor, postproc_params.postproc_args
                args.tokenizer = self._tokenizer
                out = result_handler(record, args)
            else:
                # This should only be called in streaming mode, and each time it
                # produces a single output.
                out = record.outputs[0]

            # TODO: Keep only the diff token_ids and text in streaming mode when
            # result_handler is not set
            return out

    async def _batched_put(self):
        ''' Batched IPC send. '''
        async for batch in self._mainloop():
            if batch is None:
                # notify dispatch_result corountine to quit
                await self._push_pipe.put_async(None)
                break
            assert isinstance(batch, list)
            await self._push_pipe.put_async(batch)

    async def _mainloop(self):
        ''' The loop for handle_response and keep producing outputs. '''

        PostprocWorker.Input
        Output = PostprocWorker.Output

        async def handle_single_input(inp: PostprocWorker.Input,
                                      batch: List[PostprocWorker.Output]):
            assert isinstance(inp, PostprocWorker.Input)
            rsp = inp.rsp
            client_id = rsp.client_id
            is_final = bool(rsp.is_final)
            res = await self._handle_input(inp)
            batch.append(Output(client_id, res, is_final))
            if is_final:
                self._records.pop(client_id)

        while not self._to_stop.is_set():
            batch = []
            inputs: Optional[List[PostprocWorker.Input]
                             | PostprocWorker.
                             Input] = await self._pull_pipe.get_async()

            if inputs is None:
                self._to_stop.set()
                yield None  # notify the batched_put corountine to quit
                break

            if isinstance(inputs, list):  # batched
                for inp in inputs:
                    if inp is None:
                        self._to_stop.set()
                        yield None
                        break
                    await handle_single_input(inp, batch)
            else:
                await handle_single_input(inputs, batch)

            yield batch

    def start(self):
        ''' Start the workflow in the current thread. '''

        async def main():
            await asyncio.gather(self._batched_put())

        try:
            asyncio.run(main())
        except Exception as e:
            print(traceback.format_exc())
            raise e


@print_traceback_on_error
def postproc_worker_main(feedin_ipc_addr: str, feedout_ipc_addr: str,
                         tokenizer_dir: str, record_creator: Callable):
    worker = PostprocWorker(feedin_ipc_addr,
                            feedout_ipc_addr,
                            tokenizer_dir=tokenizer_dir,
                            record_creator=record_creator)
    worker.start()
