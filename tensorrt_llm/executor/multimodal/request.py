import asyncio
from dataclasses import dataclass, field
from typing import (Any, List, Optional, AsyncIterator)

import torch

from typing import cast
from tensorrt_llm.inputs.utils import load_image
from tensorrt_llm.multimodal_params import MultimodalParams

__all__ = [
    "MultimodalRequest",
    "MultimodalResponse",
]

# TODO: this should already be in the gh-main
async def async_load_image(
        image: str,
        format: str = "pt",
        device: str = "cuda"):
    from PIL import Image
    from io import BytesIO
    from pathlib import Path
    from urllib.parse import urlparse
    import aiohttp
    import base64
    from torchvision.transforms import ToTensor

    def _load_and_convert_image(image):
        image = Image.open(image)
        image.load()
        return image.convert("RGB")

    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    parsed_url = urlparse(image)

    if parsed_url.scheme in ["http", "https"]:
        async with aiohttp.ClientSession() as session:
            async with session.get(image) as response:
                content = await response.read()
                image = _load_and_convert_image(BytesIO(content))
    elif parsed_url.scheme == "data":
        data_spec, data = parsed_url.path.split(",", 1)
        media_type, data_type = data_spec.split(";", 1)

        if data_type != "base64":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        content = base64.b64decode(data)
        image = _load_and_convert_image(BytesIO(content))
    else:
        image = _load_and_convert_image(Path(parsed_url.path))

    if format == "pt":
        return ToTensor()(image).to(device=device)
    else:
        return image


# TODO: move to a separate file
@dataclass(slots=True)
class MultimodalItem:
    # request id for this item
    req_id: int
    # the id of the mm item within each request
    id: int
    # the modality type of the item
    modality_type: str
    # The url of the item
    url: str
    # The data of the item
    data: Optional[Any] = None # Any: can be raw tensor or processed tensor
    data_handle: Optional[bytes] = None
    # Whether the item has been materialized
    materialized: bool = False

    # The # of tokens offset of the item within each request after encoder
    offset: int = 0
    # The # of tokens length of the item
    length: int = 0

    # The coroutine for the item
    coroutine: Optional[asyncio.Task] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.data is not None or self.data_handle is not None:
            self.materialized = True


    async def async_prefetch(self):
        try:
            if self.modality_type == "image":
                self.coroutine = async_load_image(self.url, format="pil", device="cpu")
            elif self.modality_type == "video":
                assert False, "video is not supported yet"
                #self.coroutine = async_load_video(self.url)
            else:
                raise ValueError(f"Unknown modality type: {self.modality_type}")
        except Exception as e:
            self.materialized = False
            self.error_message = str(e)

    async def retrieve(self):
        if self.coroutine:
            try:
                self.data = await self.coroutine
                if self.data is not None:  # Only set materialized if we got valid data
                    self.materialized = True
                return self.data
            except Exception as e:
                self.materialized = False
                self.error_message = str(e)
                return None
        else:
            return self.data

    def load(self):
        try:
            if self.modality_type == "image":
                self.data = load_image(self.url, format="pil", device="cpu")
                self.materialized = True
            elif self.modality_type == "video":
                assert False, "video is not supported yet"
                #self.coroutine = load_video(self.url)
            else:
                raise ValueError(f"Unknown modality type: {self.modality_type}")
        except Exception as e:
            self.materialized = False
            self.error_message = str(e)

    @staticmethod
    async def process_items(items: List['MultimodalItem']) -> AsyncIterator['MultimodalItem']:
        """Process a list of items concurrently and yield them as they complete.

        Args:
            items: List of MultimodalItems to process

        Yields:
            MultimodalItems as they complete loading
        """
        # Create tasks for all items
        tasks = {}
        for item in items:
            # Create and store the task
            task = asyncio.create_task(item.retrieve())
            tasks[task] = item  # Map task to item directly

        # Process results as they arrive
        while tasks:
            # Wait for any task to complete
            done, pending = await asyncio.wait(
                tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks
            for task in done:
                item = tasks.pop(task)  # Get the item directly from the task mapping
                try:
                    result = await task
                    if result is not None:
                        yield item
                    else:
                        # Item failed to load but we still yield it with error state
                        item.materialized = False
                        item.error_message = "Failed to load item"
                        yield item
                except Exception as e:
                    item.materialized = False
                    item.error_message = str(e)
                    yield item


class MultimodalRequest:
    """
    A request class for multimodal encoding.
    Handles requests containing URLs for different modalities (images, videos, etc.)
    and returns embeddings for each item.
    """

    def __init__(self, items: Optional[List[MultimodalItem]] = None):
        self.items = items or []  # type: List[MultimodalItem]
        self.is_dummy = False
        self.state = "PENDING"  # LOADED, PROCESSED, COMPLETED
        self.sampling_params = None
        self.id: Optional[int] = None
        self.error_message = None

    def set_id(self, id):
        self.id = id
        for item in self.items:
            item.req_id = id
        return self

    def has_error(self) -> bool:
        # TODO: check if any item AND output better err msg
        return any(item.error_message for item in self.items)

    async def prefetch(self):
        await asyncio.gather(*[item.async_prefetch() for item in self.items])

    async def fetch(self):
        """Load and fill data for all items in the request.

        This method will:
        1. Initialize loading for all items using prefetch()
        2. Process all items concurrently and wait for them to complete
        3. Update the request state based on the results

        Returns:
            bool: True if all items were loaded successfully, False otherwise
        """
        await self.prefetch()
        async for item in MultimodalItem.process_items(self.items):
            item.coroutine = None
            item.data_handle = None # TODO: need to remove this

    def load(self):
        for item in self.items:
            item.load()

    @classmethod
    def from_chat_messages(cls, messages) -> "MultimodalRequest":
        request = cls()
        count = 0

        for message in messages:
            content = message.get("content", [])
            # Ignore empty and txt content
            if isinstance(content, str) or content is None:
                content = []
            # Process each content part
            for part in content:
                assert isinstance(part, dict)
                part_type = part.get("type", None)
                if part_type is None or part_type == "text":
                    continue

                # Handle image_url type
                if part_type == "image_url":
                    url = part.get("image_url", {}).get("url")
                    if url:
                        url = cast(str, url)
                        request.items.append(MultimodalItem(req_id=request.id, id=count, modality_type="image", url=url))
                        count += 1
                # TODO: Handle video_url type hasn't been tested yet
                elif part_type == "video_url":
                    url = part.get("video_url", {}).get("url")
                    if url:
                        url = cast(str, url)
                        request.items.append(MultimodalItem(req_id=request.id, id=count, modality_type="video", url=url))
                        count += 1

        return request

    def create_response(self):
        """Create a response object and set up IPC communication.

        Returns:
            MultimodalResponse: The response object that will be populated with results
        """
        num_items = len(self.items)
        item_offsets = [item.offset for item in self.items]
        item_token_length = [item.length for item in self.items]
        # TODO: how to set client id? is it always the same as request id?
        response = MultimodalResponse(request_id=self.id, client_id=self.id, num_items=num_items, item_offsets=item_offsets, item_token_length=item_token_length)
        return response

@dataclass(slots=True)
class MultimodalResponse:
    """Response for multimodal requests containing embeddings for each item."""
    request_id: int
    client_id: Optional[int] = None
    num_items: int = 0
    item_offsets: List[int] = field(default_factory=list)
    item_token_length: List[int] = field(default_factory=list)
    embeddings: Optional[torch.Tensor] = None
    embedding_handle: Optional[bytes] = None
    mrope_config: Optional[dict] = None
    _is_final: bool = False
    error_msg: Optional[str] = None
    cp_event: Optional[torch.cuda.Event] = None

    def set_embeddings(self, embeddings: torch.Tensor, cp_event: Optional[torch.cuda.Event] = None) -> None:
        """Set the embeddings for the response."""
        self.embeddings = embeddings
        self.cp_event = cp_event

    # TODO: error handling is missing; hopefully pass the error from mmItem or during processing
    def set_error(self, error_msg: str) -> None:
        """Set an error message for the response."""
        self.error_msg = error_msg

    def set_mrope_config(self, mrope_config: dict) -> None:
        """Set the mrope config for the response."""
        self.mrope_config = mrope_config

    def has_error(self) -> bool:
        """Check if the response has an error."""
        return self.error_msg is not None

    def set_final(self) -> None:
        """Set the response to final."""
        self._is_final = True

    # TODO: this is a hack to make the result compatible with the proxy/worker architecture
    @property
    def result(self):
        """Return a result object compatible with the proxy/worker architecture."""
        return type('Result', (), {
            'is_final': self._is_final,  # Multimodal responses are always final
            'error_msg': self.error_msg
        })

    def get_params(self):
        if self.embedding_handle:
            # Convert the serialized tensor info to a JSON-serializable format
            embeddings = []
            for tensor_info in self.embedding_handle:
                embeddings.append(tensor_info.dump_to_dict())
        else:
            embeddings = None

        return MultimodalParams(
            embeddings=embeddings,
            mrope_config=self.mrope_config,
            num_items=self.num_items,
            item_offsets=self.item_offsets,
            item_token_length=self.item_token_length)

