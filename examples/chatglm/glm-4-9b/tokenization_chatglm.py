import base64
import json
import os
from typing import Any, Dict, List, Optional, Union

import regex as re
import tiktoken
from torch import TensorType
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, EncodedInput
from transformers.utils import PaddingStrategy


class ChatGLM4Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self,
                 vocab_file,
                 padding_side="left",
                 clean_up_tokenization_spaces=False,
                 encode_special_tokens=False,
                 **kwargs):
        self.name = "GLM4Tokenizer"
        self.vocab_file = vocab_file
        pat_str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        self.pat_str = re.compile(pat_str)
        self.encode_special_tokens = encode_special_tokens

        mergeable_ranks = {}
        with open(vocab_file) as f:
            for line in f:
                token, rank = line.strip().split()
                rank = int(rank)
                token = base64.b64decode(token)
                mergeable_ranks[token] = rank

        self.mergeable_ranks = mergeable_ranks

        self.tokenizer = tiktoken.Encoding(name="my_tokenizer",
                                           pat_str=pat_str,
                                           mergeable_ranks=mergeable_ranks,
                                           special_tokens={})
        self.decoder = {rank: token for token, rank in mergeable_ranks.items()}
        self.n_words = len(self.decoder)

        super().__init__(
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs)

    @property
    def vocab_size(self):
        return self.n_words

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {
            self._convert_id_to_token(i): i
            for i in range(self.vocab_size)
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str,
                                                          int]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, int):
                t = chr(t)
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors="replace")
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError(
                    "token should only be of type int, bytes or str")
        if temp:
            text += temp.decode("utf-8", errors="replace")
        return text

    def _tokenize(self, text, **kwargs):
        tokens = []
        ids = self.tokenizer.encode(text)
        for t in ids:
            tokens.append(self.decoder[t])
        return tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.mergeable_ranks[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, "")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory,
                                      self.vocab_files_names["vocab_file"])
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file, )

    def get_prefix_tokens(self):
        prefix_tokens = [
            self.convert_tokens_to_ids("[gMASK]"),
            self.convert_tokens_to_ids("<sop>")
        ]
        return prefix_tokens

    def build_single_message(self, role, metadata, message, tokenize=True):
        assert role in ["system", "user", "assistant", "observation"], role
        if tokenize:
            role_tokens = [self.convert_tokens_to_ids(f"<|{role}|>")
                           ] + self.tokenizer.encode(f"{metadata}\n",
                                                     disallowed_special=())
            message_tokens = self.tokenizer.encode(message,
                                                   disallowed_special=())
            tokens = role_tokens + message_tokens
            return tokens
        else:
            return str(f"<|{role}|>{metadata}\n{message}")

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]],
                            "Conversation"],
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> Union[str, List[int], List[str], List[List[int]], BatchEncoding]:

        if return_dict and not tokenize:
            raise ValueError(
                "`return_dict=True` is incompatible with `tokenize=False`, because there is no dict "
                "of tokenizer outputs to return.")

        def handle_single_conversation(conversation):
            input_ids = self.get_prefix_tokens() if add_special_tokens else []
            input_message = "[gMASK]<sop>" if add_special_tokens else ""
            for item in conversation:
                if item.get("tools"):
                    tools = item["tools"]
                    content = "你是一个名为 GhatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。"
                    content += "\n\n# 可用工具"
                    for tool in tools:
                        if tool["type"] == "function":
                            function = tool["function"]
                            content += f"\n\n## {function['name']}\n\n{json.dumps(function, ensure_ascii=False, indent=4)}"
                            content += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
                        elif tool["type"] == "python":
                            content += "\n\n## python\n\n当你向 `python` 发送包含 Python 代码的消息时，该代码将会在一个有状态的 Jupyter notebook 环境中执行。\n`python` 返回代码执行的输出，或在执行 60 秒后返回超时。\n`/mnt/data` 将会持久化存储你的文件。在此会话中，`python` 无法访问互联网。不要使用 `python` 进行任何网络请求或者在线 API 调用，这些在线内容的访问将不会成功。"
                        elif tool["type"] == "simple_browser":
                            content += "\n\n## simple_browser\n\n你可以使用 `simple_browser` 工具。该工具支持以下函数：\n`search(query: str, recency_days: int)`：使用搜索引擎进行查询并显示结果，可以使用 `recency_days` 参数控制搜索内容的时效性。\n`mclick(ids: list[int])`：获取一系列指定 id 的页面内容。每次调用时，须选择3-10个页面。选择多个角度的页面，同时尽可能选择可信任的信息来源。考虑到部分页面是无法加载的，你也可以多打开一些可能有用的页面而不用担心内容过多。\n`open_url(url: str)`：打开指定的 URL。\n\n使用 `【{引用 id}†{引用文本}】` 来引用内容。\n\n操作步骤：1. 使用 `search` 来获得信息列表; 2. 使用 `mclick` 来获取指定 ID 页面的内容; 3. 根据获得的内容进行回复。在回复中应当引用信息来源。\n 如果用户提供了 URL，也可以用 `open_url` 直接打开页面。\n如果初次搜索结果没有找到合适的信息，也可以再次使用 `search` 进行搜索。"
                        elif tool["type"] == "cogview":
                            content += "\n\n## cogview\n\n如果用户的请求中包含了对图像的描述，你可以使用 `cogview` 来生成图像并展示给用户。你需要向 `cogview` 发送图像描述，规则：\n- 发送给 `cogview` 的消息必须使用英语。用户的中文描述必须完全翻译为英语。\n- 应当尽可能详细地描述图像生成的需求，需求描述约 100 英文单词。\n- 保持用户原始描述的意图。不要虚构内容或者没见过的人物。\n- 如无特殊说明，所在地为中国，持有中国立场并遵循中国社会主义价值观。"
                        else:
                            raise NotImplementedError(
                                f"Unknown tool type {tool['type']}")
                    input = self.build_single_message("system",
                                                      "",
                                                      content,
                                                      tokenize=tokenize)
                    if tokenize:
                        input_ids.extend(input)
                    else:
                        input_message += input
                if item["content"]:
                    input = self.build_single_message(item["role"],
                                                      item.get("metadata", ""),
                                                      item["content"],
                                                      tokenize=tokenize)
                    if tokenize:
                        input_ids.extend(input)
                    else:
                        input_message += input
            if add_generation_prompt:
                if tokenize:
                    input_ids.extend(
                        [self.convert_tokens_to_ids("<|assistant|>")])
                else:
                    input_message += "<|assistant|>"
            return input_ids if tokenize else input_message

        # Main logic to handle different conversation formats
        if isinstance(conversation, list) and all(
                isinstance(i, dict) for i in conversation):
            result = handle_single_conversation(conversation)
        elif isinstance(conversation, list) and all(
                isinstance(i, list) for i in conversation):
            result = [handle_single_conversation(c) for c in conversation]
        elif hasattr(conversation, "messages"):
            result = handle_single_conversation(conversation.messages)
        else:
            raise ValueError("Invalid conversation format")

        if tokenize:
            output = self.batch_encode_plus(
                [result] if isinstance(result[0], int) else result,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                is_split_into_words=True,
                add_special_tokens=False)
            if return_dict:
                return output
            else:
                return output["input_ids"]
        else:
            return result

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [
                self.convert_tokens_to_ids("<eos>")
            ]
        return token_ids_0

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
            padding_side: str = "left",  # Fix for new transformers
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        assert self.padding_side == "left"

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (
                max_length % pad_to_multiple_of != 0):
            max_length = (
                (max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(
            required_input) != max_length

        # Initialize attention mask if not present.
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [
                    0
                ] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [
                    0
                ] * difference + encoded_inputs["position_ids"]
            encoded_inputs[self.model_input_names[
                0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs
