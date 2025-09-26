# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer


def get_token_id(tokenizer: Any, word: str) -> int:
    '''Get the token id for a word using the provided tokenizer.'''
    try:
        return tokenizer.encode(word, add_special_tokens=False)[0]
    except (IndexError, AttributeError, TypeError) as exc:
        pytest.skip(f'Could not get token id for {word}: {exc}')


async def logit_bias_effect_helper(client: Any,
                                   model_name: str,
                                   api_type: str = 'completions') -> None:
    '''Helper function to test logit bias effects for both chat and completions APIs.

    Args:
        client: OpenAI async client
        model_name: Model name to test
        api_type: Either 'completions' or 'chat' to determine which API to use
    '''
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(get_model_path(model_name))
        paris_token_id = get_token_id(tokenizer, 'Paris')
    except ImportError as exc:
        pytest.skip(f'transformers not available: {exc}')
    except Exception as exc:
        paris_token_id = 3681
        print(f'[WARNING] Using fallback token id 3681 for "Paris": {exc}')

    # Test with strong positive bias for 'Paris'
    logit_bias = {str(paris_token_id): 80}

    if api_type == 'completions':
        response = await client.completions.create(
            model=model_name,
            prompt='The capital of France is',
            max_tokens=5,
            logit_bias=logit_bias,
            temperature=0.0,
        )
        output = response.choices[0].text
    elif api_type == 'chat':
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": "The capital of France is"
            }],
            max_tokens=5,
            logit_bias=logit_bias,
            temperature=0.0,
        )
        output = response.choices[0].message.content
    else:
        raise ValueError(f"Unsupported api_type: {api_type}")

    assert 'Paris' in output, f"Expected 'Paris' in output with positive logit bias, got: {output}"

    # Test with strong negative bias for 'Paris'
    logit_bias = {str(paris_token_id): -80}

    if api_type == 'completions':
        response = await client.completions.create(
            model=model_name,
            prompt='The capital of France is',
            max_tokens=5,
            logit_bias=logit_bias,
            temperature=0.0,
        )
        output = response.choices[0].text
    elif api_type == 'chat':
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": "The capital of France is"
            }],
            max_tokens=5,
            logit_bias=logit_bias,
            temperature=0.0,
        )
        output = response.choices[0].message.content

    assert 'Paris' not in output, f"Did not expect 'Paris' in output with negative logit bias, got: {output}"


async def invalid_logit_bias_helper(client: Any,
                                    model_name: str,
                                    api_type: str = 'completions') -> None:
    '''Helper function to test invalid logit bias for both chat and completions APIs.

    Args:
        client: OpenAI async client
        model_name: Model name to test
        api_type: Either 'completions' or 'chat' to determine which API to use
    '''
    import openai

    with pytest.raises(openai.BadRequestError):
        if api_type == 'completions':
            await client.completions.create(
                model=model_name,
                prompt="Hello world",
                logit_bias={"invalid_token": 1.0},  # Non-integer key
                max_tokens=5,
            )
        elif api_type == 'chat':
            await client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": "Hello world"
                }],
                logit_bias={"invalid_token": 1.0},  # Non-integer key
                max_tokens=5,
            )
        else:
            raise ValueError(f"Unsupported api_type: {api_type}")


def make_server_with_custom_sampler_fixture(api_type: str) -> Callable:
    '''Factory for a pytest fixture that launches a server with a custom sampler config.
    api_type: 'chat' or 'completions' (for error messages only)
    '''

    @pytest.fixture(scope='function')
    def server_with_custom_sampler(model_name: str, request: Any, backend: str,
                                   tmp_path: Path) -> RemoteOpenAIServer:
        '''Fixture to launch a server (pytorch backend only) with a custom sampler configuration.'''
        sampler_type = getattr(request, 'param', {}).get('sampler_type', "auto")
        if backend != 'pytorch':
            pytest.skip(
                f"Server with custom sampler is only supported for pytorch backend, skipping for {backend}"
            )
        model_path = get_model_path(model_name)
        args = ['--backend', backend]
        temp_file_path = tmp_path / f'test_sampler_config_{request.node.name}.yaml'
        extra_llm_api_options_dict = {
            'enable_chunked_prefill': True,
            'sampler_type': sampler_type
        }
        with temp_file_path.open('w') as f:
            yaml.dump(extra_llm_api_options_dict, f)
        args.extend(['--extra_llm_api_options', str(temp_file_path)])
        args.extend(['--num_postprocess_workers',
                     str(0)])  # disable postprocess workers to avoid OOM

        with RemoteOpenAIServer(model_path, args) as remote_server:
            yield remote_server

    return server_with_custom_sampler


def expand_slurm_nodelist(nodelist_str):
    """
    Expand SLURM nodelist format into individual node names.
    An equivalent of scontrol show hostname $SLURM_JOB_NODELIST, but
    scontrol is not available in the container.
    """

    # Handle empty or None input
    if not nodelist_str or nodelist_str.strip() == "":
        return []

    # Split top-level groups by commas not inside brackets.
    groups: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in nodelist_str:
        if ch == '[':
            depth += 1
        elif ch == ']' and depth:
            depth -= 1
        if ch == ',' and depth == 0:
            groups.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        groups.append(''.join(buf).strip())

    for group in groups:
        bracket_match = re.match(r'^([^\[]+)\[(.+?)\]$', group)
        if bracket_match:
            prefix = bracket_match.group(1)
            range_part = bracket_match.group(2)

    expanded_nodes = []

    for group in groups:
        # Check if this group has bracket notation
        bracket_match = re.match(r'^(.+?)\[(.+?)\]$', group)
        if bracket_match:
            prefix = bracket_match.group(1)
            range_part = bracket_match.group(2)

            # Handle ranges and individual numbers within brackets
            range_parts = range_part.split(',')

            for part in range_parts:
                part = part.strip()

                # Check if it's a range (contains dash)
                if '-' in part:
                    range_match = re.match(r'^(\d+)-(\d+)$', part)
                    if range_match:
                        start_num = int(range_match.group(1))
                        end_num = int(range_match.group(2))
                        # Determine zero-padding width from the original format
                        start_str = range_match.group(1)
                        width = len(start_str)

                        # Generate range
                        for num in range(start_num, end_num + 1):
                            node_name = f"{prefix}{num:0{width}d}"
                            expanded_nodes.append(node_name)
                    else:
                        # Handle non-numeric ranges or invalid format
                        expanded_nodes.append(part)
                else:
                    # Individual number
                    if part.isdigit():
                        # Preserve zero-padding
                        node_name = f"{prefix}{part}"
                        expanded_nodes.append(node_name)
                    else:
                        # Handle non-numeric individual items
                        node_name = f"{prefix}{part}"
                        expanded_nodes.append(node_name)
        else:
            # No brackets, just add the node as-is
            expanded_nodes.append(group)

    return expanded_nodes
