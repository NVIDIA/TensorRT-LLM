# Guided Decoding

Guided decoding (or interchangeably constrained decoding, structured generation) guarantees that the LLM outputs are amenable to a user-specified grammar (e.g., JSON schema, [regular expression](https://en.wikipedia.org/wiki/Regular_expression) or [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) grammar).

TensorRT LLM supports two grammar backends:
* [XGrammar](https://github.com/mlc-ai/xgrammar/blob/v0.1.21/python/xgrammar/matcher.py#L341-L350): Supports JSON schema, regular expression, EBNF and [structural tag](https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html).
* [LLGuidance](https://github.com/guidance-ai/llguidance/blob/v1.1.1/python/llguidance/_lib.pyi#L363-L366): Supports JSON schema, regular expression, EBNF.


## Online API: `trtllm-serve`

If you are using `trtllm-serve`, enable guided decoding by specifying `guided_decoding_backend` with `xgrammar` or `llguidance` in the YAML configuration file, and pass it to `--config`. For example,

```{eval-rst}
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias
```

```bash
cat > config.yaml <<EOF
guided_decoding_backend: xgrammar
EOF

trtllm-serve nvidia/Llama-3.1-8B-Instruct-FP8 --config config.yaml
```

You should see a log like the following, which indicates the grammar backend is successfully enabled.

```txt
......
[TRT-LLM] [I] Guided decoder initialized with backend: GuidedDecodingBackend.XGRAMMAR
......
```

### JSON Schema

Define a JSON schema and pass it to `response_format` when creating the OpenAI chat completion request. Alternatively, the JSON schema can be created using [pydantic](https://docs.pydantic.dev/latest/).

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

json_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[\\w]+$"
        },
        "population": {
            "type": "integer"
        },
    },
    "required": ["name", "population"],
}
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "Give me the information of the capital of France in the JSON format.",
    },
]
chat_completion = client.chat.completions.create(
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    messages=messages,
    max_completion_tokens=256,
    response_format={
        "type": "json",
        "schema": json_schema
    },
)

message = chat_completion.choices[0].message
print(message.content)
```

The output would look like:
```txt
{
    "name": "Paris",
    "population": 2145200
}
```

### Regular expression

Define a regular expression and pass it to `response_format` when creating the OpenAI chat completion request.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "What is the capital of France?",
    },
]
chat_completion = client.chat.completions.create(
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    messages=messages,
    max_completion_tokens=256,
    response_format={
        "type": "regex",
        "regex": "(Paris|London)"
    },
)

message = chat_completion.choices[0].message
print(message.content)
```

The output would look like:
```txt
Paris
```

### EBNF grammar

Define an EBNF grammar and pass it to `response_format` when creating the OpenAI chat completion request.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

ebnf_grammar = """root ::= description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""
messages = [
    {
        "role": "system",
        "content": "You are a helpful geography bot."
    },
    {
        "role": "user",
        "content": "Give me the information of the capital of France.",
    },
]
chat_completion = client.chat.completions.create(
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    messages=messages,
    max_completion_tokens=256,
    response_format={
        "type": "ebnf",
        "ebnf": ebnf_grammar
    },
)

message = chat_completion.choices[0].message
print(message.content)
```

The output would look like:
```txt
Paris is the capital of France
```

### Structural tag

Define a structural tag and pass it to `response_format` when creating the OpenAI chat completion request.

Structural tag is supported by `xgrammar` backend only. It is a powerful and flexible tool to represent the LLM output constraints. Please see [structural tag usage](https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html) for a comprehensive tutorial. Below is an example of function calling with customized function call format for `Llama-3.1-8B-Instruct`.


```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

tool_get_current_weather = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state", "unit"],
        },
    },
}

tool_get_current_date = {
    "type": "function",
    "function": {
        "name": "get_current_date",
        "description": "Get the current date and time for a given timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                }
            },
            "required": ["timezone"],
        },
    },
}

system_prompt = f"""# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant."""
user_prompt = "You are in New York. Please get the current date and time, and the weather."

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": user_prompt,
    },
]

chat_completion = client.chat.completions.create(
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    messages=messages,
    max_completion_tokens=256,
    response_format={
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "begin": "<function=get_current_weather>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool_get_current_weather["function"]["parameters"]
                    },
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool_get_current_date["function"]["parameters"]
                    },
                    "end": "</function>",
                },
            ],
        },
    },
)

message = chat_completion.choices[0].message
print(message.content)
```

The output would look like:
```txt
<function=get_current_date>{"timezone": "America/New_York"}</function>
<function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>
```


## Offline API: LLM API

If you are using LLM API, enable guided decoding by specifying `guided_decoding_backend` with `xgrammar` or `llguidance` when creating the LLM instance. For example,

```python
from tensorrt_llm import LLM

llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8", guided_decoding_backend="xgrammar")
```

### JSON Schema

Create a `GuidedDecodingParams` with the `json` field specified with a JSON schema, use it to create `SamplingParams`, and then pass to `llm.generate` or `llm.generate_async`. Alternatively, the JSON schema can be created using [pydantic](https://docs.pydantic.dev/latest/).

```python
from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams, GuidedDecodingParams

if __name__ == "__main__":
    llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8", guided_decoding_backend="xgrammar")

    json_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[\\w]+$"
            },
            "population": {
                "type": "integer"
            },
        },
        "required": ["name", "population"],
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Give me the information of the capital of France in the JSON format.",
        },
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm.generate(
        prompt,
        sampling_params=SamplingParams(max_tokens=256, guided_decoding=GuidedDecodingParams(json=json_schema)),
    )
    print(output.outputs[0].text)
```

The output would look like:
```txt
{
  "name": "Paris",
  "population": 2145206
}
```


### Regular expression

Create a `GuidedDecodingParams` with the `regex` field specified with a regular expression, use it to create `SamplingParams`, and then pass to `llm.generate` or `llm.generate_async`.

```python
from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams, GuidedDecodingParams

if __name__ == "__main__":
    llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8", guided_decoding_backend="xgrammar")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm.generate(
        prompt,
        sampling_params=SamplingParams(max_tokens=256, guided_decoding=GuidedDecodingParams(regex="(Paris|London)")),
    )
    print(output.outputs[0].text)
```

The output would look like:
```txt
Paris
```

### EBNF grammar

Create a `GuidedDecodingParams` with the `grammar` field specified with an EBNF grammar, use it to create `SamplingParams`, and then pass to `llm.generate` or `llm.generate_async`.

```python
from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams, GuidedDecodingParams

if __name__ == "__main__":
    llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8", guided_decoding_backend="xgrammar")

    ebnf_grammar = """root ::= description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful geography bot."
        },
        {
            "role": "user",
            "content": "Give me the information of the capital of France.",
        },
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm.generate(
        prompt,
        sampling_params=SamplingParams(max_tokens=256, guided_decoding=GuidedDecodingParams(grammar=ebnf_grammar)),
    )
    print(output.outputs[0].text)
```

The output would look like:
```txt
Paris is the capital of France
```

### Structural tag

Create a `GuidedDecodingParams` with the `structural_tag` field specified with a structural tag string, use it to create `SamplingParams`, and then pass to `llm.generate` or `llm.generate_async`.

Structural tag is supported by `xgrammar` backend only. It is a powerful and flexible tool to represent the LLM output constraints. Please see [structural tag usage](https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html) for a comprehensive tutorial. Below is an example of function calling with customized function call format for `Llama-3.1-8B-Instruct`.

```python
import json
from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams, GuidedDecodingParams

if __name__ == "__main__":
    llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8", guided_decoding_backend="xgrammar")

    tool_get_current_weather = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }

    tool_get_current_date = {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time for a given timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                    }
                },
                "required": ["timezone"],
            },
        },
    }

    system_prompt = f"""# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant."""
    user_prompt = "You are in New York. Please get the current date and time, and the weather."
    structural_tag = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "begin": "<function=get_current_weather>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool_get_current_weather["function"]["parameters"]
                    },
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool_get_current_date["function"]["parameters"]
                    },
                    "end": "</function>",
                },
            ],
        },
    }

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm.generate(
        prompt,
        sampling_params=SamplingParams(max_tokens=256, guided_decoding=GuidedDecodingParams(structural_tag=json.dumps(structural_tag))),
    )
    print(output.outputs[0].text)
```

The output would look like:
```txt
<function=get_current_date>{"timezone": "America/New_York"}</function>
<function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>
```
