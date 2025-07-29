import argparse
import json
import re

from openai import OpenAI

system_prompt = """You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'."""

developer_prompt = """# Instructions

Use a friendly tone.

# Tools

## functions

namespace functions {

// Gets the location of the user.
type get_location = () => any;

// Gets the current weather in the provided location.
type get_current_weather = (_: {
// The city and state, e.g. San Francisco, CA
location: string,
format?: "celsius" | "fahrenheit", // default: celsius
}) => any;

// Gets the current weather in the provided list of locations.
type get_multiple_weathers = (_: {
// List of city and state, e.g. ["San Francisco, CA", "New York, NY"]
locations: string[],
format?: "celsius" | "fahrenheit", // default: celsius
}) => any;

} // namespace functions"""

schema_get_current_weather = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "format": {
            "type": "string",
            "description": "default: celsius",
            "enum": ["celsius", "fahrenheit"],
        },
    },
    "required": ["location"],
}

schema_get_multiple_weathers = {
    "type": "object",
    "properties": {
        "locations": {
            "type":
            "array",
            "items": {
                "type": "string"
            },
            "description":
            'List of city and state, e.g. ["San Francisco, CA", "New York, NY"]',
        },
        "format": {
            "type": "string",
            "description": "default: celsius",
            "enum": ["celsius", "fahrenheit"],
        },
    },
    "required": ["locations"],
}


def get_current_weather(location: str, format: str = "celsius") -> dict:
    return {"sunny": True, "temperature": 20 if format == "celsius" else 68}


def get_multiple_weathers(locations: list[str],
                          format: str = "celsius") -> list[dict]:
    return [get_current_weather(location, format) for location in locations]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt",
                        type=str,
                        default="What is the weather like in SF?")
    args = parser.parse_args()

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="tensorrt_llm",
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "developer",
            "content": developer_prompt,
        },
        {
            "role": "user",
            "content": args.prompt,
        },
    ]

    print(f"[USER PROMPT] {args.prompt}")
    chat_completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_completion_tokens=500,
        response_format={
            "type":
            "structural_tag",
            "structures": [{
                "begin":
                "<|channel|>commentary to=get_current_weather <|constrain|>json<|message|>",
                "schema": schema_get_current_weather,
                "end": "<|call|>",
            }, {
                "begin":
                "<|channel|>commentary to=get_multiple_weathers <|constrain|>json<|message|>",
                "schema": schema_get_multiple_weathers,
                "end": "<|call|>",
            }],
            "triggers": ["<|channel|>commentary to="],
        },
        stop=["<|call|>"],
        extra_body={
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
        },
    )

    response_text = chat_completion.choices[0].message.content
    print(f"[RESPONSE 1] {response_text}")

    for regex, tool in [
        (r"(<\|channel\|>commentary to=get_current_weather <\|constrain\|>json<\|message\|>)([\S\s]+)(<\|call\|>)",
         get_current_weather),
        (r"(<\|channel\|>commentary to=get_multiple_weathers <\|constrain\|>json<\|message\|>)([\S\s]+)(<\|call\|>)",
         get_multiple_weathers)
    ]:
        match = re.search(regex, response_text)
        if match is not None:
            break
    else:
        print("Failed to call functions, exiting...")
        return

    kwargs = json.loads(match.group(2))
    print(f"[FUNCTION CALL] {tool.__name__}(**{kwargs})")
    answer = tool(**kwargs)

    messages.extend([{
        "role": "assistant",
        "content": match.group(0),
    }, {
        "role": f"{tool.__name__} to=assistant",
        "content": json.dumps(answer),
    }])

    chat_completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_completion_tokens=500,
        extra_body={
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
        },
    )

    response_text = chat_completion.choices[0].message.content
    print(f"[RESPONSE 2] {response_text}")


if __name__ == "__main__":
    main()
