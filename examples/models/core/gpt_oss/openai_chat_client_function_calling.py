import argparse
import json

from openai import OpenAI

tool_get_current_weather = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Gets the current weather in the provided location.",
        "parameters": {
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
    }
}

tool_get_multiple_weathers = {
    "type": "function",
    "function": {
        "name": "get_multiple_weathers",
        "description":
        "Gets the current weather in the provided list of locations.",
        "parameters": {
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
    }
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
            "role": "user",
            "content": args.prompt,
        },
    ]

    print(f"[USER PROMPT] {args.prompt}")
    chat_completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_completion_tokens=500,
        tools=[tool_get_current_weather, tool_get_multiple_weathers],
    )
    tools = {
        "get_current_weather": get_current_weather,
        "get_multiple_weathers": get_multiple_weathers
    }
    message = chat_completion.choices[0].message
    assert message, "Empty Message"
    assert message.tool_calls, "Empty tool calls"
    assert message.content is None, "Empty content expected"
    reasoning = message.reasoning if hasattr(message, "reasoning") else None
    tool_call = message.tool_calls[0]
    func_name = tool_call.function.name
    assert func_name in tools, "Invalid function name"
    kwargs = json.loads(tool_call.function.arguments)

    tool = tools[func_name]
    print(f"[RESPONSE 1] [COT] {reasoning}")
    print(f"[RESPONSE 1] [FUNCTION CALL] {tool.__name__}(**{kwargs})")
    answer = tool(**kwargs)

    messages.extend([{
        "role": "assistant",
        "reasoning": reasoning,
        "tool_calls": [tool_call],
    }, {
        "role": "tool",
        "content": json.dumps(answer),
        "tool_call_id": tool_call.id
    }])

    chat_completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_completion_tokens=500,
    )

    response_text = chat_completion.choices[0].message.content
    print(f"[RESPONSE 2] {response_text}")


if __name__ == "__main__":
    main()
