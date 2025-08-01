### :section Customization
### :title Generate text with guided decoding
### :order 0
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import GuidedDecodingParams


def main():

    # Specify the guided decoding backend; xgrammar and llguidance are supported currently.
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              guided_decoding_backend='xgrammar')

    # An example from json-mode-eval
    schema = '{"title": "WirelessAccessPoint", "type": "object", "properties": {"ssid": {"title": "SSID", "type": "string"}, "securityProtocol": {"title": "SecurityProtocol", "type": "string"}, "bandwidth": {"title": "Bandwidth", "type": "string"}}, "required": ["ssid", "securityProtocol", "bandwidth"]}'

    prompt = [{
        'role':
        'system',
        'content':
        "You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{'title': 'WirelessAccessPoint', 'type': 'object', 'properties': {'ssid': {'title': 'SSID', 'type': 'string'}, 'securityProtocol': {'title': 'SecurityProtocol', 'type': 'string'}, 'bandwidth': {'title': 'Bandwidth', 'type': 'string'}}, 'required': ['ssid', 'securityProtocol', 'bandwidth']}\n</schema>\n"
    }, {
        'role':
        'user',
        'content':
        "I'm currently configuring a wireless access point for our office network and I need to generate a JSON object that accurately represents its settings. The access point's SSID should be 'OfficeNetSecure', it uses WPA2-Enterprise as its security protocol, and it's capable of a bandwidth of up to 1300 Mbps on the 5 GHz band. This JSON object will be used to document our network configurations and to automate the setup process for additional access points in the future. Please provide a JSON object that includes these details."
    }]
    prompt = llm.tokenizer.apply_chat_template(prompt, tokenize=False)
    print(f"Prompt: {prompt!r}")

    output = llm.generate(prompt, sampling_params=SamplingParams(max_tokens=50))
    print(f"Generated text (unguided): {output.outputs[0].text!r}")

    output = llm.generate(
        prompt,
        sampling_params=SamplingParams(
            max_tokens=50, guided_decoding=GuidedDecodingParams(json=schema)))
    print(f"Generated text (guided): {output.outputs[0].text!r}")

    # Got output like
    # Prompt: "<|system|>\nYou are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{'title': 'WirelessAccessPoint', 'type': 'object', 'properties': {'ssid': {'title': 'SSID', 'type': 'string'}, 'securityProtocol': {'title': 'SecurityProtocol', 'type': 'string'}, 'bandwidth': {'title': 'Bandwidth', 'type': 'string'}}, 'required': ['ssid', 'securityProtocol', 'bandwidth']}\n</schema>\n</s>\n<|user|>\nI'm currently configuring a wireless access point for our office network and I need to generate a JSON object that accurately represents its settings. The access point's SSID should be 'OfficeNetSecure', it uses WPA2-Enterprise as its security protocol, and it's capable of a bandwidth of up to 1300 Mbps on the 5 GHz band. This JSON object will be used to document our network configurations and to automate the setup process for additional access points in the future. Please provide a JSON object that includes these details.</s>\n"
    # Generated text (unguided): '<|assistant|>\nHere\'s a JSON object that accurately represents the settings of a wireless access point for our office network:\n\n```json\n{\n  "title": "WirelessAccessPoint",\n  "'
    # Generated text (guided): '{"ssid": "OfficeNetSecure", "securityProtocol": "WPA2-Enterprise", "bandwidth": "1300 Mbps"}'


if __name__ == '__main__':
    main()
