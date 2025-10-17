curl http://localhost:8400/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Wgaadfasl;fdj;",
        "prompt": "What type of LLM are you?",
        "max_tokens": 15,
        "temperature": 100
    }' -w "\n"