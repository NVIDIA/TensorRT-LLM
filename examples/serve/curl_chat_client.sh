### Curl Chat Client

if [[ -z $BASE_URL ]]; then
    BASE_URL=http://localhost:8000/v1
fi

curl ${BASE_URL}/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": TinyLlama-1.1B-Chat-v1.0,
        "messages":[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Where is New York?"}],
        "max_tokens": 16,
        "temperature": 0
    }'
