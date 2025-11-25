curl http://localhost:8010/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "doesnt matter",
        "prompt": "What is 200023 + 342325?",
        "max_tokens": 50,
        "temperature": 0
    }' -w "\n"