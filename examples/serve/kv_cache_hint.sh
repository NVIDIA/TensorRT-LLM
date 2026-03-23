#!/usr/bin/env bash

# ------------------------------
# Build the exact prompt (20 repeats)
# ------------------------------

unit="One Two Three Four Five Six Seven Eight Nine Ten "
prompt=""

for i in {1..20}; do
  prompt+="$unit"
done

echo "=== FIRST GENERATION ==="

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
        \"model\": \"Qwen3-8B\",
        \"messages\": [
          {\"role\": \"system\", \"content\": \"$prompt\"},
          {\"role\": \"user\", \"content\": \"$prompt\"}
        ],
        \"max_tokens\": 1024,
        \"temperature\": 0.8,
        \"top_p\": 0.95
      }"


echo
echo "=== END FIRST GENERATION ==="
echo

echo
echo "=== TRUNCATE KV CACHE ==="
echo

curl http://localhost:8000/_control/kv_cache/truncate \
  -H "Content-Type: application/json" \
  -d "{
        \"model\": \"Qwen3-8B\",
        \"messages\": [
          {\"role\": \"system\", \"content\": \"$prompt\"},
          {\"role\": \"user\", \"content\": \"$prompt\"}
        ],
        \"messages_to_retain\": [
          {\"role\": \"system\", \"content\": \"$prompt\"}
        ]
      }"


# ------------------------------
# Second generation request
# (KV cache hints cannot be sent over HTTP, so this repeats the request)
# ------------------------------

echo "=== SECOND GENERATION ==="

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
        \"model\": \"Qwen3-8B\",
        \"messages\": [
          {\"role\": \"system\", \"content\": \"$unit\"},
          {\"role\": \"user\", \"content\": \"$prompt\"}
        ],
        \"max_tokens\": 1024,
        \"temperature\": 0.8,
        \"top_p\": 0.95
      }"

echo
echo "=== END SECOND GENERATION ==="
