#!/bin/bash

# Wait for Ollama to be ready
echo "Waiting for Ollama service to be ready..."
until curl -s -o /dev/null -w "%{http_code}" http://ollama:11434/api/health | grep -q "200"; do
    sleep 5
    echo "Still waiting for Ollama service..."
done
echo "Ollama service is ready!"

# Pull the models we serving
echo "Pulling models..."
curl -X POST http://ollama:11434/api/pull -d '{"name": "qwen2.5-coder:3b"}'
curl -X POST http://ollama:11434/api/pull -d '{"name": "qwen2.5:3b"}'
curl -X POST http://ollama:11434/api/pull -d '{"name": "deepseek-r1:1.5b"}'

echo "Models are ready!"
