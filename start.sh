#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Start the application with docker-compose
docker-compose up -d

echo "Starting services..."
echo "This might take a while as the Ollama models are being downloaded..."
echo "You can check progress with: docker-compose logs -f init-models"
echo ""
echo "Once startup is complete, access the application at: http://localhost:8501" 