@echo off
echo Creating data directory...
if not exist data mkdir data

echo Starting services...
docker-compose up -d

echo This might take a while as the Ollama models are being downloaded...
echo You can check progress with: docker-compose logs -f init-models
echo.
echo Once startup is complete, access the application at: http://localhost:8501 