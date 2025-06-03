# Backend Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for Redis to be ready if REDIS_URL is set\n\
if [ ! -z "$REDIS_URL" ]; then\n\
    echo "Waiting for Redis..."\n\
    while ! python -c "import redis; r=redis.from_url(\"$REDIS_URL\"); r.ping()" 2>/dev/null; do\n\
        sleep 1\n\
    done\n\
    echo "Redis is ready!"\n\
fi\n\
\n\
# Initialize database\n\
python -c "from models.database import init_db; init_db()"\n\
\n\
# Start the application\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
