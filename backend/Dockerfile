# Multi-stage Dockerfile for Text Classification Backend
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    default-mysql-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Make sure Python packages are in PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/data /app/logs \
    && chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default environment variables
ENV DB_TYPE=mysql
ENV MYSQL_HOST=mysql
ENV MYSQL_PORT=3306
ENV MYSQL_USER=app_user
ENV MYSQL_DATABASE=text_classification
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create optimized startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Logging function\n\
log() {\n\
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*"\n\
}\n\
\n\
# Function to wait for MySQL\n\
wait_for_mysql() {\n\
    if [ "$DB_TYPE" = "mysql" ]; then\n\
        log "Waiting for MySQL at $MYSQL_HOST:$MYSQL_PORT..."\n\
        local timeout=60\n\
        local count=0\n\
        \n\
        while ! mysqladmin ping -h"$MYSQL_HOST" -P"$MYSQL_PORT" -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" --silent 2>/dev/null; do\n\
            if [ $count -gt $timeout ]; then\n\
                log "ERROR: MySQL connection timeout after ${timeout}s"\n\
                exit 1\n\
            fi\n\
            log "MySQL is unavailable - sleeping (${count}/${timeout}s)"\n\
            sleep 2\n\
            count=$((count + 2))\n\
        done\n\
        log "MySQL is ready!"\n\
    fi\n\
}\n\
\n\
# Function to wait for Redis\n\
wait_for_redis() {\n\
    if [ ! -z "$REDIS_HOST" ]; then\n\
        log "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."\n\
        local timeout=30\n\
        local count=0\n\
        \n\
        while ! python -c "import redis; r=redis.Redis(host=\"$REDIS_HOST\", port=$REDIS_PORT); r.ping()" 2>/dev/null; do\n\
            if [ $count -gt $timeout ]; then\n\
                log "ERROR: Redis connection timeout after ${timeout}s"\n\
                exit 1\n\
            fi\n\
            log "Redis is unavailable - sleeping (${count}/${timeout}s)"\n\
            sleep 1\n\
            count=$((count + 1))\n\
        done\n\
        log "Redis is ready!"\n\
    fi\n\
}\n\
\n\
# Wait for dependencies\n\
wait_for_mysql\n\
wait_for_redis\n\
\n\
# Initialize database\n\
log "Initializing database..."\n\
python -c "from models.database import init_db; init_db()"\n\
log "Database initialized successfully!"\n\
\n\
# Start the application\n\
log "Starting application with command: $*"\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command for API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
