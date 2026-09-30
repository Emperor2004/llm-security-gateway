# Multi-stage Dockerfile for LLM Security Gateway
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Create non-privileged service user
RUN useradd -m -u 10001 gatewayuser && \
    mkdir -p /app/logs /app/models && \
    chown -R gatewayuser:gatewayuser /app

# Copy python packages from builder
COPY --from=builder /root/.local /home/gatewayuser/.local
ENV PATH=/home/gatewayuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy application files
COPY --chown=gatewayuser:gatewayuser app/ ./app/
COPY --chown=gatewayuser:gatewayuser dashboard/ ./dashboard/
COPY --chown=gatewayuser:gatewayuser config.yaml .env.example ./
COPY --chown=gatewayuser:gatewayuser params.yaml ./

# Switch to non-root user
USER gatewayuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/healthz').getcode() == 200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
