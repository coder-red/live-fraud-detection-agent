# Stage 1: Build stage
FROM ghcr.io/astral-sh/uv:latest AS builder

WORKDIR /app

# Copy only the files needed for dependency resolution first
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python --no-cache-dir -r requirements.txt && \
    uv sync --frozen --no-dev --python /app/.venv/bin/python

# Copy application source code
COPY . .

# Stage 2: Runtime stage
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy Python environment and installed packages from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/ /app/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV HOME=/app

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]