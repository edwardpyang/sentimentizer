FROM python:3.12-slim AS builder

ARG EXTRAS="ray,embeddings"

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

# Copy project files and the resolved dependency graph
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY sentimentizer/ ./sentimentizer/
COPY workflows/ ./workflows/

# Export exact locked versions for the requested extras. PyTorch is pruned from
# that export and installed separately from its CPU-only index; otherwise the
# PyPI lock graph also installs CUDA runtime packages. The project itself is
# installed without dependency resolution so the container cannot drift beyond
# uv.lock.
RUN set -eu; \
    extra_args=""; \
    old_ifs="$IFS"; \
    IFS=","; \
    for extra in $EXTRAS; do \
        extra_args="$extra_args --extra $extra"; \
    done; \
    IFS="$old_ifs"; \
    uv export --frozen --no-dev --no-hashes --no-emit-project \
        --prune torch \
        --format requirements-txt $extra_args \
        --output-file /tmp/requirements.txt; \
    torch_version="$(python -c 'import tomllib; data = tomllib.load(open("uv.lock", "rb")); print(next(package["version"] for package in data["package"] if package["name"] == "torch"))')"; \
    uv pip install --system --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==$torch_version"; \
    uv pip install --system --no-cache-dir \
        --requirement /tmp/requirements.txt; \
    uv pip install --system --no-cache-dir --no-deps .

# --- Runtime stage ---
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (K8s runAsNonRoot compliance)
RUN useradd -r -s /bin/false -d /app sentimentizer \
    && mkdir -p /app/.cache/huggingface \
    && chown -R sentimentizer:sentimentizer /app

# Copy installed packages and app from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

USER sentimentizer

ENV HF_HOME=/app/.cache/huggingface

# Ray Serve default port
EXPOSE 8000
# Ray dashboard (optional)
EXPOSE 8265

CMD ["python", "-m", "sentimentizer.serve", "--host", "0.0.0.0", "--port", "8000"]
