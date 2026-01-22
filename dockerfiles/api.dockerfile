FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-cache --python 3.12 --group backend

COPY src/ src/
COPY README.md .

RUN uv sync --frozen --no-cache --group backend


RUN mkdir -p models/

# 6. Runtime Configuration
ENV PORT=8000
EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "src.toxic_comments.api:app", "--host", "0.0.0.0", "--port", "8000"]
 