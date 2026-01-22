FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-cache --python 3.12 --group frontend

COPY src/toxic_comments/frontend.py .

EXPOSE 8501

ENTRYPOINT ["uv", "run", "streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]