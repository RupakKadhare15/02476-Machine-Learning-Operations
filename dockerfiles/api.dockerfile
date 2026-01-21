FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY lightning_logs lightning_logs/ 

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.toxic_comments.api:app", "--host", "0.0.0.0", "--port", "8000"] 
 