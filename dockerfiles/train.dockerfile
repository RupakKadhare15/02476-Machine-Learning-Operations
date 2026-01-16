FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# 2. Install system dependencies ]
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Python dependencies
# Copy only lock files first to cache this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-cache --python 3.12

# 4. Copy Source Code AND Configs
COPY configs/ configs/
COPY src/ src/
COPY README.md .
COPY tasks.py .

# 5. Install the local project (so "from toxic_comments" works)
RUN uv sync --frozen --no-cache --python 3.12

# 6. Set Entrypoint
# We point to the script inside the src folder
ENTRYPOINT ["uv", "run", "invoke", "train"]