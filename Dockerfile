FROM python:3.8

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg

ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --locked --no-dev

COPY . /app/

CMD ["streamlit", "run", "home_page.py", "--server.port", "8080"]
