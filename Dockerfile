FROM python:3.11.15-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev
COPY predict.py ./
COPY models/ ./models/
COPY web/ ./web/
RUN useradd --create-home appuser
USER appuser
EXPOSE 9696
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '9696') + '/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}"]
