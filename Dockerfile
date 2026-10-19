FROM python:3.14.8-slim-trixie AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser \
    && mkdir /app && chown appuser:appuser /app
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
LABEL org.opencontainers.image.source="https://github.com/deleted04user/heart-disease-lab"

FROM base AS development
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --chown=appuser:appuser . .
USER appuser
RUN uv sync --locked --extra dev --no-editable
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "pytest", "--cov", "--cov-report=term-missing:skip-covered"]

FROM base AS runtime
COPY --chown=appuser:appuser deploy/requirements.txt /app/deploy/requirements.txt
USER appuser
RUN uv venv /app/.venv && uv pip sync --python /app/.venv/bin/python deploy/requirements.txt
COPY --chown=appuser:appuser . .
ENV PATH="/app/.venv/bin:$PATH" HEART_PUBLIC_DEMO=1 HEART_LOG_LEVEL=INFO
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=4 \
    CMD python -c "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200"
CMD ["python", "-m", "streamlit", "run", "deploy/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
