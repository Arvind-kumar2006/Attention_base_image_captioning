# Backend image: CPU-only PyTorch serving the captioning API.
#
# The CPU wheel index is ~200 MB smaller than the default one, which pulls the
# CUDA runtime that would be dead weight on a CPU host.

FROM python:3.10-slim AS base

# Keep the image small and the log unbuffered so container logs stream.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is what torch needs for its OpenMP kernels; curl is for the
# container healthcheck. No compiler is installed: every wheel here has a
# manylinux build.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# python:3.10-slim ships pip 23.0.1, which rejects the CPU index outright:
# it serves typing_extensions and jinja2 wheels whose metadata name does not
# match the requested one ("inconsistent Name"), discards them, falls back to
# the typing_extensions *sdist*, and then cannot build it because the isolated
# build environment inherits the same --index-url and cannot find flit_core.
#
# The failure surfaces as "No matching distribution found for flit_core", which
# looks nothing like its cause. Newer pip normalises those names and resolves
# the wheels directly. Upgrading here is load-bearing, not housekeeping.
RUN python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Dependency layer first so a source-only change reuses the cached wheel layer.
COPY ml/requirements.txt /tmp/ml-requirements.txt
COPY backend/requirements.txt /tmp/backend-requirements.txt

# torch and torchvision are installed explicitly, before the requirements
# files. Two reasons:
#   1. The default index serves the CUDA build, which drags in ~2 GB of runtime
#      libraries this image cannot use.
#   2. --index-url REPLACES PyPI rather than adding to it. The pytorch index
#      proxies only torch's own dependency closure (pillow and numpy are there,
#      tqdm/matplotlib/nltk/fastapi/... are not), so resolving a whole
#      requirements file against it fails. Installing torch first means the
#      later PyPI pass sees the requirement already satisfied.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && python -m pip install -r /tmp/ml-requirements.txt \
    && python -m pip install -r /tmp/backend-requirements.txt

# The code. ml/ is installed editable so `import captioning` (what the backend
# uses) and `ml.app`-style relative imports resolve to the same source.
COPY ml /app/ml
COPY backend /app/backend

RUN python -m pip install --no-deps -e /app/ml

# Run as a non-root user. /app/uploads is the only path that needs writing;
# creating it here, owned by appuser, is also what gives the compose named
# volume the right ownership when Docker seeds it from this path.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/uploads \
    && chown -R appuser:appuser /app/uploads /app/ml /app/backend
USER appuser

ENV CHECKPOINT_PATH=/app/weights/best.pt \
    UPLOAD_DIR=/app/uploads \
    DEVICE=cpu \
    PYTHONPATH=/app

EXPOSE 8000

# Liveness, not model readiness. /health answers 200 with status "degraded" when
# no checkpoint is present, and nginx waits on this check -- gating on the model
# instead would mean a first run could never bring the stack up. The frontend
# reads model_loaded from the same payload and tells the user captions will 503.
# Make it strict only once a checkpoint is always mounted:
#   CMD curl -fsS http://127.0.0.1:8000/health | grep -q '"model_loaded":true'
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Two workers: torch already uses every core for a single inference, so this is
# for request concurrency, not for the maths.
#
# --forwarded-allow-ips "*" is safe only because compose does not publish this
# container's port; nginx is then the only possible client. If you ever add
# `ports:` to the api service, pin this to nginx's address instead.
CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]