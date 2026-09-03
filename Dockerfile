# Build stage: resolve dependencies into a virtualenv the runtime image copies.
FROM python:3.12-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch. Cloud Run has no GPU, and the default CUDA build is several
# gigabytes of wheels that would never be used.
RUN pip install --no-cache-dir \
    torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Dependencies come from pyproject.toml so the image cannot drift from the
# declared set. The static viewer and the bundled games install with the
# package via [tool.setuptools.package-data].
WORKDIR /build
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .


# Runtime stage: no build tooling, no source tree, no package index access.
FROM python:3.12-slim

COPY --from=builder /opt/venv /opt/venv

# The served weights. 1.7 MB, so baking them in is simpler and faster than
# fetching them at boot, and it keeps the image a single self-contained unit.
COPY data/checkpoints/transformer_model/epoch-10.pt /app/model/epoch-10.pt

ENV PATH="/opt/venv/bin:$PATH" \
    MAHJONG_MIND_CHECKPOINT=/app/model/epoch-10.pt \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as a non-root user; nothing here needs to write to the filesystem.
RUN useradd --create-home --uid 1000 app
USER app

# Cloud Run assigns the port through $PORT and routes HTTPS to it; 8080 is its
# default and only a hint here.
EXPOSE 8080

CMD ["python", "-m", "mahjong_mind.api.service"]
