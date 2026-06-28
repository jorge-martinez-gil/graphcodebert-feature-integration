# Reproducible CPU environment for the FeatFuse benchmark.
#   docker build -t featfuse .
#   docker run --rm featfuse featfuse run -c configs/smoke.yaml
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e ".[dev]"

# Validate the install at build time (fails fast if anything is broken).
RUN featfuse info && pytest -q

ENTRYPOINT ["featfuse"]
CMD ["info"]
