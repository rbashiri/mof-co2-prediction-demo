ARG BASE_IMAGE=mofid-base:demo

# Build the React frontend in a lightweight Node stage.
FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json /frontend/
RUN npm config set fetch-retries 5 && \
    npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000 && \
    npm ci --no-audit --no-fund

COPY frontend/ /frontend/
RUN npm run build

# Runtime image: native MOFid + Python + FastAPI + static frontend.
FROM ${BASE_IMAGE}

WORKDIR /app

COPY scripts/install_zeopp.py /app/scripts/install_zeopp.py
RUN python /app/scripts/install_zeopp.py --destination /opt/zeopp
ENV ZEO_NETWORK_PATH=/opt/zeopp/network \
    GEMINI_MODEL=gemini-3.5-flash

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
COPY configs/config.yaml /app/configs/config.yaml
COPY results/models/ /app/results/models/
COPY mof_models.py /app/mof_models.py

COPY --from=frontend-builder /frontend/dist /app/frontend/dist

EXPOSE 8501

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8501}"]
