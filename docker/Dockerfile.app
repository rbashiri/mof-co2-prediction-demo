ARG BASE_IMAGE=mofid-base:demo
FROM ${BASE_IMAGE}

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
COPY configs/config.yaml /app/configs/config.yaml
COPY results/models/ /app/results/models/
COPY mof_models.py /app/mof_models.py

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501

CMD ["sh", "-c", "python -m streamlit run src/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
