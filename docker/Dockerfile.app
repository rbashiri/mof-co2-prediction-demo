FROM mofid-base:demo

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
COPY configs/config.yaml /app/configs/config.yaml
COPY results/models/ /app/results/models/
COPY mof_models.py /app/mof_models.py

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
