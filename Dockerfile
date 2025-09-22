FROM python:3.11-slim
ENV PIP_DEFAULT_TIMEOUT=1200 PIP_RETRIES=20 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt requirements.txt
COPY constraints.txt constraints.txt
RUN python -m pip install --upgrade pip setuptools wheel
# pin CPU torch from official index (faster & stable)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -c constraints.txt torch==2.4.1
RUN pip install -r requirements.txt -c constraints.txt --prefer-binary

COPY . .
EXPOSE 8502
CMD ["streamlit","run","risk_analysis_agent/ui_streamlit.py","--server.port=8502","--server.headless=true"]
