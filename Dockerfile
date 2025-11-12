# ---- Base image ----
FROM python:3.10-slim

# ---- System setup ----
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Copy project and install dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---- Default command ----
CMD ["python", "src/train.py"]
