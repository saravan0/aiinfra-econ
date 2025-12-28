FROM python:3.11-slim

# ---- system deps (only what is needed) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- working dir ----
WORKDIR /app

# ---- install python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- copy project ----
COPY . .

# ---- default command ----
CMD ["bash", "run_pipeline.sh"]
