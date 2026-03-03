FROM python:3.10-slim

WORKDIR /app

# Install ffmpeg for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# HF Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 appuser

# Give the user write access to /tmp for model caching
ENV HF_HOME=/tmp/hf_home

USER appuser

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
