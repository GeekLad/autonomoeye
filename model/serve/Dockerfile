FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY --chown=root:root . .

RUN mkdir -p /root/.cache/torch/hub/checkpoints \
    && curl https://download.pytorch.org/models/mobilenet_v2-b0353104.pth --output /root/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENV MODEL_WEIGHTS_PATH=/app
ENV NMS_THRESHOLD=0.1

ENTRYPOINT ["python3", "app.py"]