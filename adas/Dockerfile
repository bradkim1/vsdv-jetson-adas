# ADAS Inference — Jetson Nano (ARM64, L4T base with CUDA + TensorRT)
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

RUN apt-get update && apt-get install -y python3-pip libopencv-dev python3-opencv && \
    pip3 install --no-cache-dir kuksa-client grpcio ultralytics && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY adas_inference.py .

ENV PYTHONUNBUFFERED=1
CMD ["python3", "-u", "adas_inference.py"]
