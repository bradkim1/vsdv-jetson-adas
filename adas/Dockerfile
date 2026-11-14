# ADAS Perception Node — Jetson Orin Nano (ARM64, L4T base with CUDA + TensorRT)
#
# Build from the REPOSITORY ROOT, not from adas/ — the node imports the shared
# VSS layer from <repo>/agent:
#     docker build -f adas/Dockerfile -t vsdv-adas .
# docker-compose.jetson.yml sets the context accordingly.
#
# NOTE: this base is l4t r36.2.0 while the board currently runs L4T R36.4.7
# (JetPack 6.2). It still loads, but if TensorRT plan deserialisation fails,
# the version skew is the first thing to check.
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

RUN apt-get update && apt-get install -y python3-pip libopencv-dev python3-opencv && \
    pip3 install --no-cache-dir kuksa-client grpcio ultralytics && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY agent/vss.py            agent/
COPY adas/adas_inference.py  adas/

# The TensorRT engine is a build artefact, never baked into the image — it is
# device- and version-specific. It lives on the mounted volume (see ENGINE).
ENV PYTHONUNBUFFERED=1 \
    ENGINE=/app/models/yolov8n.engine

CMD ["python3", "-u", "adas/adas_inference.py"]
