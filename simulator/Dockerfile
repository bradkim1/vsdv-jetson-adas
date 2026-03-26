FROM ros:humble
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install kuksa-client grpcio && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY vehicle_simulator.py .
ENV PYTHONUNBUFFERED=1
CMD ["python3", "-u", "vehicle_simulator.py"]
