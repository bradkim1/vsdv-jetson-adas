#!/usr/bin/env python3
"""
vSDV Vehicle Simulator — Host (RTX 5090)
Publishes realistic vehicle dynamics to Kuksa Databroker on Jetson via gRPC.
"""

import asyncio
import math
import time
import grpc

KUKSA_HOST = "192.168.55.1"
KUKSA_PORT = 55555

SIGNALS = {
    "Vehicle.Speed":                          "float",
    "Vehicle.Powertrain.ElectricMotor.Speed": "int32",
    "Vehicle.Chassis.SteeringWheel.Angle":    "float",
    "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current": "float",
    "Vehicle.CurrentLocation.Latitude":       "double",
    "Vehicle.CurrentLocation.Longitude":      "double",
    "Vehicle.CurrentLocation.Speed":          "double",
}


async def set_value(stub, path, value, sig_type):
    from kuksa.val.v1 import val_pb2, types_pb2
    dp = types_pb2.Datapoint()
    if sig_type == "float":
        dp.float = float(value)
    elif sig_type == "double":
        dp.double = float(value)
    elif sig_type == "int32":
        dp.int32 = int(value)
    entry = val_pb2.EntryUpdate(
        entry=types_pb2.DataEntry(path=path, value=dp),
        fields=[types_pb2.Field.FIELD_VALUE],
    )
    await stub.Set(val_pb2.SetRequest(updates=[entry]))


async def simulate(stub):
    t = 0
    lat, lon = 37.7749, -122.4194
    print(f"[Simulator] Publishing to Kuksa @ {KUKSA_HOST}:{KUKSA_PORT}\n")
    while True:
        speed_kmh = max(0, 40 + 40 * math.sin(t * 0.1))
        updates = {
            "Vehicle.Speed":                          speed_kmh,
            "Vehicle.Powertrain.ElectricMotor.Speed": int(speed_kmh * 80 + 800),
            "Vehicle.Chassis.SteeringWheel.Angle":    15 * math.sin(t * 0.05),
            "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current": max(20.0, 85.0 - t * 0.005),
            "Vehicle.CurrentLocation.Latitude":       lat + t * 0.000005,
            "Vehicle.CurrentLocation.Longitude":      lon + t * 0.000003,
            "Vehicle.CurrentLocation.Speed":          speed_kmh / 3.6,
        }
        for path, value in updates.items():
            try:
                await set_value(stub, path, value, SIGNALS[path])
            except Exception as e:
                print(f"[ERROR] {path}: {e}")
        print(f"[t={t:5.1f}s] Speed={speed_kmh:5.1f}km/h  RPM={int(speed_kmh*80+800):5d}")
        t += 1
        await asyncio.sleep(1.0)


async def main():
    channel = grpc.aio.insecure_channel(f"{KUKSA_HOST}:{KUKSA_PORT}")
    from kuksa.val.v1 import val_pb2_grpc
    stub = val_pb2_grpc.VALStub(channel)
    try:
        await simulate(stub)
    except KeyboardInterrupt:
        print("\n[Simulator] Stopped.")
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(main())
