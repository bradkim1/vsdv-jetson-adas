#!/usr/bin/env python3
"""
VSS access layer for the vSDV scenario-search agent.

Thin wrapper over the Kuksa val.v1 gRPC API so the rest of the harness never
touches protobufs. Targets the databroker on the Jetson (192.168.55.1:55555)
over the USB-gadget net.
"""

import asyncio
import os

import grpc
from kuksa.val.v1 import val_pb2, val_pb2_grpc, types_pb2

KUKSA_HOST = os.environ.get("KUKSA_HOST", "192.168.55.1")
KUKSA_PORT = int(os.environ.get("KUKSA_PORT", "55555"))


def _target() -> str:
    return f"{KUKSA_HOST}:{KUKSA_PORT}"


# ── value coercion ────────────────────────────────────────────────────────────

def _unwrap(dp: types_pb2.Datapoint):
    """Datapoint → python value, whichever oneof field is populated."""
    which = dp.WhichOneof("value")
    if which is None:
        return None
    val = getattr(dp, which)
    # Repeated array types come back as protobuf containers
    if which.endswith("_array"):
        return list(val.values)
    return val


def _wrap(value, datatype: int) -> types_pb2.Datapoint:
    """python value + VSS datatype enum → Datapoint."""
    dp = types_pb2.Datapoint()
    DT = types_pb2.DataType
    if datatype == DT.DATA_TYPE_BOOLEAN:
        dp.bool = bool(value)
    elif datatype in (DT.DATA_TYPE_INT8, DT.DATA_TYPE_INT16, DT.DATA_TYPE_INT32):
        dp.int32 = int(value)
    elif datatype == DT.DATA_TYPE_INT64:
        dp.int64 = int(value)
    elif datatype in (DT.DATA_TYPE_UINT8, DT.DATA_TYPE_UINT16, DT.DATA_TYPE_UINT32):
        dp.uint32 = int(value)
    elif datatype == DT.DATA_TYPE_UINT64:
        dp.uint64 = int(value)
    elif datatype == DT.DATA_TYPE_FLOAT:
        dp.float = float(value)
    elif datatype == DT.DATA_TYPE_DOUBLE:
        dp.double = float(value)
    elif datatype == DT.DATA_TYPE_STRING:
        dp.string = str(value)
    else:
        raise ValueError(f"unsupported VSS datatype enum {datatype}")
    return dp


# ── async client ──────────────────────────────────────────────────────────────

class VSS:
    """Async VSS client. Caches metadata so callers never pass datatypes."""

    def __init__(self, target: str | None = None):
        self._target = target or _target()
        self._channel = None
        self._stub = None
        self._meta: dict[str, int] = {}

    async def __aenter__(self):
        self._channel = grpc.aio.insecure_channel(self._target)
        self._stub = val_pb2_grpc.VALStub(self._channel)
        return self

    async def __aexit__(self, *exc):
        if self._channel is not None:
            await self._channel.close()

    async def metadata(self, pattern: str = "Vehicle.**") -> dict[str, dict]:
        """Resolve a VSS path pattern → {path: {datatype, type, unit, description}}."""
        req = val_pb2.GetRequest(entries=[
            val_pb2.EntryRequest(
                path=pattern,
                view=types_pb2.View.VIEW_METADATA,
                fields=[types_pb2.Field.FIELD_METADATA],
            )
        ])
        resp = await self._stub.Get(req)
        out = {}
        for e in resp.entries:
            m = e.metadata
            out[e.path] = {
                "datatype": m.data_type,
                "datatype_name": types_pb2.DataType.Name(m.data_type),
                "entry_type": types_pb2.EntryType.Name(m.entry_type),
                "unit": m.unit,
                "description": m.description,
            }
            self._meta[e.path] = m.data_type
        return out

    async def _datatype(self, path: str) -> int:
        if path not in self._meta:
            await self.metadata(path)
        if path not in self._meta:
            raise KeyError(f"{path} is not in the databroker's VSS tree")
        return self._meta[path]

    async def get(self, *paths: str) -> dict:
        """Read current values. Missing/unset signals come back as None."""
        req = val_pb2.GetRequest(entries=[
            val_pb2.EntryRequest(
                path=p,
                view=types_pb2.View.VIEW_CURRENT_VALUE,
                fields=[types_pb2.Field.FIELD_VALUE],
            ) for p in paths
        ])
        resp = await self._stub.Get(req)
        out = {p: None for p in paths}
        for e in resp.entries:
            out[e.path] = _unwrap(e.value)
        return out

    async def set(self, values: dict, actuator: bool = False) -> None:
        """
        Write sensor values (default) or actuator targets.

        Sensors are published with FIELD_VALUE; actuators must use
        FIELD_ACTUATOR_TARGET or the databroker rejects the write.
        """
        updates = []
        for path, value in values.items():
            dt = await self._datatype(path)
            dp = _wrap(value, dt)
            if actuator:
                updates.append(val_pb2.EntryUpdate(
                    entry=types_pb2.DataEntry(path=path, actuator_target=dp),
                    fields=[types_pb2.Field.FIELD_ACTUATOR_TARGET],
                ))
            else:
                updates.append(val_pb2.EntryUpdate(
                    entry=types_pb2.DataEntry(path=path, value=dp),
                    fields=[types_pb2.Field.FIELD_VALUE],
                ))
        resp = await self._stub.Set(val_pb2.SetRequest(updates=updates))
        if resp.errors:
            details = "; ".join(
                f"{e.path}: {e.error.code} {e.error.reason} {e.error.message}"
                for e in resp.errors
            )
            raise RuntimeError(f"VSS set rejected — {details}")

    async def subscribe(self, paths: list[str], fields=None):
        """
        Async-iterate VSS updates: yields {path: value} deltas.

        The databroker pushes one notification per changed entry set; we yield
        only the paths present in that notification, not a full snapshot.
        """
        fields = fields or [types_pb2.Field.FIELD_VALUE]
        req = val_pb2.SubscribeRequest(entries=[
            val_pb2.SubscribeEntry(
                path=p,
                view=types_pb2.View.VIEW_CURRENT_VALUE,
                fields=fields,
            ) for p in paths
        ])
        async for resp in self._stub.Subscribe(req):
            delta = {}
            for u in resp.updates:
                delta[u.entry.path] = _unwrap(u.entry.value)
            if delta:
                yield delta


# ── sync convenience, for one-shot CLI probes ─────────────────────────────────

def probe(pattern: str = "Vehicle.ADAS.**") -> dict:
    async def _run():
        async with VSS() as v:
            return await v.metadata(pattern)
    return asyncio.run(_run())


if __name__ == "__main__":
    import sys
    pat = sys.argv[1] if len(sys.argv) > 1 else "Vehicle.ADAS.**"
    meta = probe(pat)
    print(f"{len(meta)} entries matching {pat!r} on {_target()}\n")
    for path in sorted(meta):
        m = meta[path]
        print(f"  {path:<58} {m['datatype_name']:<22} {m['entry_type']:<22} {m['unit']}")
