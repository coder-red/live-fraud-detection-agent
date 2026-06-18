import asyncio
import json
from typing import Any


class EventManager:
    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue):
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def broadcast(self, event: str, data: dict[str, Any]):
        payload = json.dumps({"event": event, "data": data}, default=str)
        for q in self._subscribers:
            await q.put(payload)


events = EventManager()
