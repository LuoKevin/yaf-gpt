from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import HTTPException, Request, status

_RATE_LIMIT_WINDOW_SECONDS = 60
_store_lock = threading.Lock()
_request_store: dict[tuple[str, str], deque[float]] = defaultdict(deque)


def clear_rate_limit_store() -> None:
    with _store_lock:
        _request_store.clear()


def limit_requests(*, bucket: str, max_requests: int, window_seconds: int = _RATE_LIMIT_WINDOW_SECONDS) -> Callable[[Request], None]:
    if max_requests <= 0:
        raise ValueError("max_requests must be positive.")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")

    def dependency(request: Request) -> None:
        client_id = _get_client_id(request)
        now = time.monotonic()
        key = (bucket, client_id)

        with _store_lock:
            timestamps = _request_store[key]
            while timestamps and now - timestamps[0] >= window_seconds:
                timestamps.popleft()

            if len(timestamps) >= max_requests:
                retry_after = max(1, int(window_seconds - (now - timestamps[0])))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please wait a moment and try again.",
                    headers={"Retry-After": str(retry_after)},
                )

            timestamps.append(now)

    return dependency


def _get_client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        first_hop = forwarded_for.split(",")[0].strip()
        if first_hop:
            return first_hop

    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip

    client = request.client
    if client and client.host:
        return client.host

    return "unknown"
