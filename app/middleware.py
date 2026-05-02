"""Global body-size middleware — enforces DECISIONS.md §17's 25 MB cap.

Applies to every endpoint, not path-scoped to /document. /text is bounded by
the 3M-char text cap, /chat request bodies are tiny — the global cap costs
nothing on those routes and avoids per-path branching here.
"""

from collections.abc import Awaitable, Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.limits import MAX_UPLOAD_BYTES


class MaxBodySizeMiddleware:
    """Reject requests whose body exceeds MAX_UPLOAD_BYTES with 413.

    Cheap path: trust Content-Length when it's present and parses. Fallback
    path: when the header is missing or unparseable (chunked transfer
    encoding), wrap `receive` and abort once cumulative bytes exceed the cap.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = _parse_content_length(scope)
        if content_length is not None and content_length > self.max_bytes:
            await _send_413(send, content_length, self.max_bytes)
            return

        if content_length is None:
            # Header missing or unparseable — wrap receive to count bytes as
            # they arrive and abort once the cap is exceeded.
            await self.app(scope, _make_counting_receive(receive, send, self.max_bytes), send)
            return

        await self.app(scope, receive, send)


def _parse_content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _make_counting_receive(receive: Receive, send: Send, max_bytes: int) -> Receive:
    total = 0
    aborted = False

    async def counting_receive() -> Message:
        nonlocal total, aborted
        message = await receive()
        if aborted or message["type"] != "http.request":
            return message
        total += len(message.get("body", b""))
        if total > max_bytes:
            aborted = True
            await _send_413(send, total, max_bytes)
            # Drain remaining body so the client doesn't hang waiting for us
            # to ack. Returning a `more_body=False` empty chunk ends the
            # request stream cleanly from the app's perspective.
            return {"type": "http.request", "body": b"", "more_body": False}
        return message

    return counting_receive


async def _send_413(send: Send, got_bytes: int, max_bytes: int) -> None:
    got_mb = got_bytes / 1024 / 1024
    max_mb = max_bytes // 1024 // 1024
    detail = f"upload exceeds the {max_mb} MB limit (got {got_mb:.1f} MB)"
    body = f'{{"detail":"{detail}"}}'.encode()
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": body})
