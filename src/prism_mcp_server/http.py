"""Authenticated loopback HTTP using only the Python standard library."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import hmac
import json
import math
import re
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from prism_mcp import ErrorCode, McpError, TransportRequest

if TYPE_CHECKING:
    from . import Endpoint, RequestHandler

_TOKEN = re.compile(r"[A-Za-z0-9._~+/-]+=*", re.ASCII)
_ORIGIN = re.compile(r"https?://(127\.0\.0\.1|\[::1\])(?::([1-9][0-9]{0,4}))?", re.ASCII)


def _valid_origin(origin: str) -> bool:
    match = _ORIGIN.fullmatch(origin)
    return match is not None and (match[2] is None or int(match[2]) <= 65535)


class LoopbackHttpTransport:
    """POST-only JSON-RPC on literal 127.0.0.1. Handler work stays on the caller's loop."""

    def __init__(
        self,
        *,
        port: int = 0,
        token: str | None = None,
        allowed_origins: list[str] | tuple[str, ...] | None = None,
        max_request_bytes: int = 1024 * 1024,
    ) -> None:
        if type(port) is not int or not 0 <= port <= 65535:
            raise McpError(ErrorCode.PROTOCOL_FAILURE, "Invalid HTTP port.")
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise McpError(ErrorCode.PROTOCOL_FAILURE, "Invalid HTTP request limit.")
        self._port = port
        self._configured_token = token
        self._configured_origins = None if allowed_origins is None else tuple(allowed_origins)
        self._max_request_bytes = max_request_bytes
        self._origins: set[str] = set()
        self._expected = b""
        self._server: _HttpServer | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handler: RequestHandler | None = None
        self._closed = False
        self._closing: asyncio.Task[None] | None = None
        self._lock = threading.Lock()
        self._sockets: set[socket.socket] = set()
        self._pending: set[concurrent.futures.Future[Any]] = set()

    async def listen(self, handler: RequestHandler) -> Endpoint:
        if self._closed or self._server is not None:
            raise McpError(ErrorCode.PROTOCOL_FAILURE, "Transport is closed or already listening.")
        token = self._configured_token
        if token is None:
            token = secrets.token_hex(32)
        # Reject configured empty/whitespace tokens before creating a listener.
        if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
            raise McpError(
                ErrorCode.PROTOCOL_FAILURE,
                "Bearer token must be non-empty and use bearer-token characters.",
            )
        if self._configured_origins is not None and any(
            not _valid_origin(origin) for origin in self._configured_origins
        ):
            raise McpError(
                ErrorCode.PROTOCOL_FAILURE,
                "Allowed origins must be exact numeric loopback origins.",
            )
        self._expected = hashlib.sha256(token.encode("ascii")).digest()
        self._loop = asyncio.get_running_loop()
        self._handler = handler
        try:
            server = _HttpServer(("127.0.0.1", self._port), self)
        except OSError:
            self._handler = None
            raise McpError(
                ErrorCode.PROTOCOL_FAILURE, "Could not start loopback HTTP transport."
            ) from None
        self._server = server
        origin = f"http://127.0.0.1:{server.server_port}"
        self._origins = set(
            (origin,) if self._configured_origins is None else self._configured_origins
        )
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="prism-mcp-http",
            daemon=True,
        )
        self._thread.start()
        return {"url": origin + "/mcp", "token": token}

    async def _dispatch(self, request: TransportRequest) -> Any:
        handler = self._handler
        if self._closed or handler is None:
            raise McpError(ErrorCode.PROTOCOL_FAILURE, "Transport closed during request.")
        return await handler(request)

    async def close(self) -> None:
        self._closed = True
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        self._handler = None
        with self._lock:
            pending = tuple(self._pending)
            sockets = tuple(self._sockets)
        for future in pending:
            future.cancel()
        for connection in sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # An already disconnected peer owns no live connection.
            connection.close()
        server = self._server
        if server is not None:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join)
        # Do not await handler-owned work that suppresses cancellation. Listener
        # and request sockets are already released independently of that work.


class _HttpServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], owner: LoopbackHttpTransport) -> None:
        self.owner = owner
        super().__init__(address, _RequestHandler)

    def get_request(self) -> tuple[socket.socket, Any]:
        connection, address = super().get_request()
        connection.settimeout(None)
        with self.owner._lock:
            if self.owner._closed:
                connection.close()
                raise OSError("Transport closed")
            self.owner._sockets.add(connection)
        return connection, address

    def shutdown_request(self, request: Any) -> None:
        try:
            super().shutdown_request(request)
        finally:
            with self.owner._lock:
                self.owner._sockets.discard(request)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # stdlib's default traceback may include embedder diagnostics. Never log.
        return


class _RequestHandler(BaseHTTPRequestHandler):
    server: _HttpServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return  # No access log: requests and credentials stay private.

    def _authorize(self) -> int | None:
        owner = self.server.owner
        origins = self.headers.get_all("Origin")
        if origins is not None and (len(origins) != 1 or origins[0] not in owner._origins):
            return 403
        values = self.headers.get_all("Authorization")
        match = (
            re.fullmatch(r"Bearer ([A-Za-z0-9._~+/-]+=*)", values[0], re.IGNORECASE | re.ASCII)
            if values is not None and len(values) == 1
            else None
        )
        if match is None or not hmac.compare_digest(
            hashlib.sha256(match[1].encode("ascii")).digest(), owner._expected
        ):
            return 401
        return None

    def handle_expect_100(self) -> bool:
        status = self._authorize()
        if status is not None:
            self._send(status, _rpc_error(None, -32000))
            return False
        return super().handle_expect_100()

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        status = self._authorize()
        if status is not None:
            self._send(status, _rpc_error(None, -32000))
            return False
        return True

    def do_GET(self) -> None:
        self._send(405, _rpc_error(None, -32600))

    do_HEAD = do_GET
    do_OPTIONS = do_GET
    do_PUT = do_GET
    do_DELETE = do_GET
    do_PATCH = do_GET
    do_CONNECT = do_GET

    def do_POST(self) -> None:
        try:
            self._post()
        except (OSError, concurrent.futures.CancelledError):
            return  # Peer/transport closed. No response connection remains.
        except Exception:  # noqa: BLE001 -- transport diagnostics must not escape
            self._send(500, _rpc_error(None, -32603))

    def _post(self) -> None:
        owner = self.server.owner
        if self.path != "/mcp":
            self._send(404, _rpc_error(None, -32600))
            return
        if self.headers.get("Upgrade") is not None:
            self._send(405, _rpc_error(None, -32600))
            return
        accept = self.headers.get("Accept")
        if accept and not any(
            entry.split(";")[0].strip().lower() in ("application/json", "*/*")
            for entry in accept.split(",")
        ):
            self._send(406, _rpc_error(None, -32600))
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self._send(415, _rpc_error(None, -32600))
            return
        headers: dict[str, str] = {}
        for name, value in self.headers.items():
            key = name.lower()
            if key in headers:
                self._send(400, _rpc_error(None, -32600))
                return
            headers[key] = value
        if "transfer-encoding" in headers:
            self._send(400, _rpc_error(None, -32600))
            return
        length = headers.get("content-length", "")
        if re.fullmatch(r"0|[1-9][0-9]*", length, re.ASCII) is None:
            self._send(411, _rpc_error(None, -32600))
            return
        if len(length) > 10 or int(length) > owner._max_request_bytes:
            self._send(413, _rpc_error(None, -32600))
            return
        data = self.rfile.read(int(length))
        if len(data) != int(length):
            self._send(400, _rpc_error(None, -32600))
            return
        try:
            body = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError, RecursionError):
            self._send(400, _rpc_error(None, -32700))
            return
        if (
            not isinstance(body, dict)
            or body.get("jsonrpc") != "2.0"
            or not (
                isinstance(body.get("id"), str)
                or (type(body.get("id")) in (int, float) and math.isfinite(body["id"]))
            )
            or not isinstance(body.get("method"), str)
            or ("params" in body and not isinstance(body["params"], dict))
        ):
            self._send(400, _rpc_error(None, -32600))
            return
        loop = owner._loop
        assert loop is not None
        with owner._lock:
            if owner._closed:
                return
            future = asyncio.run_coroutine_threadsafe(
                owner._dispatch(TransportRequest(body["method"], body.get("params"), headers)), loop
            )
            owner._pending.add(future)
        try:
            try:
                result = future.result()  # Deliberately no request deadline.
                reply = {"jsonrpc": "2.0", "id": body["id"], "result": result}
            except McpError as error:
                reply = _rpc_error(body["id"], -32000, error.code)
            except concurrent.futures.CancelledError:
                return
            except Exception:  # noqa: BLE001 -- never return application diagnostics
                reply = _rpc_error(body["id"], -32000, ErrorCode.PROTOCOL_FAILURE.value)
            if not owner._closed:
                self._send(200, reply)
        finally:
            with owner._lock:
                owner._pending.discard(future)

    def _send(self, status: int, body: Any) -> None:
        data = json.dumps(body, ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)


def _reject_constant(value: str) -> Any:
    raise ValueError("Non-JSON numeric constant")


def _rpc_error(identifier: Any, code: int, category: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": "MCP request refused."}
    if category is not None:
        error["data"] = {"code": category}
    return {"jsonrpc": "2.0", "id": identifier, "error": error}
