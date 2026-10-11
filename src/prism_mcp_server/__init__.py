"""Stateless MCP tool serving; network framing belongs to ServerTransport."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, TypedDict

from prism_mcp import (
    KNOWN_PROTOCOL_VERSIONS,
    LATEST_PROTOCOL_VERSION,
    ErrorCode,
    McpError,
    MirroredParameters,
    ResultGuard,
    ToolDefinition,
    ToolGate,
    ToolResult,
    TransportRequest,
    TrustPolicy,
    allow_all,
)

from .http import LoopbackHttpTransport

__all__ = [
    "ErrorCode",
    "InMemoryTransport",
    "JsonObject",
    "LoopbackHttpTransport",
    "McpError",
    "MirroredParameters",
    "ResultGuard",
    "Server",
    "ServerTransport",
    "ToolContext",
    "ToolDefinition",
    "ToolGate",
    "ToolHandler",
    "ToolResult",
    "TransportRequest",
    "TrustPolicy",
    "allow_all",
]

# TrustPolicy is a CLIENT defence against a hostile server's definitions, not an
# inbound authorization policy. The embedder owns registration; ToolGate admits calls.
JsonObject: TypeAlias = dict[str, Any]


class ToolContext(TypedDict):
    server: str
    tool: str
    headers: dict[str, str]


ToolHandler: TypeAlias = Callable[[JsonObject, ToolContext], ToolResult | Awaitable[ToolResult]]
RequestHandler: TypeAlias = Callable[[TransportRequest], Awaitable[Any]]


class Endpoint(TypedDict):
    url: str
    token: str | None


class ServerTransport(Protocol):
    async def listen(self, handler: RequestHandler) -> Endpoint: ...

    async def close(self) -> None: ...


_VERSION = "io.modelcontextprotocol/protocolVersion"
_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
_SERVER_INFO = "io.modelcontextprotocol/serverInfo"
_PREFIX = "mcp-param-"


@dataclass(frozen=True)
class _RegisteredTool:
    definition: ToolDefinition
    handler: ToolHandler
    mirrored: MirroredParameters


class Server:
    def __init__(
        self,
        server: str,
        transport: ServerTransport,
        gate: ToolGate | Callable[[str, str, JsonObject], Awaitable[bool]] | None = None,
        guard: ResultGuard | None = None,
    ) -> None:
        self._server = server
        self._transport = transport
        self._gate = gate if gate is not None else allow_all
        # Clients frame provenance; default serving caps bytes without altering text.
        self._guard = guard if guard is not None else ResultGuard(frame_provenance=False)
        self._tools: dict[str, _RegisteredTool] = {}
        self._refused: set[str] = set()
        self._closed = False

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        self._assert_open()
        if definition.name in self._tools or definition.name in self._refused:
            raise McpError(
                ErrorCode.PROTOCOL_FAILURE, "A tool with this name is registered or refused."
            )
        snapshot = ToolDefinition.from_payload(_payload_of(definition))
        try:
            mirrored = MirroredParameters.from_schema(snapshot.name, snapshot.input_schema)
        except McpError as error:
            if error.code == ErrorCode.MIRRORED_PARAMETER_REFUSED.value:
                self._refused.add(snapshot.name)
            raise
        self._tools[snapshot.name] = _RegisteredTool(snapshot, handler, mirrored)

    async def listen(self) -> Endpoint:
        self._assert_open()
        return await self._transport.listen(self.handle)

    async def close(self) -> None:
        self._closed = True
        await self._transport.close()

    async def handle(self, request: TransportRequest) -> Any:
        self._assert_open()
        # Discovery precedes version selection, so no agreed-revision metadata
        # can be required here. HTTP bearer/Origin checks stay in the transport.
        if request.method == "server/discover":
            return _complete(
                {
                    "supportedVersions": [LATEST_PROTOCOL_VERSION],
                    "capabilities": {"tools": {}},
                    "ttlMs": 0,
                    "cacheScope": "private",
                }
            )
        params = request.params if request.params is not None else {}
        _assert_protocol(params)
        if request.method == "tools/list":
            return _complete(
                {
                    "tools": [_payload_of(tool.definition) for tool in self._tools.values()],
                    "ttlMs": 0,
                    "cacheScope": "private",
                }
            )
        if request.method != "tools/call":
            raise McpError(
                ErrorCode.PROTOCOL_FAILURE,
                "Only server/discover, tools/list and tools/call are supported; no initialize handshake.",
            )
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise McpError(ErrorCode.PROTOCOL_FAILURE, "tools/call requires a tool name.")
        if name in self._refused:
            raise McpError(
                ErrorCode.MIRRORED_PARAMETER_REFUSED,
                "The tool has invalid mirrored-parameter annotations.",
            )
        tool = self._tools.get(name)
        if tool is None:
            raise McpError(ErrorCode.TOOL_CALL_FAILED, "The requested tool is not registered.")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise McpError(ErrorCode.PROTOCOL_FAILURE, "Tool arguments must be an object.")
        args = _clone_object(arguments)
        headers = _mirrored_headers(request.headers)
        _resolve_mirrors(tool.mirrored, headers, args)
        verdict = self._gate(self._server, name, args)
        # Serving is asynchronous in both languages; gates may be sync or async.
        # The sibling Python client's synchronous API remains unchanged.
        if inspect.isawaitable(verdict):
            verdict = await verdict
        if not verdict:
            raise McpError(ErrorCode.TOOL_DENIED, "The gate refused this tool call.")
        self._assert_open()
        try:
            result = tool.handler(args, {"server": self._server, "tool": name, "headers": headers})
            if inspect.isawaitable(result):
                result = await result
        except Exception as error:  # noqa: BLE001 -- sanitize arbitrary application failures
            # Categories survive; arbitrary handler diagnostics may contain secrets.
            code = error.code if isinstance(error, McpError) else ErrorCode.TOOL_CALL_FAILED.value
            raise McpError(code, "The tool handler could not complete the request.") from None
        if (
            not isinstance(result, ToolResult)
            or not isinstance(result.text, str)
            or type(result.is_error) is not bool
        ):
            raise McpError(ErrorCode.PROTOCOL_FAILURE, "The handler must return ToolResult.")
        text = self._guard.guard(self._server, name, result.text)
        return _complete({"content": [{"type": "text", "text": text}], "isError": result.is_error})

    def _assert_open(self) -> None:
        if self._closed:
            raise McpError(ErrorCode.PROTOCOL_FAILURE, "The server is closed.")


def _assert_protocol(params: JsonObject) -> None:
    if not isinstance(params, dict) or not isinstance(params.get("_meta"), dict):
        raise McpError(ErrorCode.PROTOCOL_FAILURE, "Stateless requests require per-request _meta.")
    meta = params["_meta"]
    version = meta.get(_VERSION)
    if version != LATEST_PROTOCOL_VERSION:
        kind = "revision" if version in KNOWN_PROTOCOL_VERSIONS else "version"
        value = version if isinstance(version, str) else "missing"
        raise McpError(
            ErrorCode.UNSUPPORTED_PROTOCOL_VERSION,
            f"Protocol {kind} [{value}] is unsupported; expected [{LATEST_PROTOCOL_VERSION}].",
        )
    if not isinstance(meta.get(_CAPABILITIES), dict):
        raise McpError(
            ErrorCode.PROTOCOL_FAILURE,
            "Requests require client capabilities in _meta.",
        )
    if _CLIENT_INFO in meta and not isinstance(meta[_CLIENT_INFO], dict):
        raise McpError(ErrorCode.PROTOCOL_FAILURE, "Client info must be an object when supplied.")


def _complete(result: JsonObject) -> JsonObject:
    # Identify the serving software independently of the gate/context label.
    # Fresh objects keep caller mutation from affecting subsequent responses.
    return {
        **result,
        "resultType": "complete",
        "_meta": {_SERVER_INFO: {"name": "prism-ai-mcp-server", "version": "0.1.0"}},
    }


def _clone_object(value: JsonObject) -> JsonObject:
    try:
        result: JsonObject = json.loads(json.dumps(value, allow_nan=False))
        return result
    except (TypeError, ValueError, RecursionError):
        raise McpError(ErrorCode.PROTOCOL_FAILURE, "Expected JSON-serializable data.") from None


def _payload_of(tool: ToolDefinition) -> JsonObject:
    return _clone_object(
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
            "title": tool.title,
            "annotations": tool.annotations,
        }
    )


def _mirrored_headers(raw: dict[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in raw.items():
        if not name.lower().startswith(_PREFIX):
            continue
        key = name[len(_PREFIX) :].lower()
        if not isinstance(value, str) or key in headers:
            raise McpError(
                ErrorCode.PROTOCOL_FAILURE, "Mirrored headers require unambiguous string values."
            )
        headers[key] = value
    # Authorization, Cookie, Origin and all other unnamespaced headers stay private.
    return headers


def _resolve_mirrors(
    mirrored: MirroredParameters, headers: dict[str, str], args: JsonObject
) -> None:
    for name, parameter in mirrored.parameters.items():
        raw = headers.get(name.lower())
        if raw is None:
            continue
        value: str | int | bool = raw
        if parameter["type"] == "boolean":
            if raw not in ("true", "false"):
                raise McpError(ErrorCode.PROTOCOL_FAILURE, "Invalid mirrored boolean.")
            value = raw == "true"
        elif parameter["type"] == "integer":
            if re.fullmatch(r"-?(0|[1-9][0-9]*)", raw) is None:
                raise McpError(ErrorCode.PROTOCOL_FAILURE, "Invalid mirrored integer.")
            value = int(raw)
            if abs(value) > 2**53 - 1:
                raise McpError(ErrorCode.PROTOCOL_FAILURE, "Invalid mirrored integer.")
        target = args
        path = parameter["path"]
        for segment in path[:-1]:
            if segment not in target:
                target[segment] = {}
            nested = target[segment]
            if not isinstance(nested, dict):
                raise McpError(
                    ErrorCode.PROTOCOL_FAILURE, "Mirrored path conflicts with tool arguments."
                )
            target = nested
        leaf = path[-1]
        if leaf in target and (
            target[leaf] != value
            or (
                (type(target[leaf]) is bool or type(value) is bool)
                and type(target[leaf]) is not type(value)
            )
        ):
            raise McpError(
                ErrorCode.PROTOCOL_FAILURE, "Mirrored header conflicts with tool arguments."
            )
        target[leaf] = value


class InMemoryTransport:
    """In-process transport. No sockets, HTTP authentication or request timeout."""

    def __init__(self) -> None:
        self._handler: RequestHandler | None = None
        self._closed = False
        self._pending: dict[asyncio.Future[Any], asyncio.Future[Any]] = {}

    async def listen(self, handler: RequestHandler) -> Endpoint:
        if self._closed or self._handler is not None:
            raise McpError(ErrorCode.PROTOCOL_FAILURE, "Transport is closed or already listening.")
        self._handler = handler
        return {"url": "memory://prism-mcp-server", "token": None}

    async def request(self, request: TransportRequest) -> Any:
        handler = self._handler
        if self._closed or handler is None:
            raise McpError(ErrorCode.PROTOCOL_FAILURE, "Transport is not listening.")
        task = asyncio.ensure_future(handler(request))
        waiter: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[waiter] = task

        def completed(future: asyncio.Future[Any]) -> None:
            # Always consume completion, even if close already released the caller.
            try:
                result = future.result()
            except asyncio.CancelledError:
                if not waiter.done():
                    waiter.cancel()
            except Exception as error:  # noqa: BLE001 -- forward every handler task failure
                if not waiter.done():
                    waiter.set_exception(error)
            else:
                if not waiter.done():
                    waiter.set_result(result)

        task.add_done_callback(completed)
        try:
            return await waiter
        finally:
            self._pending.pop(waiter, None)
            if not task.done():
                task.cancel()

    async def close(self) -> None:
        self._closed = True
        self._handler = None
        for waiter, task in self._pending.items():
            if not waiter.done():
                waiter.set_exception(
                    McpError(ErrorCode.PROTOCOL_FAILURE, "Transport closed during request.")
                )
            task.cancel()
        self._pending.clear()
        # Do not wait on application code that may suppress cancellation. Pending
        # request waiters must still be released; no network resource is held here.
