"""Stateless MCP tool serving; network framing belongs to ServerTransport."""

from __future__ import annotations

import asyncio
import inspect
import json
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

from ._headers import (
    HEADER_MISMATCH,
    HeaderMismatchError,
    check_mirrored_headers,
    check_standard_headers,
    header_view,
)
from .http import LoopbackHttpTransport

__all__ = [
    "HEADER_MISMATCH",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "ErrorCode",
    "HeaderMismatchError",
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
    """A transport may set ``requires_mirrored_headers = False`` to opt out of requiring headers.

    When it is absent or true, requests arrive with HTTP headers, so a 2026-07-28 tools/call must
    carry an ``Mcp-Param-*`` header for every mirrored value in its body. A transport opts OUT, so a
    custom HTTP transport is held to the rule by default. Headers that ARE present are checked
    against the body on every transport and revision.
    """

    async def listen(self, handler: RequestHandler) -> Endpoint: ...

    async def close(self) -> None: ...


_VERSION = "io.modelcontextprotocol/protocolVersion"
_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
_SERVER_INFO = "io.modelcontextprotocol/serverInfo"
_STATELESS_KEYS = (_VERSION, _CAPABILITIES, _CLIENT_INFO)
_SERVER_NAME = "prism-ai-mcp-server"
_SERVER_VERSION = "0.2.1"

# The handshake revisions, served for clients that open with `initialize`
# (Codex's MCP client always does). Newest first; an unknown requested revision
# is answered with the first, as the spec says.
_CLASSIC_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

#: Every MCP protocol version this server serves, newest first: the stateless
#: revision, then the handshake revisions. The standing rule is that the five
#: newest official versions are always here; protocol-versions.json publishes
#: this list for the drift watcher, and a test keeps the two identical.
SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (LATEST_PROTOCOL_VERSION, *_CLASSIC_VERSIONS)

# Instructions are guidance for a model, not a document: bounded so one option
# cannot become a token sink.
_MAX_INSTRUCTIONS_BYTES = 16_384


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
        instructions: str | None = None,
    ) -> None:
        # Guidance a client may give its model (G-277), sent in the initialize and
        # server/discover results. It reaches the model, so it comes from the
        # embedder's configuration and is fixed here; no request can set it.
        self._instructions = _instructions_of(instructions)
        self._server = server
        self._transport = transport
        self._gate = gate if gate is not None else allow_all
        # Keep the serving limit independent of client defaults; clients frame provenance.
        self._guard = (
            guard if guard is not None else ResultGuard(max_bytes=65536, frame_provenance=False)
        )
        self._tools: dict[str, _RegisteredTool] = {}
        self._refused: set[str] = set()
        self._require_headers = getattr(transport, "requires_mirrored_headers", True) is not False
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
        params = request.params if request.params is not None else {}
        # A request carrying ANY stateless metadata key is a stateless request and is held to the
        # stateless rules; only one with none of them is classic. A half-formed stateless request
        # must not fall through to the classic path.
        meta = params.get("_meta") if isinstance(params, dict) else None
        stateless = isinstance(meta, dict) and any(key in meta for key in _STATELESS_KEYS)
        # Headers that mirror the body are checked against it before anything else reads either.
        headers = header_view(request.headers)
        check_standard_headers(headers, request.method, params, stateless)
        # Discovery precedes version selection, so no agreed-revision metadata
        # can be required here. HTTP bearer/Origin checks stay in the transport.
        if request.method == "server/discover":
            discovered: JsonObject = {
                "supportedVersions": [LATEST_PROTOCOL_VERSION],
                "capabilities": {"tools": {}},
            }
            if self._instructions is not None:
                discovered["instructions"] = self._instructions
            return _complete({**discovered, "ttlMs": 0, "cacheScope": "private"})
        if not stateless:
            return await self._classic(request, params, headers)
        _assert_protocol(params)
        if request.method == "tools/list":
            return _complete({"tools": self._listed(), "ttlMs": 0, "cacheScope": "private"})
        if request.method != "tools/call":
            raise McpError(
                ErrorCode.PROTOCOL_FAILURE,
                "Only server/discover, tools/list and tools/call are supported on the stateless revision.",
            )
        return _complete(await self._call(params, headers, True))

    async def _classic(
        self, request: TransportRequest, params: JsonObject, headers: dict[str, Any]
    ) -> Any:
        """The handshake protocol, for clients that open with ``initialize`` (Codex's always does).

        Stateless per request, which the streamable-HTTP transport permits: no session id is issued, so
        nothing is bound to one. Authentication is the transport's; the registry, gate, mirrors and result
        guard are the stateless path's own.
        """
        if request.method.startswith("notifications/"):
            return None
        if request.method == "initialize":
            return _initialize(params, self._instructions)
        header = request.headers.get("mcp-protocol-version")
        if header is not None and header not in _CLASSIC_VERSIONS:
            raise McpError(
                ErrorCode.UNSUPPORTED_PROTOCOL_VERSION,
                f"Protocol version [{header}] is not served on the handshake path; "
                f"expected one of [{', '.join(_CLASSIC_VERSIONS)}].",
            )
        if request.method == "ping":
            return {}
        if request.method == "tools/list":
            return {"tools": self._listed()}
        if request.method == "tools/call":
            return await self._call(params, headers, False)
        raise McpError(
            ErrorCode.PROTOCOL_FAILURE,
            "Only initialize, ping, tools/list and tools/call are supported.",
        )

    def _listed(self) -> list[JsonObject]:
        return [_without_nulls(_payload_of(tool.definition)) for tool in self._tools.values()]

    async def _call(
        self, params: JsonObject, headers: dict[str, Any], stateless: bool
    ) -> JsonObject:
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
        # Before the gate and the tool: neither may act on a call whose headers say something its
        # body does not.
        mirrored = check_mirrored_headers(
            tool.mirrored, headers, args, stateless and self._require_headers
        )
        verdict = self._gate(self._server, name, args)
        # Serving is asynchronous in both languages; gates may be sync or async,
        # as they may in the sibling client since prism-ai-mcp 0.2.1.
        if inspect.isawaitable(verdict):
            verdict = await verdict
        # A gate that delegates and forgets an inner await resolves to another
        # awaitable, which is truthy. Refused rather than read as consent (G-83).
        if inspect.isawaitable(verdict):
            if inspect.iscoroutine(verdict):
                verdict.close()
            raise TypeError(
                f"The gate must resolve to a verdict, got {type(verdict).__name__}. "
                "Await any coroutine inside the gate before returning."
            )
        if not verdict:
            raise McpError(ErrorCode.TOOL_DENIED, "The gate refused this tool call.")
        self._assert_open()
        try:
            result = tool.handler(args, {"server": self._server, "tool": name, "headers": mirrored})
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
        return {"content": [{"type": "text", "text": text}], "isError": result.is_error}

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


def _instructions_of(value: object) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > _MAX_INSTRUCTIONS_BYTES
    ):
        raise McpError(
            ErrorCode.PROTOCOL_FAILURE,
            f"Server instructions must be a non-empty string of at most {_MAX_INSTRUCTIONS_BYTES} bytes.",
        )
    return value


def _initialize(params: JsonObject, instructions: str | None) -> JsonObject:
    if (
        not isinstance(params, dict)
        or not isinstance(params.get("protocolVersion"), str)
        or ("capabilities" in params and not isinstance(params["capabilities"], dict))
        or ("clientInfo" in params and not isinstance(params["clientInfo"], dict))
    ):
        raise McpError(
            ErrorCode.PROTOCOL_FAILURE,
            "initialize requires a protocolVersion, and object capabilities and clientInfo.",
        )
    requested = params["protocolVersion"]
    version = requested if requested in _CLASSIC_VERSIONS else _CLASSIC_VERSIONS[0]
    result: JsonObject = {
        "protocolVersion": version,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
    }
    if instructions is not None:
        result["instructions"] = instructions
    return result


def _complete(result: JsonObject) -> JsonObject:
    # Identify the serving software independently of the gate/context label.
    # Fresh objects keep caller mutation from affecting subsequent responses.
    return {
        **result,
        "resultType": "complete",
        "_meta": {_SERVER_INFO: {"name": _SERVER_NAME, "version": _SERVER_VERSION}},
    }


def _clone_object(value: JsonObject) -> JsonObject:
    try:
        result: JsonObject = json.loads(json.dumps(value, allow_nan=False))
        return result
    except (TypeError, ValueError, RecursionError):
        raise McpError(ErrorCode.PROTOCOL_FAILURE, "Expected JSON-serializable data.") from None


def _without_nulls(tool: JsonObject) -> JsonObject:
    """A tools/list entry with its null fields left out.

    The spec types ``title`` (like every optional field) as an optional string: absent is valid, null is not.
    Claude Code validates strictly and dropped the WHOLE server on one ``title: null`` (tynn, Genie 2,
    2026-10-22). Pins are unaffected: ToolDefinition.from_payload() reads an absent title as None, so the
    digest a client computes from the entry is the registered definition's.
    """
    return {key: value for key, value in tool.items() if value is not None}


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


class InMemoryTransport:
    """In-process transport. No sockets, HTTP authentication or request timeout."""

    #: In process there are no HTTP headers to require; any that a request does carry are still checked.
    requires_mirrored_headers = False

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
