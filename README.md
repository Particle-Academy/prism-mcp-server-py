# prism-ai-mcp-server

Phase 1: stateless MCP tool registration and dispatch over a consumer-supplied
`ServerTransport`, with an in-memory implementation. Python 3.10+, typed module
`prism_mcp_server`. No HTTP listener, sockets, stdio, handshake, sessions, push,
resources or prompts.

The sole runtime dependency is `prism-ai-mcp`; there are **zero third-party
runtime dependencies**. Shared definitions/errors/guards come from `prism_mcp`.
Python keeps the existing names `ErrorCode` and `ToolResult`. The local
`JsonObject = dict[str, Any]` alias adds no runtime behavior. The official `mcp`
SDK is in the dev extra only, reserved for phase 3 reference-client tests;
production code does not import it.

```python
from prism_mcp import ToolDefinition, ToolResult
from prism_mcp_server import InMemoryTransport, JsonObject, Server, ToolContext

transport = InMemoryTransport()
server = Server("example", transport)


def echo(args: JsonObject, context: ToolContext) -> ToolResult:
    return ToolResult(str(args.get("text", "")))


server.register(
    ToolDefinition(
        "echo",
        "Echo text",
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
    ),
    echo,
)
# Inside your event loop: await server.listen(), await transport.request(...),
# then await server.close(). Synchronous and awaitable handlers are accepted.
```

Handlers take arguments plus `{server, tool, headers}` context and return the
shared `ToolResult`. Registration snapshots definitions; duplicate names fail.
Invalid mirrored annotations raise `mirrored_parameter_refused`, disappear from
`tools/list`, and cannot run. Listed definitions preserve the client's digest.

`ToolGate` defaults to **allow_all on the server**, unlike the client's
deny-by-default trust posture. Registration explicitly chooses what is served;
an embedder's custom gate may refuse calls before handlers run. `TrustPolicy`
does not apply inbound. Default outbound `ResultGuard` caps UTF-8 text at 64 KiB
for both success and `is_error` results, without provenance framing; clients
apply that framing once. Custom guards can intentionally transform text.

Only `Mcp-Param-*` headers reach handlers. The prefix is required and stripped
case-insensitively; context keys are lowercase. Raw `Authorization`, `Cookie`,
`Origin` and other unnamespaced headers stay private. Shared
`MirroredParameters` resolve nested string/integer/boolean fields into arguments
before the gate. Conflicting body/header values, ambiguous headers and invalid
primitive encodings are refused. Integers use the shared JS-safe range for
cross-language agreement. Other application argument validation belongs to handlers.

Every request requires `params['_meta']` with
`io.modelcontextprotocol/protocolVersion: '2026-07-28'`,
`io.modelcontextprotocol/clientCapabilities: {}` and
`io.modelcontextprotocol/clientInfo: {name: 'example', version: '1'}`.
Older revisions raise shared `unsupported_protocol_version`, naming both
revisions. Only `tools/list` and `tools/call` are supported. Wire results are
`{content: [{type: 'text', text: ...}], isError: ...}`; failures raise shared
`McpError` for transport mapping. Arbitrary handler diagnostic messages are
excluded; a shared error category is retained.

In-memory `listen()` returns a `memory://` identifier and `token: None`, not HTTP
credentials or a dialable URL. Requests have no fixed timeout. `close()` releases
pending request waiters, rejects further requests and is idempotent. Cancellation
of handler-owned work is best-effort; an embedder must clean up application work
that suppresses cancellation. Phase 2 adds secure loopback HTTP transport;
phase 3 adds SDK conformance and consumer acceptance tests.

Local setup: create a virtual environment, install `-e ../prism-mcp-py` and
`-e .[dev]`. Gates: `python -m pytest`, `python -m mypy --strict src tests`,
`python -m ruff check .`, `python -m ruff format --check .`.
No remote, publishing workflow or tag is configured here.
