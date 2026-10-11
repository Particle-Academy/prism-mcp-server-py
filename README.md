# prism-ai-mcp-server

Stateless MCP tool registration and dispatch over a consumer-supplied
`ServerTransport`, with in-memory and authenticated loopback HTTP implementations.
Python 3.10+, typed module `prism_mcp_server`. No stdio, handshake, sessions, push,
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

Server gates may return a boolean or an awaitable boolean. The asynchronous
server awaits the verdict before permitting execution; `async def` returning
`False` refuses the call. The sibling Python client's synchronous API is unchanged.

Only `Mcp-Param-*` headers reach handlers. The prefix is required and stripped
case-insensitively; context keys are lowercase. Raw `Authorization`, `Cookie`,
`Origin` and other unnamespaced headers stay private. Shared
`MirroredParameters` resolve nested string/integer/boolean fields into arguments
before the gate. Conflicting body/header values, ambiguous headers and invalid
primitive encodings are refused. Integers use the shared JS-safe range for
cross-language agreement. Other application argument validation belongs to handlers.

`tools/list` and `tools/call` require `params['_meta']` with
`io.modelcontextprotocol/protocolVersion: '2026-07-28'`,
`io.modelcontextprotocol/clientCapabilities: {}`.
`io.modelcontextprotocol/clientInfo: {name: 'example', version: '1'}` is optional.
Older revisions raise shared `unsupported_protocol_version`, naming both
revisions. `server/discover`, `tools/list` and `tools/call` are supported.
Discovery advertises `supportedVersions: ['2026-07-28']` and `capabilities: {tools: {}}`
regardless of the probe's requested version, including absent, older or unknown
versions. The probe needs no `params`, `_meta`, capabilities or client identity,
creates no session, and does not change server state. It uses the same
HTTP bearer and Origin checks as tool requests. Every successful result includes
`resultType: 'complete'` and `_meta['io.modelcontextprotocol/serverInfo']` identifying
the serving software as `{name: 'prism-ai-mcp-server', version: '0.1.0'}`.
Discovery and tool lists include `ttlMs: 0, cacheScope: 'private'`; callers re-fetch
to observe registration changes. Tool call results also carry
`{content: [{type: 'text', text: ...}], isError: ...}`; failures raise shared
`McpError` for transport mapping. Arbitrary handler diagnostic messages are
excluded; a shared error category is retained.

In-memory `listen()` returns a `memory://` identifier and `token: None`, not HTTP
credentials or a dialable URL. Requests have no fixed timeout. `close()` releases
pending request waiters, rejects further requests and is idempotent. Cancellation
of handler-owned work is best-effort; an embedder must clean up application work
that suppresses cancellation. SDK conformance and consumer acceptance remain phase 3.

For HTTP, replace the transport with `LoopbackHttpTransport()`:

```python
from prism_mcp_server import LoopbackHttpTransport, Server

transport = LoopbackHttpTransport()
server = Server("example", transport)
# Register application tools. Inside your event loop:
# endpoint = await server.listen()
# Give endpoint["url"] and endpoint["token"] privately to the client.
# Each POST needs Authorization: Bearer <token> and Content-Type: application/json.
# Send JSON-RPC 2.0 with id, method and the per-request params["_meta"] above.
# await server.close() when the embedder shuts down.
```

The HTTP URL is `http://127.0.0.1:<port>/mcp`. `port` defaults to `0` (an OS-chosen
port); the bind address is fixed to literal `127.0.0.1`. `token` is generated at
startup with `secrets.token_hex(32)`, or may be supplied as a keyword argument.
Configured tokens must be non-empty bearer-token strings; whitespace is refused
before listening. HTTP always returns a real token; only memory returns `None`.
The URL alone does not authorize a client. Keep the token out of URLs and logs.
Every request requires `Authorization: Bearer <token>`; comparison uses fixed-size
SHA-256 digests and `hmac.compare_digest`.

Requests without `Origin` may proceed with valid credentials. A browser `Origin`
must exactly match the listener's own origin by default. `allowed_origins` replaces
that allow-list with explicit `http://127.0.0.1:<port>` / `https://[::1]:<port>`
origins (either scheme is supported); arbitrary hostnames, paths, wildcard origins
and non-loopback addresses are refused at startup. No browser CORS/preflight
support is provided. GET, upgrades, CONNECT and SSE-only requests are refused;
an Accept header offering both JSON and SSE receives JSON.

POSTs require `Content-Length` and JSON content; chunked uploads, batches,
notifications and ambiguous duplicate headers are refused. `max_request_bytes`
defaults to 1 MiB. Responses carry the same request id and either `result` or a
JSON-RPC `error` with the shared Prism category in `error.data.code`; arbitrary
diagnostic messages are withheld. HTTP refusal uses 401 for credentials and 403
for Origin. There is no fixed request or socket timeout. The stdlib HTTP listener
uses daemon threads for HTTP I/O; registered handlers run on the calling asyncio
loop. `close()` releases every socket, including incomplete headers and held
long-polls, before resolving. Application cancellation remains best-effort.

Local setup: create a virtual environment and install `-e .[dev]` against the
published sibling dependency. Gates: `python -m pytest`, `python -m mypy --strict src tests`,
`python -m ruff check .`, `python -m ruff format --check .`.
No remote, publishing workflow or tag is configured here.
The complete suite includes a real 310-second HTTP long-poll and takes about
six minutes. SDK reference-client conformance is not part of this phase.
