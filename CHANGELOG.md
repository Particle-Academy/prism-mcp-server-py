# Changelog

## 0.2.1

- Package metadata names Particle Academy as the author and links the
  repository, so PyPI shows who publishes `prism-ai-mcp-server` and where its
  source is. No code change.

## 0.2.0

- The classic handshake is served beside the stateless revision: `initialize`
  (2025-11-25, 2025-06-18, 2025-03-26, 2024-11-05), `notifications/initialized`
  (HTTP 202, no body), `ping`, `tools/list` and `tools/call`, as in
  prism-mcp-server-ts. Together with 2026-07-28 that is all five official MCP
  schema versions. A request with no stateless `_meta` key is now a classic
  request rather than a refused one.
- `Server(..., instructions=...)`: the server's guidance for a client's model,
  sent in the `initialize` and `server/discover` results (G-277). Embedder
  configuration only; a non-empty string of at most 16 KiB.
- `SUPPORTED_PROTOCOL_VERSIONS` and `protocol-versions.json` publish the
  versions served.

- **Breaking:** re-exported `TrustPolicy.allowing([])` now raises
  `server_not_trusted`; a `ResultGuard` you construct without `max_bytes` now
  defaults to 262,144 bytes. The server's own default guard keeps 65,536.
- When constructing a custom guard, pass `max_bytes=65536` to retain
  its previous limit. An explicitly supplied server guard remains caller-controlled.
- Update the exact client dependency from 0.2.1 to 0.3.0. The server uses
  `ResultGuard` directly; it does not call `Client.call_tool`. Handler result
  flags must still be booleans. Trust policy is not applied to inbound calls.
