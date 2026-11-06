# Changelog

## Unreleased

- **Breaking:** re-exported `TrustPolicy.allowing([])` now raises
  `server_not_trusted`; a `ResultGuard` you construct without `max_bytes` now
  defaults to 262,144 bytes. The server's own default guard keeps 65,536.
- When constructing a custom guard, pass `max_bytes=65536` to retain
  its previous limit. An explicitly supplied server guard remains caller-controlled.
- Update the exact client dependency from 0.2.1 to 0.3.0. The server uses
  `ResultGuard` directly; it does not call `Client.call_tool`. Handler result
  flags must still be booleans. Trust policy is not applied to inbound calls.
