"""The HTTP headers that mirror request-body fields, checked against the body they mirror.

MCP 2026-07-28, Streamable HTTP: "Request Metadata" and "Server Validation". The body is the source
of truth. A header exists so something in front of the server can route on it without parsing the
body, so a header that disagrees with the body is refused before any tool runs: otherwise the
intermediary and the server act on different values.

Shared with prism-mcp-server-ts (``headers.ts``); the corpus in
tests/fixtures/mcp-header-mirror.json holds both to the same outcomes and messages.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from typing import Any

from prism_mcp import ErrorCode, McpError, MirroredParameters

#: The JSON-RPC error code the specification assigns to a header/body disagreement (``HeaderMismatch``).
HEADER_MISMATCH = -32020

REPEATED = "a header that mirrors the request body is repeated."
_PARAM_PREFIX = "mcp-param-"
_SENTINEL_PREFIX = "=?base64?"
_SENTINEL_SUFFIX = "?="
# RFC 4648 section 4 with its padding: the only spelling the specification's encoding produces.
_BASE64 = re.compile(r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?", re.ASCII)
# A field value is visible ASCII, space and horizontal tab (RFC 9110 section 5.5).
_FIELD_VALUE = re.compile(r"[\x20-\x7E\t]*", re.ASCII)
# An integer, optionally with a zero fraction ("42.0" is 42). No sign but minus, no exponent, no leading zero.
_INTEGER = re.compile(r"(-?(?:0|[1-9][0-9]*))(?:\.0+)?", re.ASCII)
_MAX_SAFE = 2**53 - 1


class HeaderMismatchError(McpError):
    """A header that disagrees with the request body. Transports answer it with HTTP 400 and -32020."""

    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.PROTOCOL_FAILURE, f"Header mismatch: {message}")


def is_mirror_header(name: str) -> bool:
    """Whether a header name is one this module checks: every ``Mcp-*`` header mirrors part of the body."""
    return name.lower().startswith("mcp-")


def header_view(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Lowercased names to values. A repeated mirror header (in any casing) is refused here."""
    view: dict[str, Any] = {}
    for name, value in (raw or {}).items():
        key = name.lower()
        if key in view:
            if is_mirror_header(key):
                raise HeaderMismatchError(REPEATED)
            continue
        view[key] = value
    return view


def _field_value(value: Any) -> str | None:
    """A field value without its optional surrounding whitespace, or None when it is not a legal one.

    Whitespace around a field value is never part of it, so a value that needs it travels in the sentinel.
    """
    if not isinstance(value, str):
        return None
    trimmed = value.strip(" \t")
    return trimmed if trimmed and _FIELD_VALUE.fullmatch(trimmed) else None


def _decoded(value: Any) -> str | None:
    """The value a header carries: the sentinel decoded, anything else as written. None when malformed."""
    field = _field_value(value)
    if field is None:
        return None
    if (
        len(field) < len(_SENTINEL_PREFIX) + len(_SENTINEL_SUFFIX)
        or not field.startswith(_SENTINEL_PREFIX)
        or not field.endswith(_SENTINEL_SUFFIX)
    ):
        return field
    payload = field[len(_SENTINEL_PREFIX) : -len(_SENTINEL_SUFFIX)]
    if _BASE64.fullmatch(payload) is None:
        return None
    try:
        return base64.b64decode(payload, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None


def _check_standard(name: str, value: Any, expected: Any, encoded: bool) -> None:
    if value is None:
        return
    actual = _decoded(value) if encoded else _field_value(value)
    if actual is None:
        raise HeaderMismatchError(f"the [{name}] header value is malformed.")
    if not isinstance(expected, str) or actual != expected:
        raise HeaderMismatchError(f"the [{name}] header does not match the request body.")


def check_standard_headers(
    headers: Mapping[str, Any], method: str, params: Mapping[str, Any], stateless: bool
) -> None:
    """``Mcp-Method``, ``Mcp-Name`` and, on the stateless revision, ``MCP-Protocol-Version``.

    Each must equal the body field it mirrors. A header present on a request it does not apply to has
    nothing to agree with, so it is refused too.

    Presence is not yet required. The specification requires these three on every 2026-07-28
    request, and the published prism-mcp clients do not send them yet; requiring them now would
    refuse every one of those clients.
    """
    _check_standard("Mcp-Method", headers.get("mcp-method"), method, False)
    name = params.get("name") if method == "tools/call" else None
    _check_standard("Mcp-Name", headers.get("mcp-name"), name, True)
    if stateless:
        meta = params.get("_meta")
        version = (
            meta.get("io.modelcontextprotocol/protocolVersion") if isinstance(meta, dict) else None
        )
        _check_standard("MCP-Protocol-Version", headers.get("mcp-protocol-version"), version, False)
    if method != "tools/call" and any(key.startswith(_PARAM_PREFIX) for key in headers):
        raise HeaderMismatchError("Mcp-Param headers apply only to tools/call.")


def _body_value(args: Mapping[str, Any], path: list[str]) -> Any:
    """The body value at a mirrored path; None (null, absent, or a path through a non-object) means none."""
    value: Any = args
    for segment in path:
        value = value.get(segment) if isinstance(value, dict) else None
    return value


def _valid_body(kind: str, body: Any) -> bool:
    if kind == "integer":
        if isinstance(body, float) and body.is_integer():
            body = int(body)
        return type(body) is int and -_MAX_SAFE <= body <= _MAX_SAFE
    if kind == "boolean":
        return type(body) is bool
    return isinstance(body, str)


def _matches(kind: str, header: str, body: Any) -> bool:
    if kind == "integer":
        match = _INTEGER.fullmatch(header)
        # A safe integer is at most 17 characters with its sign; a longer header cannot equal one and is never converted.
        return match is not None and len(match[1]) <= 17 and int(match[1]) == int(body)
    if kind == "boolean":
        return header == ("true" if body else "false")
    return bool(header == body)


def check_mirrored_headers(
    mirrored: MirroredParameters,
    headers: Mapping[str, Any],
    args: Mapping[str, Any],
    required: bool,
) -> dict[str, str]:
    """Every ``Mcp-Param-*`` header against the tool's declarations and the call's own arguments.

    - A header the tool does not declare is refused. So is a header for a value the body does not
      carry: a header never becomes an argument, the body is the only source of the call's arguments.
    - A header that is present is decoded (the ``=?base64?...?=`` sentinel) and must equal the body
      value. Integers compare by value, so ``3.0`` is ``3``, exactly and without floating point.
    - When ``required``, a body value needs its header. That is the 2026-07-28 rule for a
      header-carrying transport; the earlier revisions define no mirrored headers, so a client
      speaking one of them is not asked for any.

    Returns the decoded values, by lowercased name, for the tool's context. Arguments are never modified.
    """
    declared = {name.lower() for name in mirrored.parameters}
    for key in headers:
        if key.startswith(_PARAM_PREFIX) and key[len(_PARAM_PREFIX) :] not in declared:
            raise HeaderMismatchError("an Mcp-Param header is not declared by this tool.")
    values: dict[str, str] = {}
    for name, parameter in mirrored.parameters.items():
        label = f"Mcp-Param-{name}"
        key = name.lower()
        present = _PARAM_PREFIX + key in headers
        body = _body_value(args, parameter["path"])
        if body is None:
            if present:
                raise HeaderMismatchError(
                    f"the [{label}] header is present but the request body has no value for it."
                )
            continue
        if not present:
            if required:
                raise HeaderMismatchError(f"the [{label}] header is required.")
            continue
        header = _decoded(headers[_PARAM_PREFIX + key])
        if header is None:
            raise HeaderMismatchError(f"the [{label}] header value is malformed.")
        kind = parameter["type"]
        if not _valid_body(kind, body):
            raise HeaderMismatchError(
                f"the request body value for [{label}] is not a valid {kind}."
            )
        if not _matches(kind, header, body):
            raise HeaderMismatchError(f"the [{label}] header does not match the request body.")
        values[key] = header
    return values
