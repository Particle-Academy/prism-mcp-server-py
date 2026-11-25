"""The shared header-mirroring corpus, through the real loopback server.

Each row is written byte for byte on a raw socket so every header reaches the server exactly as the
row spells it: casing, repeats, padding and non-ASCII bytes included. The TypeScript server runs the
same file; both pin its SHA-256 (line endings normalised) so a row cannot change in one language only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from prism_mcp import ToolDefinition, ToolResult, TransportRequest

from prism_mcp_server import (
    HeaderMismatchError,
    InMemoryTransport,
    JsonObject,
    LoopbackHttpTransport,
    Server,
    ToolContext,
)

CORPUS_FILE = Path(__file__).parent / "fixtures" / "mcp-header-mirror.json"
CORPUS_SHA256 = "cdacbf274bf31c03ef7078c66105bbb9c069c19ac2fd97cc76f832ebb3abb961"
RAW = CORPUS_FILE.read_bytes()
CORPUS: dict[str, Any] = json.loads(RAW.decode("utf-8"))
STATELESS = "2026-07-28"


def _body(row: dict[str, Any]) -> JsonObject:
    params: JsonObject = {"name": row["tool"]}
    if "arguments" in row:
        params["arguments"] = row["arguments"]
    if row["revision"] == STATELESS:
        params["_meta"] = {
            "io.modelcontextprotocol/protocolVersion": STATELESS,
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {"name": "header-mirror-corpus", "version": "1"},
        }
    return {"jsonrpc": "2.0", "id": row["id"], "method": "tools/call", "params": params}


def _standard(row: dict[str, Any]) -> list[list[str]]:
    if not row["standard"]:
        return []
    if row["revision"] == STATELESS:
        return [
            ["MCP-Protocol-Version", STATELESS],
            ["Mcp-Method", "tools/call"],
            ["Mcp-Name", row["tool"]],
        ]
    return [["MCP-Protocol-Version", row["revision"]]] if row["revision"] >= "2025-06-18" else []


def _post(url: str, token: str, headers: list[list[str]], payload: JsonObject) -> tuple[int, Any]:
    """One HTTP/1.1 POST written byte for byte; http.client would refuse the hostile rows."""
    parsed = urlsplit(url)
    data = json.dumps(payload).encode("utf-8")
    lines = [
        f"POST {parsed.path} HTTP/1.1",
        f"Host: {parsed.netloc}",
        f"Authorization: Bearer {token}",
        "Content-Type: application/json",
        "Accept: application/json",
        f"Content-Length: {len(data)}",
        "Connection: close",
        *(f"{name}: {value}" for name, value in headers),
    ]
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
    with socket.create_connection(
        (parsed.hostname or "", parsed.port or 0), timeout=3
    ) as connection:
        connection.sendall(head + data)
        chunks: list[bytes] = []
        while chunk := connection.recv(65536):
            chunks.append(chunk)
    reply = b"".join(chunks)
    head_bytes, _, rest = reply.partition(b"\r\n\r\n")
    status = int(head_bytes.split(b" ", 2)[1])
    return status, json.loads(rest) if rest else {}


def test_the_corpus_is_the_exact_file_the_typescript_server_runs() -> None:
    text = RAW.decode("utf-8").replace("\r\n", "\n")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == CORPUS_SHA256
    assert len({row["id"] for row in CORPUS["cases"]}) == len(CORPUS["cases"])


@pytest.mark.parametrize(
    "row", CORPUS["cases"], ids=[f"{row['id']} {row['title']}" for row in CORPUS["cases"]]
)
def test_header_mirror_corpus(row: dict[str, Any]) -> None:
    async def scenario() -> None:
        calls: list[tuple[JsonObject, ToolContext]] = []
        gated: list[str] = []

        def gate(server: str, tool: str, args: JsonObject) -> bool:
            gated.append(tool)
            return True

        def handler(args: JsonObject, context: ToolContext) -> ToolResult:
            calls.append((args, context))
            return ToolResult("ok", False)

        server = Server("fixture", LoopbackHttpTransport(), gate=gate)
        for tool in CORPUS["tools"]:
            server.register(ToolDefinition(tool["name"], "", tool["inputSchema"]), handler)
        endpoint = await server.listen()
        token = endpoint["token"]
        assert isinstance(token, str)
        try:
            status, reply = await asyncio.to_thread(
                _post, endpoint["url"], token, [*_standard(row), *row["headers"]], _body(row)
            )
        finally:
            await server.close()
        expected = row["expect"]
        if expected["outcome"] == "served":
            assert status == 200
            assert "error" not in reply
            assert len(calls) == 1
            assert calls[0][0] == expected["arguments"]
            assert calls[0][1]["headers"] == expected["headers"]
            return
        assert calls == []
        assert gated == []
        if expected["outcome"] == "refused":
            assert status == 400
            assert reply == {
                "jsonrpc": "2.0",
                "id": row["id"],
                "error": {"code": -32020, "message": expected["message"]},
            }
        else:
            assert status == 200
            assert reply["error"]["code"] == -32000
            assert reply["error"]["data"] == {"code": "tool_call_failed"}

    asyncio.run(scenario())


def test_in_process_requires_no_header_but_checks_every_header_a_request_carries() -> None:
    async def scenario() -> None:
        calls: list[JsonObject] = []

        def handler(args: JsonObject, context: ToolContext) -> ToolResult:
            calls.append(args)
            return ToolResult("ok", False)

        transport = InMemoryTransport()
        server = Server("fixture", transport)
        server.register(ToolDefinition("route", "", CORPUS["tools"][0]["inputSchema"]), handler)
        await server.listen()

        def call(args: JsonObject, headers: dict[str, str]) -> TransportRequest:
            meta = {
                "io.modelcontextprotocol/protocolVersion": STATELESS,
                "io.modelcontextprotocol/clientCapabilities": {},
            }
            params = {"name": "route", "arguments": args, "_meta": meta}
            return TransportRequest("tools/call", params, headers)

        assert (await transport.request(call({"region": "us-west1"}, {})))["isError"] is False
        for args, headers in (
            ({}, {"Mcp-Param-Region": "us-west1"}),
            ({"region": "us-west1"}, {"Mcp-Param-Region": "eu-west1"}),
        ):
            with pytest.raises(HeaderMismatchError):
                await transport.request(call(args, headers))
        with pytest.raises(HeaderMismatchError) as error:
            await transport.request(
                call(
                    {"region": "us-west1"},
                    {"Mcp-Param-Region": "us-west1", "mcp-param-region": "us-west1"},
                )
            )
        assert error.value.message == (
            "Header mismatch: a header that mirrors the request body is repeated."
        )
        assert calls == [{"region": "us-west1"}]
        await server.close()

    asyncio.run(scenario())
