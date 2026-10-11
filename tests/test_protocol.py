from __future__ import annotations

import asyncio

import pytest
from prism_mcp import LATEST_PROTOCOL_VERSION

from prism_mcp_server import (
    ErrorCode,
    InMemoryTransport,
    JsonObject,
    McpError,
    Server,
    ToolDefinition,
    ToolResult,
    TransportRequest,
)

VERSION = "io.modelcontextprotocol/protocolVersion"
CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
SERVER_INFO = "io.modelcontextprotocol/serverInfo"
IDENTITY = {"name": "prism-ai-mcp-server", "version": "0.1.0"}


def request(method: str, client_info: bool = True) -> TransportRequest:
    return TransportRequest(
        method,
        {
            "name": "echo",
            "_meta": {
                VERSION: LATEST_PROTOCOL_VERSION,
                CAPABILITIES: {},
                **({CLIENT_INFO: {"name": "protocol-test", "version": "1"}} if client_info else {}),
            },
        },
    )


async def setup(is_error: bool = False) -> tuple[Server, InMemoryTransport]:
    transport = InMemoryTransport()
    server = Server("fixture", transport)
    server.register(ToolDefinition("echo", "", {}), lambda *_args: ToolResult("served", is_error))
    await server.listen()
    return server, transport


@pytest.mark.parametrize("method", ["tools/list", "tools/call"])
def test_serves_when_client_info_is_omitted(method: str) -> None:
    async def scenario() -> None:
        server, transport = await setup()
        try:
            result = await transport.request(request(method, False))
            if method == "tools/list":
                assert len(result["tools"]) == 1
            else:
                assert result["content"] == [{"type": "text", "text": "served"}]
        finally:
            await server.close()

    asyncio.run(scenario())


def test_answers_server_discover_with_supported_version_and_implemented_capabilities() -> None:
    async def scenario() -> None:
        server, transport = await setup()
        try:
            raw = request("server/discover")
            assert raw.params is not None
            del raw.params["name"]
            result = await transport.request(raw)
            assert result["supportedVersions"] == ["2026-07-28"]
            assert result["capabilities"] == {"tools": {}}
            assert result["resultType"] == "complete"
            assert result["_meta"] == {SERVER_INFO: IDENTITY}
            assert result["ttlMs"] == 0 and result["cacheScope"] == "private"
            assert "sessionId" not in result
            # Discovery is optional; a fresh server needs no preceding probe.
            fresh, other = await setup()
            try:
                assert (await other.request(request("tools/call")))["isError"] is False
            finally:
                await fresh.close()
        finally:
            await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method,is_error", [("tools/list", False), ("tools/call", False), ("tools/call", True)]
)
def test_includes_server_info_in_every_successful_result(method: str, is_error: bool) -> None:
    async def scenario() -> None:
        server, transport = await setup(is_error)
        try:
            result = await transport.request(request(method))
            assert result.get("_meta") == {SERVER_INFO: IDENTITY}
            assert result["resultType"] == "complete"
            if method == "tools/call":
                assert result["isError"] is is_error
            else:
                assert result["ttlMs"] == 0 and result["cacheScope"] == "private"
        finally:
            await server.close()

    asyncio.run(scenario())


def test_does_not_retain_client_identity_or_caller_mutations_between_results() -> None:
    async def scenario() -> None:
        server, transport = await setup()
        try:
            first = await transport.request(request("server/discover"))
            first["_meta"][SERVER_INFO]["name"] = "mutated"
            first["capabilities"]["resources"] = {}
            second = await transport.request(request("server/discover", False))
            assert second["_meta"] == {SERVER_INFO: IDENTITY}
            assert second["capabilities"] == {"tools": {}}
        finally:
            await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("version", [None, "wrong", "2025-11-25"])
@pytest.mark.parametrize("method", ["tools/list", "tools/call"])
def test_still_refuses_protocol_version_without_client_info(
    method: str, version: str | None
) -> None:
    async def scenario() -> None:
        server, transport = await setup()
        try:
            raw = request(method, False)
            assert raw.params is not None
            meta: JsonObject = raw.params["_meta"]
            if version is None:
                del meta[VERSION]
            else:
                meta[VERSION] = version
            with pytest.raises(McpError) as error:
                await transport.request(raw)
            assert error.value.code == ErrorCode.UNSUPPORTED_PROTOCOL_VERSION.value
        finally:
            await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["tools/list", "tools/call"])
def test_still_requires_client_capabilities_without_client_info(method: str) -> None:
    async def scenario() -> None:
        server, transport = await setup()
        try:
            raw = request(method, False)
            assert raw.params is not None
            del raw.params["_meta"][CAPABILITIES]
            with pytest.raises(McpError) as error:
                await transport.request(raw)
            assert error.value.code == ErrorCode.PROTOCOL_FAILURE.value
        finally:
            await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "params",
    [
        None,
        {},
        {"name": "ignored"},
        {"_meta": {}},
        {"_meta": {VERSION: "2025-11-25"}},
        {"_meta": {VERSION: "unknown"}},
        {"_meta": {VERSION: LATEST_PROTOCOL_VERSION}},
    ],
    ids=[
        "no params",
        "empty params",
        "no _meta",
        "empty _meta",
        "older version",
        "unknown version",
        "no capabilities",
    ],
)
def test_answers_discovery_before_negotiation(params: JsonObject | None) -> None:
    async def scenario() -> None:
        server, transport = await setup()
        try:
            before = await transport.request(request("tools/list"))
            result = await transport.request(TransportRequest("server/discover", params))
            assert result == {
                "supportedVersions": ["2026-07-28"],
                "capabilities": {"tools": {}},
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
                "_meta": {SERVER_INFO: IDENTITY},
            }
            assert await transport.request(request("tools/list")) == before
        finally:
            await server.close()

    asyncio.run(scenario())
