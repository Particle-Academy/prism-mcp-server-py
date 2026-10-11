from __future__ import annotations

import asyncio
import hashlib
import hmac
import http.client
import json
import socket
import time
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
from prism_mcp import LATEST_PROTOCOL_VERSION, ToolDefinition, ToolResult

from prism_mcp_server import JsonObject, LoopbackHttpTransport, McpError, Server, ToolContext
from prism_mcp_server.http import _HttpServer


def rpc(method: str = "tools/list", params: JsonObject | None = None) -> JsonObject:
    return {
        "jsonrpc": "2.0",
        "id": "http-test",
        "method": method,
        "params": {
            **(params or {}),
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": LATEST_PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
                "io.modelcontextprotocol/clientInfo": {"name": "http-test", "version": "1"},
            },
        },
    }


async def setup(**options: Any) -> tuple[Server, LoopbackHttpTransport, str, str]:
    transport = LoopbackHttpTransport(**options)
    server = Server("fixture", transport)
    endpoint = await server.listen()
    assert isinstance(endpoint["token"], str)
    return server, transport, endpoint["url"], endpoint["token"]


def _send(
    url: str,
    token: str | None = None,
    headers: dict[str, str | list[str]] | None = None,
    body: Any = None,
    method: str = "POST",
    timeout: float | None = 3,
) -> tuple[int, JsonObject]:
    parsed = urlsplit(url)
    connection = http.client.HTTPConnection(parsed.hostname or "", parsed.port, timeout=timeout)
    data = json.dumps(rpc() if body is None else body).encode("utf-8")
    values: dict[str, str | list[str]] = {
        "Content-Type": "application/json",
        "Content-Length": str(len(data)),
        **({"Authorization": f"Bearer {token}"} if token is not None else {}),
        **(headers or {}),
    }
    try:
        connection.putrequest(method, parsed.path)
        for name, value in values.items():
            for item in value if isinstance(value, list) else [value]:
                connection.putheader(name, item)
        connection.endheaders(data)
        response = connection.getresponse()
        payload: JsonObject = json.loads(response.read())
        return response.status, payload
    finally:
        connection.close()


async def send(
    url: str,
    token: str | None = None,
    headers: dict[str, str | list[str]] | None = None,
    body: Any = None,
    method: str = "POST",
    timeout: float | None = 3,
) -> tuple[int, JsonObject]:
    return await asyncio.to_thread(_send, url, token, headers, body, method, timeout)


def run(scenario: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(asyncio.wait_for(scenario(), 15))


def test_literal_loopback_bind_is_unreachable_on_non_loopback_ipv4() -> None:
    async def scenario() -> None:
        bind_targets: list[str] = []

        def bind(server: _HttpServer) -> None:
            bind_targets.append(str(server.server_address[0]))
            super(_HttpServer, server).server_bind()

        with patch.object(_HttpServer, "server_bind", bind):
            # Record the actual bind target while still using the real server.
            server, _, url, token = await setup()
            try:
                assert bind_targets == ["127.0.0.1"]
                assert urlsplit(url).hostname == "127.0.0.1"
                assert (await send(url, token))[0] == 200
                for entry in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                    address = str(entry[4][0])
                    if address.startswith("127."):
                        continue
                    with pytest.raises(OSError):
                        await send(url.replace("127.0.0.1", address), token, timeout=1)
            finally:
                await server.close()

    run(scenario)


def test_tokens_are_generated_at_startup_and_differ_between_instances() -> None:
    async def scenario() -> None:
        first, _, url, token = await setup()
        second, _, _, other = await setup()
        try:
            assert len(token) == 64 and len(bytes.fromhex(token)) == 32
            assert token != other and token not in url
            assert (await send(url, other))[0] == 401
        finally:
            await first.close()
            await second.close()

    run(scenario)


@pytest.mark.parametrize("token", ["", " ", "\t\r\n", " padded ", "bad token", "valid\n"])
def test_invalid_configured_token_refuses_to_listen(token: str) -> None:
    async def scenario() -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        transport = LoopbackHttpTransport(token=token, port=port)
        try:
            with patch.object(_HttpServer, "server_bind", autospec=True) as bound:
                bound.side_effect = lambda server: super(_HttpServer, server).server_bind()
                with pytest.raises(McpError):
                    await transport.listen(lambda _: asyncio.sleep(0))
                bound.assert_not_called()
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        finally:
            await transport.close()

    run(scenario)


def test_absent_wrong_one_byte_empty_and_duplicate_credentials_refuse_before_dispatch() -> None:
    async def scenario() -> None:
        server, _, url, token = await setup()
        called: list[str] = []

        def handler(_args: JsonObject, _context: ToolContext) -> ToolResult:
            called.append("handler")
            return ToolResult("authorized")

        server.register(ToolDefinition("run", "", {}), handler)
        call = rpc("tools/call", {"name": "run"})
        try:
            different = ("b" if token[0] == "a" else "a") + token[1:]
            for candidate in (None, "wrong", different, "", " "):
                assert (await send(url, candidate, body=call))[0] == 401
            assert (
                await send(
                    url,
                    headers={"Authorization": [f"Bearer {token}", f"Bearer {token}"]},
                    body=call,
                )
            )[0] == 401
            assert called == []
            assert (await send(url, token, body=call))[0] == 200
            assert called == ["handler"]
        finally:
            await server.close()

    run(scenario)


def test_constant_time_primitive_compares_fixed_size_digests_for_all_token_lengths() -> None:
    async def scenario() -> None:
        server, _, url, token = await setup()
        try:
            with patch.object(hmac, "compare_digest", wraps=hmac.compare_digest) as compare:
                for candidate in (token, "x", "x" * 100):
                    await send(url, candidate)
                assert compare.call_count == 3
                for call in compare.call_args_list:
                    assert len(call.args[0]) == len(call.args[1]) == hashlib.sha256().digest_size
        finally:
            await server.close()

    run(scenario)


@pytest.mark.parametrize(
    "origin",
    [
        "https://hostile.example",
        "http://127.0.0.1.evil.example",
        "http://localhost:1234",
        "http://127.0.0.2:1234",
        "null",
        "",
        "http://127.0.0.1:1234",
    ],
)
def test_hostile_or_unlisted_browser_origin_never_dispatches(origin: str) -> None:
    async def scenario() -> None:
        server, _, url, token = await setup()
        called: list[str] = []

        def handler(_args: JsonObject, _context: ToolContext) -> ToolResult:
            called.append("handler")
            return ToolResult("secret")

        server.register(ToolDefinition("run", "", {}), handler)
        try:
            assert (await send(url, token, {"Origin": origin}, rpc("tools/call", {"name": "run"})))[
                0
            ] == 403
            assert called == []
        finally:
            await server.close()

    run(scenario)


def test_only_own_origin_or_configured_exact_loopback_origins_are_admitted() -> None:
    async def scenario() -> None:
        own, _, url, token = await setup()
        configured, _, other_url, other_token = await setup(
            allowed_origins=["http://127.0.0.1:1234", "https://[::1]:4567"]
        )
        try:
            assert (await send(url, token, {"Origin": url.removesuffix("/mcp")}))[0] == 200
            for origin in ("http://127.0.0.1:1234", "https://[::1]:4567"):
                assert (await send(other_url, other_token, {"Origin": origin}))[0] == 200
            assert (await send(other_url, other_token, {"Origin": other_url.removesuffix("/mcp")}))[
                0
            ] == 403
        finally:
            await own.close()
            await configured.close()

    run(scenario)


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        "http://0.0.0.0:1",
        "http://127.1:1",
        "http://localhost:1",
        "http://127.0.0.1:70000",
        "http://127.0.0.1:1/path",
    ],
)
def test_non_loopback_or_malformed_allowlist_refuses_startup(origin: str) -> None:
    async def scenario() -> None:
        transport = LoopbackHttpTransport(allowed_origins=[origin])
        with pytest.raises(McpError):
            await transport.listen(lambda _: asyncio.sleep(0))
        await transport.close()

    run(scenario)


def test_get_sse_upgrade_connect_and_sse_only_post_are_refused() -> None:
    async def scenario() -> None:
        server, _, url, token = await setup()
        try:
            for candidate in (None, token):
                cases: list[tuple[str, dict[str, str | list[str]]]] = [
                    ("GET", {"Accept": "text/event-stream"}),
                    ("GET", {"Connection": "Upgrade", "Upgrade": "websocket"}),
                    ("CONNECT", {}),
                ]
                for method, headers in cases:
                    assert (await send(url, candidate, headers, method=method))[0] == (
                        405 if candidate else 401
                    )
            assert (await send(url, token, {"Accept": "text/event-stream"}))[0] == 406
            assert (await send(url, token, {"Accept": "application/json, text/event-stream"}))[
                0
            ] == 200
        finally:
            await server.close()

    run(scenario)


def test_http_preserves_pin_mirrors_and_credential_boundary_with_and_without_mirror() -> None:
    async def scenario() -> None:
        server, _, url, token = await setup()
        tool = ToolDefinition(
            "mirror",
            "",
            {"properties": {"value": {"type": "string", "x-mcp-header": "authorization"}}},
        )
        observed: list[tuple[JsonObject, ToolContext]] = []

        def handler(args: JsonObject, context: ToolContext) -> ToolResult:
            observed.append((args, context))
            return ToolResult(json.dumps({"args": args, "context": context}), True)

        server.register(tool, handler)
        try:
            _, listed = await send(url, token)
            assert (
                ToolDefinition.from_payload(listed["result"]["tools"][0]).digest() == tool.digest()
            )
            for mirror in (False, True):
                headers: dict[str, str | list[str]] = {"Cookie": "PRIVATE-COOKIE"}
                if mirror:
                    headers["Mcp-Param-authorization"] = "public"
                _, reply = await send(url, token, headers, rpc("tools/call", {"name": "mirror"}))
                assert observed[-1][0] == ({"value": "public"} if mirror else {})
                assert observed[-1][1]["headers"] == ({"authorization": "public"} if mirror else {})
                assert token not in json.dumps(reply) and "PRIVATE-COOKIE" not in json.dumps(reply)
                assert reply["result"]["isError"] is True
        finally:
            await server.close()

    run(scenario)


def test_auth_and_origin_precede_expect_continue_and_unusual_methods() -> None:
    async def scenario() -> None:
        server, _, url, token = await setup()
        try:
            for method in ("POST", "TRACE"):
                assert (await send(url, method=method))[0] == 401
                assert (
                    await send(url, token, {"Origin": "https://hostile.example"}, method=method)
                )[0] == 403
            assert (await send(url, headers={"Expect": "100-continue"}))[0] == 401
            assert (await send(url, token, {"Expect": "100-continue"}))[0] == 200
        finally:
            await server.close()

    run(scenario)


def test_shared_errors_and_oversized_results_are_mapped_without_diagnostic_leaks() -> None:
    async def scenario() -> None:
        server, _, url, token = await setup()
        server.register(
            ToolDefinition("large", "", {}), lambda *_args: ToolResult("x" * 65537, True)
        )

        def handler(_args: JsonObject, _context: ToolContext) -> ToolResult:
            raise RuntimeError(f"PRIVATE-{token}")

        server.register(ToolDefinition("throws", "", {}), handler)
        try:
            for name, code in (("large", "result_too_large"), ("throws", "tool_call_failed")):
                status, reply = await send(url, token, body=rpc("tools/call", {"name": name}))
                assert status == 200 and reply["error"]["data"]["code"] == code
                assert token not in json.dumps(reply)
        finally:
            await server.close()

    run(scenario)


def test_body_limit_malformed_rpc_content_type_and_duplicate_mirrors_are_refused() -> None:
    async def scenario() -> None:
        server, _, url, token = await setup(max_request_bytes=512)
        try:
            assert (await send(url, token, body=rpc(params={"padding": "x" * 513})))[0] == 413
            bodies: list[Any] = [
                [],
                {},
                {"jsonrpc": "2.0", "id": True, "method": "tools/list"},
                {**rpc(), "params": []},
            ]
            for body in bodies:
                assert (await send(url, token, body=body))[0] == 400
            assert (await send(url, token, {"Content-Type": "text/plain"}))[0] == 415
            assert (await send(url, token, {"Mcp-Param-X": ["one", "two"]}))[0] == 400
            assert (await send(url.replace("/mcp", "/other"), token))[0] == 404
        finally:
            await server.close()

    run(scenario)


def test_accepted_socket_timeout_is_explicitly_unlimited() -> None:
    async def scenario() -> None:
        original = _HttpServer.get_request
        observed: list[float | None] = []

        def capture(server: _HttpServer) -> tuple[socket.socket, Any]:
            connection, address = original(server)
            observed.append(connection.gettimeout())
            return connection, address

        old_default = socket.getdefaulttimeout()
        socket.setdefaulttimeout(0.1)
        try:
            with patch.object(_HttpServer, "get_request", capture):
                server, _, url, token = await setup()
                try:
                    assert (await send(url, token))[0] == 200
                    assert observed == [None]
                finally:
                    await server.close()
        finally:
            socket.setdefaulttimeout(old_default)

    run(scenario)


def test_real_long_poll_held_past_300_seconds_completes() -> None:
    async def scenario() -> None:
        server, _, url, token = await setup()
        started, finish = asyncio.Event(), asyncio.Event()

        async def handler(_args: JsonObject, _context: ToolContext) -> ToolResult:
            started.set()
            await finish.wait()
            return ToolResult("long-poll-complete")

        server.register(ToolDefinition("receive", "", {}), handler)
        pending = asyncio.create_task(
            send(url, token, body=rpc("tools/call", {"name": "receive"}), timeout=330)
        )
        try:
            await asyncio.wait_for(started.wait(), 3)
            before = time.monotonic()
            await asyncio.sleep(310)
            assert time.monotonic() - before > 300
            assert not pending.done()
            finish.set()
            status, reply = await pending
            assert status == 200
            assert reply["result"]["content"] == [{"type": "text", "text": "long-poll-complete"}]
        finally:
            await server.close()
            await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(asyncio.wait_for(scenario(), 340))


def test_close_releases_held_polls_even_with_suppressed_cancellation_and_releases_port() -> None:
    async def scenario() -> None:
        server, transport, url, token = await setup()
        started, finish, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def handler(_args: JsonObject, _context: ToolContext) -> ToolResult:
            started.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await finish.wait()
            return ToolResult("late")

        server.register(ToolDefinition("receive", "", {}), handler)
        pending = asyncio.create_task(send(url, token, body=rpc("tools/call", {"name": "receive"})))
        idle: socket.socket | None = None
        try:
            await asyncio.wait_for(started.wait(), 3)
            port = urlsplit(url).port
            assert port is not None
            idle = socket.create_connection(("127.0.0.1", port), timeout=2)
            idle.sendall(b"POST /mcp HTTP/1.1\r\n")
            await asyncio.wait_for(asyncio.gather(server.close(), transport.close()), 2)
            with pytest.raises(OSError):
                await pending
            try:
                assert await asyncio.to_thread(idle.recv, 1) == b""
            except ConnectionResetError:
                pass
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
                probe.listen()
            with pytest.raises(McpError):
                await transport.listen(lambda _: asyncio.sleep(0))
            await asyncio.wait_for(cancelled.wait(), 2)
        finally:
            finish.set()
            if idle is not None:
                idle.close()
            await server.close()
            await asyncio.gather(pending, return_exceptions=True)
            await asyncio.sleep(0)

    run(scenario)
