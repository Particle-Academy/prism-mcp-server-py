from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from prism_mcp import (
    KNOWN_PROTOCOL_VERSIONS,
    LATEST_PROTOCOL_VERSION,
    Client,
)
from prism_mcp import (
    ToolDefinition as ClientToolDefinition,
)

from prism_mcp_server import (
    ErrorCode,
    InMemoryTransport,
    JsonObject,
    McpError,
    ResultGuard,
    Server,
    ToolContext,
    ToolDefinition,
    ToolGate,
    ToolResult,
    TransportRequest,
    TrustPolicy,
)

VERSION = "io.modelcontextprotocol/protocolVersion"
CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
RESULT_META = {
    "resultType": "complete",
    "_meta": {
        "io.modelcontextprotocol/serverInfo": {"name": "prism-ai-mcp-server", "version": "0.1.0"}
    },
}
CORPUS: JsonObject = json.loads(
    (Path(__file__).parent / "fixtures" / "mcp-tool-digest.json").read_text(encoding="utf-8")
)
CASES: list[JsonObject] = CORPUS["cases"]


def request(
    method: str = "tools/list",
    params: JsonObject | None = None,
    headers: dict[str, str] | None = None,
) -> TransportRequest:
    return TransportRequest(
        method,
        {
            **(params or {}),
            "_meta": {
                VERSION: LATEST_PROTOCOL_VERSION,
                CAPABILITIES: {},
                CLIENT_INFO: {"name": "phase-one-test", "version": "1"},
            },
        },
        headers or {},
    )


def definition(name: str = "echo", schema: JsonObject | None = None) -> ToolDefinition:
    return ToolDefinition(name, "Echo a value", schema or {}, "Echo", {"readOnlyHint": True})


def echo(args: JsonObject, _context: ToolContext) -> ToolResult:
    return ToolResult(json.dumps(args, separators=(",", ":")))


async def setup(
    gate: ToolGate | Callable[[str, str, JsonObject], Awaitable[bool]] | None = None,
    guard: ResultGuard | None = None,
) -> tuple[Server, InMemoryTransport]:
    transport = InMemoryTransport()
    server = Server("fixture", transport, gate, guard)
    await server.listen()
    return server, transport


def test_shared_runtime_class_and_entire_corpus() -> None:
    assert ToolDefinition is ClientToolDefinition
    assert len(CASES) == 13


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_real_tools_list_preserves_the_shared_digest(case: JsonObject) -> None:
    async def scenario() -> None:
        server, transport = await setup()
        tool = ToolDefinition.from_payload(case["payload"])
        server.register(tool, echo)
        reply = await transport.request(request())
        round_trip = ToolDefinition.from_payload(json.loads(json.dumps(reply["tools"][0])))
        assert round_trip.digest() == tool.digest()
        assert round_trip.digest() == case["digest"]["py"]
        assert round_trip.digest() == case["digest"]["ts"]
        assert round_trip.digest() == case["digest"]["php"]
        await server.close()

    asyncio.run(scenario())


def test_registration_and_list_replies_are_snapshots() -> None:
    async def scenario() -> None:
        server, transport = await setup()
        tool = definition(
            schema={
                "properties": {
                    "region": {"type": "string", "x-mcp-header": "Region"},
                }
            }
        )
        digest = tool.digest()
        server.register(tool, echo)
        tool.input_schema["properties"] = {"unsafe": {"type": "array", "x-mcp-header": "Region"}}
        first = await transport.request(request())
        first["tools"][0]["description"] = "mutated reply"
        second = await transport.request(request())
        assert ToolDefinition.from_payload(second["tools"][0]).digest() == digest
        assert await transport.request(
            request(
                "tools/call",
                {"name": "echo"},
                {
                    "Mcp-Param-region": "east",
                },
            )
        ) == {
            "content": [{"type": "text", "text": '{"region":"east"}'}],
            "isError": False,
            **RESULT_META,
        }
        await server.close()

    asyncio.run(scenario())


def test_duplicate_registration_does_not_replace_a_handler() -> None:
    async def scenario() -> None:
        server, transport = await setup()
        server.register(definition(), lambda _args, _context: ToolResult("original"))
        with pytest.raises(McpError, match="registered or refused"):
            server.register(definition(), echo)
        assert await transport.request(request("tools/call", {"name": "echo"})) == {
            "content": [{"type": "text", "text": "original"}],
            "isError": False,
            **RESULT_META,
        }
        await server.close()

    asyncio.run(scenario())


ILLEGAL: list[JsonObject] = [
    {"properties": {"value": {"type": "string", "x-mcp-header": ""}}},
    {"properties": {"value": {"type": "string", "x-mcp-header": "bad\r\nheader"}}},
    {"properties": {"value": {"type": "number", "x-mcp-header": "value"}}},
    {"properties": {"value": {"type": "array", "x-mcp-header": "value"}}},
    {"properties": {"value": {"type": "object", "x-mcp-header": "value"}}},
    {
        "properties": {
            "a": {"type": "string", "x-mcp-header": "Dup"},
            "b": {"type": "boolean", "x-mcp-header": "dup"},
        }
    },
]


@pytest.mark.parametrize("schema", ILLEGAL)
def test_illegal_mirrored_annotations_are_refused_and_excluded(schema: JsonObject) -> None:
    async def scenario() -> None:
        server, transport = await setup()
        called = False

        def handler(args: JsonObject, context: ToolContext) -> ToolResult:
            nonlocal called
            called = True
            return echo(args, context)

        with pytest.raises(McpError) as error:
            server.register(definition("bad", schema), handler)
        assert error.value.code == ErrorCode.MIRRORED_PARAMETER_REFUSED.value
        server.register(definition("valid"), echo)
        assert [item["name"] for item in (await transport.request(request()))["tools"]] == ["valid"]
        with pytest.raises(McpError) as refused:
            await transport.request(request("tools/call", {"name": "bad"}))
        assert refused.value.code == ErrorCode.MIRRORED_PARAMETER_REFUSED.value
        assert not called
        await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["authorization", "cookie"])
def test_credentials_are_never_visible_with_or_without_a_namespaced_mirror(name: str) -> None:
    async def scenario() -> None:
        server, transport = await setup()
        token = "Bearer " + secrets.token_hex(32)
        cookie = "session=PRIVATE-COOKIE-FIXTURE"
        observed: list[tuple[JsonObject, ToolContext]] = []

        def handler(args: JsonObject, context: ToolContext) -> ToolResult:
            observed.append((args, context))
            return echo(args, context)

        server.register(
            definition(
                "mirror",
                {
                    "properties": {
                        "value": {"type": "string", "x-mcp-header": name},
                    }
                },
            ),
            handler,
        )
        for mirrored in (False, True):
            headers = {
                "Authorization": token,
                "Cookie": cookie,
                "Origin": "https://untrusted.example",
            }
            if mirrored:
                headers[f"mCp-PaRaM-{name.upper()}"] = "public-mirrored-value"
            reply = await transport.request(request("tools/call", {"name": "mirror"}, headers))
            args, context = observed[-1]
            assert args == ({"value": "public-mirrored-value"} if mirrored else {})
            assert context == {
                "server": "fixture",
                "tool": "mirror",
                "headers": {
                    **({name: "public-mirrored-value"} if mirrored else {}),
                },
            }
            for item in (args, context, reply):
                assert token not in json.dumps(item)
                assert cookie not in json.dumps(item)
        await server.close()

    asyncio.run(scenario())


def test_nested_primitives_resolve_before_gate_without_mutating_the_caller() -> None:
    async def scenario() -> None:
        seen: list[JsonObject] = []

        def gate(_server: str, _tool: str, args: JsonObject) -> bool:
            seen.append(args)
            return True

        server, transport = await setup(gate=gate)
        server.register(
            definition(
                "nested",
                {
                    "properties": {
                        "options": {
                            "type": "object",
                            "properties": {
                                "region": {"type": "string", "x-mcp-header": "Region"},
                                "count": {"type": "integer", "x-mcp-header": "Count"},
                                "enabled": {"type": "boolean", "x-mcp-header": "Enabled"},
                            },
                        }
                    }
                },
            ),
            echo,
        )
        args = {"other": "untouched"}
        reply = await transport.request(
            request(
                "tools/call",
                {"name": "nested", "arguments": args},
                {
                    "Mcp-Param-Region": "east",
                    "MCP-PARAM-COUNT": "0",
                    "mcp-param-enabled": "false",
                },
            )
        )
        assert seen == [
            {"other": "untouched", "options": {"region": "east", "count": 0, "enabled": False}}
        ]
        assert args == {"other": "untouched"}
        assert reply == {
            "content": [
                {
                    "type": "text",
                    "text": '{"other":"untouched","options":{"region":"east","count":0,"enabled":false}}',
                }
            ],
            "isError": False,
            **RESULT_META,
        }
        await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["1.5", "01", "+1", "9007199254740992", "false"])
def test_malformed_mirrored_integers_never_reach_the_handler(value: str) -> None:
    async def scenario() -> None:
        server, transport = await setup()
        server.register(
            definition(
                "integer",
                {
                    "properties": {
                        "value": {"type": "integer", "x-mcp-header": "N"},
                    }
                },
            ),
            lambda _args, _context: pytest.fail("Handler must not run"),
        )
        with pytest.raises(McpError) as error:
            await transport.request(
                request("tools/call", {"name": "integer"}, {"Mcp-Param-N": value})
            )
        assert error.value.code == ErrorCode.PROTOCOL_FAILURE.value
        await server.close()

    asyncio.run(scenario())


def test_body_conflicts_and_duplicate_header_casing_fail_before_the_gate() -> None:
    async def scenario() -> None:
        server, transport = await setup(gate=lambda *_args: pytest.fail("Gate must not run"))
        server.register(
            definition(
                "mirror",
                {
                    "properties": {
                        "value": {"type": "string", "x-mcp-header": "V"},
                    }
                },
            ),
            echo,
        )
        for raw in (
            request(
                "tools/call",
                {"name": "mirror", "arguments": {"value": "body"}},
                {"Mcp-Param-V": "header"},
            ),
            request("tools/call", {"name": "mirror"}, {"Mcp-Param-V": "one", "mcp-param-v": "two"}),
        ):
            with pytest.raises(McpError) as error:
                await transport.request(raw)
            assert error.value.code == ErrorCode.PROTOCOL_FAILURE.value
        await server.close()

    asyncio.run(scenario())


def test_prototype_shaped_paths_are_ordinary_dictionary_fields() -> None:
    async def scenario() -> None:
        server, transport = await setup()
        schema = json.loads(
            '{"properties":{"__proto__":{"properties":{"polluted":{"type":"string","x-mcp-header":"safe"}}}}}'
        )
        server.register(definition("prototype", schema), echo)
        reply = await transport.request(
            request("tools/call", {"name": "prototype"}, {"Mcp-Param-safe": "data"})
        )
        assert json.loads(reply["content"][0]["text"]) == {"__proto__": {"polluted": "data"}}
        await server.close()

    asyncio.run(scenario())


def test_round_trip_through_the_actual_sibling_client() -> None:
    async def scenario() -> None:
        server, transport = await setup()
        tool = definition(
            schema={"properties": {"region": {"type": "string", "x-mcp-header": "region"}}}
        )
        result = ToolResult("exact-text", True)

        async def handler(args: JsonObject, context: ToolContext) -> ToolResult:
            assert args == {"region": "west"}
            assert context["headers"] == {"region": "west"}
            return result

        # The client is async since prism-ai-mcp 0.2.0, so its transport awaits
        # the server directly rather than spinning a loop per request.
        async def client_transport(raw: TransportRequest) -> Any:
            return await transport.request(request(raw.method, raw.params, raw.headers))

        server.register(tool, handler)
        client = Client(
            "fixture",
            client_transport,
            trust=TrustPolicy.allowing_every_tool({"echo": tool.digest()}),
            guard=ResultGuard(frame_provenance=False),
        )
        listed = await client.list_tools()
        assert listed[0].digest() == tool.digest()
        assert await client.call_tool(listed[0], {"region": "west"}) == result
        await server.close()

    asyncio.run(scenario())


def test_gate_resolving_to_another_awaitable_is_refused() -> None:
    # An async gate that delegates and forgets the inner await resolves to a
    # second coroutine, which is truthy. Awaiting once would read that as
    # consent: the G-83 bypass one level down.
    async def check(*_args: Any) -> bool:
        return False

    async def gate(*args: Any) -> bool:
        return check(*args)  # type: ignore[return-value]

    async def scenario() -> None:
        server, transport = await setup(gate=gate)
        server.register(definition(), lambda *_args: pytest.fail("Handler must not run"))
        with pytest.raises(TypeError, match="gate must resolve to a verdict"):
            await transport.request(request("tools/call", {"name": "echo"}))
        await server.close()

    asyncio.run(scenario())


def test_custom_gate_refuses_before_execution() -> None:
    async def scenario() -> None:
        server, transport = await setup(gate=lambda *_args: False)
        server.register(definition(), lambda *_args: pytest.fail("Handler must not run"))
        with pytest.raises(McpError) as error:
            await transport.request(request("tools/call", {"name": "echo"}))
        assert error.value.code == ErrorCode.TOOL_DENIED.value
        await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("verdict", [False, True])
def test_async_gate_is_awaited_before_execution(verdict: bool) -> None:
    async def scenario() -> None:
        called: list[str] = []

        async def gate(_server: str, _tool: str, _args: JsonObject) -> bool:
            called.append("gate body")
            return verdict

        server, transport = await setup(gate=gate)

        def handler(_args: JsonObject, _context: ToolContext) -> ToolResult:
            called.append("handler")
            return ToolResult("approved")

        server.register(definition(), handler)
        try:
            if verdict:
                assert await transport.request(request("tools/call", {"name": "echo"})) == {
                    "content": [{"type": "text", "text": "approved"}],
                    "isError": False,
                    **RESULT_META,
                }
                assert called == ["gate body", "handler"]
            else:
                with pytest.raises(McpError) as error:
                    await transport.request(request("tools/call", {"name": "echo"}))
                assert error.value.code == ErrorCode.TOOL_DENIED.value
                assert called == ["gate body"]
        finally:
            await server.close()

    asyncio.run(scenario())


def test_future_gate_is_awaited_before_execution() -> None:
    async def scenario() -> None:
        verdict: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        gate_started = asyncio.Event()

        def gate(_server: str, _tool: str, _args: JsonObject) -> Awaitable[bool]:
            gate_started.set()
            return verdict

        server, transport = await setup(gate=gate)
        called: list[str] = []

        def handler(_args: JsonObject, _context: ToolContext) -> ToolResult:
            called.append("handler")
            return ToolResult("must not execute")

        server.register(definition(), handler)
        try:
            pending = asyncio.create_task(
                transport.request(request("tools/call", {"name": "echo"}))
            )
            await gate_started.wait()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert called == []
            assert not pending.done()
            verdict.set_result(False)
            with pytest.raises(McpError) as error:
                await pending
            assert error.value.code == ErrorCode.TOOL_DENIED.value
            assert called == []
        finally:
            verdict.cancel()
            await server.close()

    asyncio.run(scenario())


def test_in_memory_endpoint_has_no_credential() -> None:
    async def scenario() -> None:
        transport = InMemoryTransport()
        server = Server("fixture", transport)
        assert await server.listen() == {"url": "memory://prism-mcp-server", "token": None}
        await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("is_error", [False, True])
def test_default_cap_refuses_success_and_error_results_without_truncation(is_error: bool) -> None:
    async def scenario() -> None:
        server, transport = await setup()
        server.register(definition(), lambda *_args: ToolResult("x" * 65537, is_error))
        with pytest.raises(McpError) as error:
            await transport.request(request("tools/call", {"name": "echo"}))
        assert error.value.code == ErrorCode.RESULT_TOO_LARGE.value
        await server.close()

    asyncio.run(scenario())


def test_result_guard_counts_utf8_bytes() -> None:
    async def scenario() -> None:
        server, transport = await setup(guard=ResultGuard(max_bytes=4, frame_provenance=False))
        server.register(definition(), lambda args, _context: ToolResult(str(args["text"])))
        assert await transport.request(
            request("tools/call", {"name": "echo", "arguments": {"text": "😀"}})
        ) == {
            "content": [{"type": "text", "text": "😀"}],
            "isError": False,
            **RESULT_META,
        }
        with pytest.raises(McpError) as error:
            await transport.request(
                request("tools/call", {"name": "echo", "arguments": {"text": "😀😀"}})
            )
        assert error.value.code == ErrorCode.RESULT_TOO_LARGE.value
        await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "version", [v for v in KNOWN_PROTOCOL_VERSIONS if v != LATEST_PROTOCOL_VERSION]
)
def test_older_revisions_are_refused_by_name(version: str) -> None:
    async def scenario() -> None:
        server, transport = await setup()
        raw = request()
        assert raw.params is not None
        raw.params["_meta"][VERSION] = version
        with pytest.raises(McpError) as error:
            await transport.request(raw)
        assert error.value.code == ErrorCode.UNSUPPORTED_PROTOCOL_VERSION.value
        assert version in error.value.message
        assert LATEST_PROTOCOL_VERSION in error.value.message
        await server.close()

    asyncio.run(scenario())


def test_malformed_arguments_and_unsupported_methods_fail_closed() -> None:
    async def scenario() -> None:
        server, transport = await setup()
        server.register(definition(), echo)
        for raw in (
            TransportRequest("tools/list"),
            request("initialize"),
            request("resources/list"),
            request("tools/call", {"name": "echo", "arguments": []}),
        ):
            with pytest.raises(McpError) as error:
                await transport.request(raw)
            assert error.value.code == ErrorCode.PROTOCOL_FAILURE.value
        with pytest.raises(McpError) as error:
            await transport.request(request("tools/call", {"name": "missing"}))
        assert error.value.code == ErrorCode.TOOL_CALL_FAILED.value
        await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("shared", [False, True])
def test_handler_error_mapping_excludes_sensitive_diagnostics(shared: bool) -> None:
    async def scenario() -> None:
        server, transport = await setup()

        def handler(_args: JsonObject, _context: ToolContext) -> ToolResult:
            if shared:
                raise McpError(ErrorCode.TOOL_DENIED, "SECRET-DIAGNOSTIC")
            raise RuntimeError("SECRET-DIAGNOSTIC")

        server.register(definition(), handler)
        with pytest.raises(McpError) as error:
            await transport.request(request("tools/call", {"name": "echo"}))
        assert error.value.code == (
            ErrorCode.TOOL_DENIED.value if shared else ErrorCode.TOOL_CALL_FAILED.value
        )
        assert error.value.message == "The tool handler could not complete the request."
        await server.close()

    asyncio.run(scenario())


def test_long_poll_survives_a_virtual_ten_minutes_and_completes() -> None:
    async def scenario() -> None:
        server, transport = await setup()
        started, finish = asyncio.Event(), asyncio.Event()

        async def handler(_args: JsonObject, _context: ToolContext) -> ToolResult:
            started.set()
            await finish.wait()
            return ToolResult("long-poll-complete")

        server.register(definition("receive"), handler)
        pending = asyncio.create_task(transport.request(request("tools/call", {"name": "receive"})))
        await started.wait()
        loop = asyncio.get_running_loop()
        with patch.object(loop, "time", return_value=loop.time() + 600):
            await asyncio.sleep(0)
            assert not pending.done()
            finish.set()
            assert await pending == {
                "content": [{"type": "text", "text": "long-poll-complete"}],
                "isError": False,
                **RESULT_META,
            }
        await server.close()

    asyncio.run(scenario())


def test_close_releases_long_poll_even_when_application_code_suppresses_cancellation() -> None:
    async def scenario() -> None:
        server, transport = await setup()
        started, finish, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def handler(_args: JsonObject, _context: ToolContext) -> ToolResult:
            started.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await finish.wait()
            return ToolResult("late handler completion")

        server.register(definition("receive"), handler)
        pending = asyncio.create_task(transport.request(request("tools/call", {"name": "receive"})))
        await started.wait()
        await server.close()
        with pytest.raises(McpError) as error:
            await pending
        assert error.value.code == ErrorCode.PROTOCOL_FAILURE.value
        await server.close()
        with pytest.raises(McpError):
            await transport.request(request())
        with pytest.raises(McpError):
            server.register(definition("later"), echo)
        await cancelled.wait()
        finish.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(scenario())


# A NULL IS NOT AN OPTIONAL STRING (tynn, Genie 2, 2026-10-22). Claude Code dropped the whole server on one
# `title: null`, so a single tool without a title hid every tool the server had.
def test_a_tool_with_no_title_is_listed_without_one_and_its_pin_holds() -> None:
    async def scenario() -> None:
        server, transport = await setup()
        untitled = ToolDefinition("plain", "No title.", {"type": "object"})
        assert untitled.title is None
        server.register(untitled, echo)
        server.register(definition("titled"), echo)
        reply = await transport.request(request())
        tools = json.loads(json.dumps(reply["tools"]))
        plain = next(tool for tool in tools if tool["name"] == "plain")
        assert "title" not in plain
        assert all(value is not None for tool in tools for value in tool.values())
        assert next(tool for tool in tools if tool["name"] == "titled")["title"] == "Echo"
        assert ClientToolDefinition.from_payload(plain).digest() == untitled.digest()
        await server.close()

    asyncio.run(scenario())
