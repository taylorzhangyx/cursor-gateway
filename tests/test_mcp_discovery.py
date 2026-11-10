"""Modern MCP directory queries must precede custom-tool execution safely."""

import pytest

from cursor_gateway.config import Config
from cursor_gateway.proto import agent_pb2 as pb
from cursor_gateway.protocol import FrameDecoder, connect_frame
from cursor_gateway.session import run_turn
from cursor_gateway.translate import TurnInput


class FakeStream:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.sent = []
        self.headers = None

    def __call__(self, origin, path, headers, **kwargs):
        self.headers = headers
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def response_headers(self):
        return {":status": "200"}

    async def send(self, data):
        for frame in FrameDecoder().feed(data):
            self.sent.append(pb.AgentClientMessage.FromString(frame.payload))

    async def __anext__(self):
        try:
            return connect_frame(next(self.frames).SerializeToString())
        except StopIteration:
            raise StopAsyncIteration from None


def tool(name, provider="openai"):
    return pb.McpToolDefinition(
        name=name,
        tool_name=name,
        provider_identifier=provider,
        description="Local fixture lookup",
        input_schema_json='{"type":"object"}',
    )


def ended():
    return pb.AgentServerMessage(
        interaction_update=pb.InteractionUpdate(turn_ended=pb.TurnEndedUpdate())
    )


@pytest.mark.parametrize(
    "servers,expected", [([], ["openai", "secondary"]), (["openai"], ["openai"]), (["absent"], [])]
)
@pytest.mark.parametrize("kick_only", [False, True])
async def test_mcp_state_returns_only_declared_filtered_tools(
    monkeypatch, servers, expected, kick_only
):
    query = pb.AgentServerMessage(
        exec_server_message=pb.ExecServerMessage(
            id=73,
            exec_id="registry-73",
            mcp_state_exec_args=pb.McpStateExecArgs(
                server_identifiers=servers, kick_only=kick_only
            ),
        )
    )
    stream = FakeStream([query, ended()])
    monkeypatch.setattr("cursor_gateway.session.H2Stream", stream)
    tools = [tool("lookup"), tool("other", "secondary")]
    events = [
        event
        async for event in run_turn(
            token="fake", model="auto", config=Config(), turn=TurnInput([], "lookup", tools)
        )
    ]
    assert events[-1].reason == "stop", events[-1].error
    replies = [m.exec_client_message for m in stream.sent if m.HasField("exec_client_message")]
    assert len(replies) == 1
    reply = replies[0]
    assert (reply.id, reply.exec_id) == (73, "registry-73")
    assert reply.WhichOneof("message") == "mcp_state_exec_result"
    assert reply.mcp_state_exec_result.WhichOneof("result") == "success"
    actual = reply.mcp_state_exec_result.success.servers
    assert [server.server_identifier for server in actual] == expected
    for server in actual:
        assert server.server_name == server.server_identifier
        assert list(server.tools) == [
            t for t in tools if t.provider_identifier == server.server_identifier
        ]


async def test_discovery_then_native_dispatch_preserves_name_and_id(monkeypatch):
    stream = FakeStream(
        [
            pb.AgentServerMessage(
                exec_server_message=pb.ExecServerMessage(
                    id=11, exec_id="registry", mcp_state_exec_args=pb.McpStateExecArgs()
                )
            ),
            pb.AgentServerMessage(
                exec_server_message=pb.ExecServerMessage(
                    id=12,
                    exec_id="lookup-exec",
                    mcp_args=pb.McpArgs(
                        name="lookup", tool_name="lookup", tool_call_id="lookup-call"
                    ),
                )
            ),
        ]
    )
    monkeypatch.setattr("cursor_gateway.session.H2Stream", stream)
    events = [
        event
        async for event in run_turn(
            token="fake",
            model="auto",
            config=Config(),
            turn=TurnInput([], "lookup", [tool("lookup")]),
        )
    ]
    assert events[0].type == "tool_call", events[0].error
    assert (events[0].name, events[0].id) == ("lookup", "lookup-call")
    assert events[-1].reason == "tool_calls"
    advertised = stream.headers["x-cursor-agent-allowed-tools"].split(",")
    assert "mcp_tool_call" in advertised and "get_mcp_tools_tool_call" in advertised
    assert not any(name in advertised for name in ["read_tool_call", "shell_tool_call"])


@pytest.mark.parametrize(
    "allowed,with_tools,expected",
    [
        (("mcp_tool_call",), True, "mcp_tool_call,get_mcp_tools_tool_call"),
        (("mcp_tool_call",), False, "mcp_tool_call"),
        ((), True, None),
        (("read_tool_call",), True, "read_tool_call"),
        (
            ("mcp_tool_call", "get_mcp_tools_tool_call"),
            True,
            "mcp_tool_call,get_mcp_tools_tool_call",
        ),
    ],
)
async def test_discovery_advertisement_preserves_explicit_policy(
    monkeypatch, allowed, with_tools, expected
):
    stream = FakeStream([ended()])
    monkeypatch.setattr("cursor_gateway.session.H2Stream", stream)
    events = [
        event
        async for event in run_turn(
            token="fake",
            model="auto",
            config=Config(allowed_native_tools=allowed),
            turn=TurnInput([], "lookup", [tool("lookup")] if with_tools else []),
        )
    ]
    assert events[-1].reason == "stop"
    assert stream.headers.get("x-cursor-agent-allowed-tools") == expected


async def test_empty_directory_never_invents_tools(monkeypatch):
    stream = FakeStream(
        [
            pb.AgentServerMessage(
                exec_server_message=pb.ExecServerMessage(
                    id=99, exec_id="empty", mcp_state_exec_args=pb.McpStateExecArgs()
                )
            ),
            ended(),
        ]
    )
    monkeypatch.setattr("cursor_gateway.session.H2Stream", stream)
    events = [
        event
        async for event in run_turn(
            token="fake", model="auto", config=Config(), turn=TurnInput([], "hello", [])
        )
    ]
    assert events[-1].reason == "stop"
    result = stream.sent[1].exec_client_message.mcp_state_exec_result
    assert result.WhichOneof("result") == "success"
    assert not result.success.servers


async def test_host_native_execution_still_rejected(monkeypatch):
    stream = FakeStream(
        [
            pb.AgentServerMessage(
                exec_server_message=pb.ExecServerMessage(
                    id=1, exec_id="native-read", read_args=pb.ReadArgs()
                )
            )
        ]
    )
    monkeypatch.setattr("cursor_gateway.session.H2Stream", stream)
    events = [
        event
        async for event in run_turn(
            token="fake",
            model="auto",
            config=Config(),
            turn=TurnInput([], "lookup", [tool("lookup")]),
        )
    ]
    assert events[-1].reason == "error"
    assert "unsupported native tool: read_args" in events[-1].error
