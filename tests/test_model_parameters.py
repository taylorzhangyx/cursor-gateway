"""Parameter selection must reach Cursor's real Run wire, not only display IDs."""

from dataclasses import replace

import httpx
import pytest
from test_integration import upstream
from test_mcp_discovery import FakeStream, ended

from cursor_gateway.auth import Account, write_accounts
from cursor_gateway.config import Config
from cursor_gateway.session import run_turn
from cursor_gateway.translate import RequestError, TurnInput

SPEC = "grok-4.7[context=500k,reasoning_effort=xhigh,fast=true]"
PARAMS = {"context": "500k", "reasoning_effort": "xhigh", "fast": "true"}


@pytest.mark.parametrize(
    "model,base,legacy,parameters,max_mode",
    [
        (SPEC, "grok-4.7", "grok-4.7-xhigh-fast", PARAMS, True),
        (
            "grok-4.7[context=256k,reasoning_effort=low,fast=false]",
            "grok-4.7",
            "grok-4.7-low",
            {"context": "256k", "reasoning_effort": "low", "fast": "false"},
            False,
        ),
        ("glm-5p3[reasoning=max]", "glm-5p3", "glm-5p3", {"reasoning": "max"}, None),
        ("grok-4.7-xhigh-fast", "grok-4.7-xhigh-fast", "grok-4.7-xhigh-fast", {}, None),
    ],
)
async def test_model_parameters_are_serialized_on_run(
    monkeypatch, model, base, legacy, parameters, max_mode
):
    stream = FakeStream([ended()])
    monkeypatch.setattr("cursor_gateway.session.H2Stream", stream)
    events = [
        event
        async for event in run_turn(
            token="fake-token", model=model, turn=TurnInput([], "hello", []), config=Config()
        )
    ]
    assert events[-1].reason == "stop"
    run = stream.sent[0].run_request
    assert run.model_details.model_id == legacy
    assert run.requested_model.model_id == base
    assert {p.id: p.value for p in run.requested_model.parameters} == parameters
    if max_mode is None:
        assert not run.model_details.HasField("max_mode")
    else:
        assert run.model_details.HasField("max_mode")
        assert run.model_details.max_mode is max_mode
        assert run.requested_model.max_mode is max_mode


@pytest.mark.parametrize(
    "path,stream",
    [
        ("/v1/chat/completions", False),
        ("/v1/chat/completions", True),
        ("/v1/responses", False),
        ("/v1/responses", True),
        ("/v1/responses/compact", False),
    ],
)
@pytest.mark.parametrize("spec", [SPEC, "grok-4.7-xhigh-fast[context=500k]"])
async def test_parameterized_selection_reaches_real_h2(tmp_path, path, stream, spec):
    async with upstream() as state:
        file = tmp_path / "accounts.json"
        write_accounts(file, [Account("test", "fake-token")])
        from cursor_gateway.server import create_app

        app = create_app(replace(state.config, auth_file=file, model_prefix="gateway-"))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            body = {"model": "gateway-" + spec, "stream": stream}
            if path == "/v1/chat/completions":
                body["messages"] = [{"role": "user", "content": "hello"}]
            else:
                body["input"] = "hello"
            response = await client.post(path, json=body)
            assert response.status_code == 200, response.text
        assert len(state.runs) == 1
        run = state.runs[0]
        assert run.model_details.model_id == "grok-4.7-xhigh-fast"
        assert run.requested_model.model_id == "grok-4.7"
        assert {p.id: p.value for p in run.requested_model.parameters} == PARAMS
        assert run.model_details.max_mode and run.requested_model.max_mode


@pytest.mark.parametrize(
    "model",
    [
        "grok-4.7[context=500k",
        "grok-4.7context=500k]",
        "grok-4.7[]",
        "grok-4.7[context=500k,context=256k]",
        "grok-4.7[context=]",
        "grok-4.7[context=500k,]",
        "grok-4.7[context=500k]junk",
        "grok-4.7[context=1m]",
        "grok-4.7[fast=1]",
        "grok-4.7[reasoning_effort=invalid]",
        "grok-4.7[effort=xhigh]",
    ],
)
async def test_invalid_parameterized_models_fail_closed(tmp_path, model):
    async with upstream() as state:
        file = tmp_path / "accounts.json"
        write_accounts(file, [Account("test", "fake-token")])
        from cursor_gateway.server import create_app

        app = create_app(replace(state.config, auth_file=file))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/responses", json={"model": model, "input": "hi"})
        assert response.status_code == 400, response.text
        assert response.json()["error"]["param"] == "model"
        assert not state.runs


def test_parameter_parser_preserves_values_and_rejects_whitespace():
    from cursor_gateway.model_parameters import parse_model_spec

    spec = parse_model_spec("custom[version=V1.2,fast=false]")
    assert spec.model_id == "custom"
    assert dict(spec.parameters) == {"version": "V1.2", "fast": "false"}
    with pytest.raises(RequestError):
        parse_model_spec("grok-4.7[fast= true]")


@pytest.mark.parametrize(
    "model,expected",
    [
        ("grok-4.7-xhigh-fast[context=500k]", PARAMS),
        (
            "grok-4.7-low[context=256k]",
            {"context": "256k", "reasoning_effort": "low", "fast": "false"},
        ),
    ],
)
def test_legacy_parameter_overrides_preserve_profile(model, expected):
    from cursor_gateway.model_parameters import parse_model_spec

    result = parse_model_spec(model)
    assert result.model_id == "grok-4.7"
    assert dict(result.parameters) == expected
    assert result.max_mode is (expected["context"] == "500k")


@pytest.mark.parametrize(
    "model",
    [
        "grok-4.7-xhigh-fast[context=1m]",
        "grok-4.7-xhigh-fast[context=500k,fast=false]",
        "grok-4.7-xhigh-fast[reasoning_effort=low]",
    ],
)
def test_legacy_parameter_overrides_reject_invalid_or_conflicting_values(model):
    from cursor_gateway.model_parameters import parse_model_spec

    with pytest.raises(RequestError):
        parse_model_spec(model)
