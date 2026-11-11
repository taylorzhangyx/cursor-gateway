"""Parse Cursor App bracket IDs without silently dropping model parameters."""

import re
from dataclasses import dataclass

from .translate_error import RequestError

_SPEC = re.compile(
    r"([^\s\[\]]+)\["
    r"([A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.:-]+"
    r"(?:,[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.:-]+)*)\]"
)
_GROK47_VALUES = {
    "context": {"256k", "500k"},
    "reasoning_effort": {"low", "medium", "high", "xhigh"},
    "fast": {"false", "true"},
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    parameters: tuple[tuple[str, str], ...] = ()
    max_mode: bool | None = None

    @property
    def legacy_model_id(self) -> str:
        values = dict(self.parameters)
        if (
            self.model_id.removeprefix("cursor-") == "grok-4.7"
            and "reasoning_effort" in values
            and "fast" in values
        ):
            suffix = "-fast" if values["fast"] == "true" else ""
            return f"{self.model_id}-{values['reasoning_effort']}{suffix}"
        return self.model_id


def parse_model_spec(model: str) -> ModelSpec:
    if "[" not in model and "]" not in model:
        return ModelSpec(model)
    match = _SPEC.fullmatch(model)
    if match is None:
        raise RequestError("invalid parameterized model syntax", param="model")
    base, raw = match.groups()
    values: dict[str, str] = {}
    for pair in raw.split(","):
        key, value = pair.split("=", 1)
        if key in values:
            raise RequestError("duplicate model parameter: " + key, param="model")
        values[key] = value
    max_mode = None
    if base.removeprefix("cursor-") == "grok-4.7":
        for key, value in values.items():
            if key not in _GROK47_VALUES or value not in _GROK47_VALUES[key]:
                raise RequestError("unsupported Grok 4.7 parameter: " + key, param="model")
        if "context" in values:
            max_mode = values["context"] == "500k"
    return ModelSpec(base, tuple(values.items()), max_mode)
