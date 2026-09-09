"""Application configuration loading and validation.

Configuration lives in ``config.json`` at the project root. The file is
merged on top of built-in defaults (so it may stay partial), and every value
is validated so configuration errors surface early at startup instead of
failing later in the gesture pipeline.
"""

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed or out of range."""


def _coerce(field_type: type, value: Any, name: str) -> Any:
    if isinstance(value, bool):
        if field_type is bool:
            return value
        raise ConfigError(f"'{name}' must be a boolean")
    if field_type is bool:
        raise ConfigError(f"'{name}' must be a boolean")
    if field_type is int:
        if isinstance(value, int):
            return value
        raise ConfigError(f"'{name}' must be an integer, got {type(value).__name__}")
    if field_type is float:
        if isinstance(value, (int, float)):
            return float(value)
        raise ConfigError(f"'{name}' must be a number, got {type(value).__name__}")
    if field_type is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"'{name}' must be a string, got {type(value).__name__}")
    return value


def _section(section_type: type[T], data: Any, path: str) -> T:
    if not isinstance(data, Mapping):
        raise ConfigError(f"'{path}' must be a JSON object")
    known = {field.name: field for field in fields(section_type)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"unknown option(s) under '{path}': {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for field_name, field in known.items():
        if field_name not in data:
            continue
        value = data[field_name]
        child_path = f"{path}.{field_name}"
        if is_dataclass(field.type):
            kwargs[field_name] = _section(field.type, value, child_path)
        else:
            kwargs[field_name] = _coerce(field.type, value, child_path)
    return section_type(**kwargs)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ConfigError("camera.index must be >= 0")
        if self.width <= 0 or self.height <= 0:
            raise ConfigError("camera.width and camera.height must be positive")


@dataclass(frozen=True)
class StabilizerConfig:
    window_frames: int = 7
    required_frames: int = 5
    stable_ms: int = 120

    def __post_init__(self) -> None:
        if self.window_frames < 1:
            raise ConfigError("stabilizer.window_frames must be >= 1")
        if not 1 <= self.required_frames <= self.window_frames:
            raise ConfigError(
                "stabilizer.required_frames must be between 1 and "
                f"window_frames ({self.window_frames})"
            )
        if self.stable_ms < 0:
            raise ConfigError("stabilizer.stable_ms must be >= 0")


@dataclass(frozen=True)
class PinchConfig:
    threshold: float = 0.35
    drag_hold_ms: int = 400
    click_cooldown_ms: int = 350

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ConfigError("pinch.threshold must be in (0, 1)")
        if self.drag_hold_ms < 0:
            raise ConfigError("pinch.drag_hold_ms must be >= 0")
        if self.click_cooldown_ms < 0:
            raise ConfigError("pinch.click_cooldown_ms must be >= 0")


@dataclass(frozen=True)
class CursorConfig:
    smoothing_alpha: float = 0.25
    dead_zone: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ConfigError("cursor.smoothing_alpha must be in (0, 1]")
        if self.dead_zone < 0:
            raise ConfigError("cursor.dead_zone must be >= 0")


@dataclass(frozen=True)
class ScrollConfig:
    enabled: bool = True
    wheel_step: int = 1
    cooldown_ms: int = 300

    def __post_init__(self) -> None:
        if self.wheel_step < 1:
            raise ConfigError("scroll.wheel_step must be >= 1")
        if self.cooldown_ms < 0:
            raise ConfigError("scroll.cooldown_ms must be >= 0")


@dataclass(frozen=True)
class VolumeConfig:
    enabled: bool = True
    step: float = 0.04
    cooldown_ms: int = 300

    def __post_init__(self) -> None:
        if not 0.0 < self.step <= 1.0:
            raise ConfigError("volume.step must be in (0, 1]")
        if self.cooldown_ms < 0:
            raise ConfigError("volume.cooldown_ms must be >= 0")


@dataclass(frozen=True)
class HandConfig:
    model_path: str = "models/hand_landmarker.task"
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ConfigError("hand.model_path must be a non-empty string")
        if self.num_hands < 1:
            raise ConfigError("hand.num_hands must be >= 1")
        for name in (
            "min_hand_detection_confidence",
            "min_hand_presence_confidence",
            "min_tracking_confidence",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"hand.{name} must be in [0, 1]")


@dataclass(frozen=True)
class HotkeysConfig:
    toggle: str = "space"
    quit: str = "esc"

    def __post_init__(self) -> None:
        if not self.toggle.strip():
            raise ConfigError("hotkeys.toggle must be a non-empty string")
        if not self.quit.strip():
            raise ConfigError("hotkeys.quit must be a non-empty string")


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    hand: HandConfig
    stabilizer: StabilizerConfig
    pinch: PinchConfig
    cursor: CursorConfig
    scroll: ScrollConfig
    volume: VolumeConfig
    hotkeys: HotkeysConfig

    @classmethod
    def defaults(cls) -> "AppConfig":
        return cls(
            camera=CameraConfig(),
            hand=HandConfig(),
            stabilizer=StabilizerConfig(),
            pinch=PinchConfig(),
            cursor=CursorConfig(),
            scroll=ScrollConfig(),
            volume=VolumeConfig(),
            hotkeys=HotkeysConfig(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        if not isinstance(data, Mapping):
            raise ConfigError("root configuration must be a JSON object")
        merged = _deep_merge(asdict(cls.defaults()), data)
        return _section(cls, merged, "config")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        try:
            raw = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config file '{config_path}': {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in '{config_path}': {exc}") from exc
        return cls.from_dict(data)
