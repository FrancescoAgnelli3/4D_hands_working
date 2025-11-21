"""SFgDNetTry: identical architecture to SFgDNet but treated as a separate model option."""

from __future__ import annotations

from .SFgDNet import SFgDNetConfig, SFgDNetForecaster


class SFgDNetTryConfig(SFgDNetConfig):
    """Configuration wrapper for the SFgDNetTry variant."""


class SFgDNetTryForecaster(SFgDNetForecaster):
    """Forecaster wrapper for the SFgDNetTry variant."""

    def __init__(self, cfg: "SFgDNetTryConfig") -> None:
        super().__init__(cfg)


__all__ = ["SFgDNetTryConfig", "SFgDNetTryForecaster"]
