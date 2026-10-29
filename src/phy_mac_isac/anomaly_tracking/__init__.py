"""Geometry-based SISO PHY for externally scheduled anomaly tracking."""
from .parameters import SharedParams, WirelessParams, Metrics

__all__ = [
    "SharedParams",
    "WirelessParams",
    "Metrics",
    "AnomalyTrackingScenario",
    "generate_environment"
]


def __getattr__(name):
    if name in {"AnomalyTrackingScenario", "generate_ep_environment"}:
        from importlib import import_module

        module = import_module(".scenario", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )