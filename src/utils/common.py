# Generic reflection helpers
import dataclasses
import inspect


def filter_kwargs(obj, **kwargs) -> dict:
    """Filter kwargs down to only those a callable, class, or dataclass (instance or
    class) actually accepts -- for collecting parameters from a larger kwargs blob into
    just what a specific target needs.

    :param obj: a function/method, a plain class, or a dataclass (class or instance).
        For a dataclass, all fields are considered (e.g. `field(init=False)` are included)
    :returns: dict, the subset of kwargs whose keys `method` supports.
    """
    if dataclasses.is_dataclass(obj):
        supported_params = {f.name for f in dataclasses.fields(obj)}
    else:
        sig = inspect.signature(obj)
        supported_params = sig.parameters.keys()

    return {k: v for k, v in kwargs.items() if k in supported_params}