# Generic reflection helpers
import dataclasses
import inspect


def keyword_parameters(obj) -> set[str]:
    """Return the explicitly declared parameters that accept keyword arguments.

    :param obj: Callable, including a class constructor or a bound method.
    :returns: set[str], names of positional-or-keyword and keyword-only parameters.
        Positional-only parameters and variadic parameter names are excluded.
    """
    return {
        name
        for name, parameter in inspect.signature(obj).parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def filter_call_kwargs(obj, **kwargs) -> dict:
    """Filter keyword arguments for a callable or class constructor.

    :param obj: Callable whose explicit keyword parameters select the arguments.
        A dataclass constructor excludes fields declared with ``init=False``.
    :param kwargs: Arguments to filter; unsupported names are omitted.
    :returns: dict, arguments explicitly accepted by the callable.
    """
    supported = keyword_parameters(obj)
    return {key: value for key, value in kwargs.items() if key in supported}


def validate_kwargs(kwargs: dict, *targets, exclude=(), name: str = "kwargs") -> None:
    """Reject arguments unsupported by every target in a shared argument dictionary.

    :param kwargs: Arguments to validate before routing them to multiple callables.
    :param targets: Callables whose explicit keyword parameters form the allowed set.
        A variadic ``**kwargs`` does not allow arbitrary names in this validation.
    :param exclude: Names supplied internally and unavailable to the caller.
    :param name: Argument dictionary name included in the error message.
    :raises TypeError: If an argument is unknown or reserved for internal use.
    """
    supported = set().union(*(keyword_parameters(target) for target in targets))
    unsupported = set(kwargs) - (supported - set(exclude))
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(f"Unsupported {name} argument(s): {names}")


def filter_kwargs(obj, **kwargs) -> dict:
    """Filter kwargs down to only those a callable, class, or dataclass (instance or
    class) actually accepts -- for collecting parameters from a larger kwargs blob into
    just what a specific target needs.

    :param obj: a function/method, a plain class, or a dataclass (class or instance).
        For a dataclass, all fields are considered (e.g. `field(init=False)` are included).
        Use ``filter_call_kwargs`` instead when passing arguments to a constructor.
    :returns: dict, the subset of kwargs whose keys `method` supports.
    """
    if dataclasses.is_dataclass(obj):
        supported_params = {f.name for f in dataclasses.fields(obj)}
    else:
        sig = inspect.signature(obj)
        supported_params = sig.parameters.keys()

    return {k: v for k, v in kwargs.items() if k in supported_params}
