# Contributing

Create a branch from `develop` and describe the resulting behavior, simulation configuration, and validation in a pull request.

## Setup

Use Python 3.11 or newer. From the repository root, with your virtual environment active:

```bash
python -m pip install -e ".[dev]"
```

The `dev` extra installs the build and distribution-check tools alongside the runtime dependencies. Declared dependency requirements are maintained in [pyproject.toml](pyproject.toml).

## Source and configuration

Keep internal imports relative to `phy_mac_isac`. Launch experiments with `python -m phy_mac_isac.semantic_isac.<module>`.

Shared parameter dataclasses live in `modules/params.py`; semantic ISAC subclasses and experiment defaults live in `semantic_isac/parameters.py`. Keep derived values out of constructor argument dictionaries. Use `filter_call_kwargs()` for callable arguments, `validate_kwargs()` to check shared dictionaries before routing them, and `filter_kwargs()` when restoring dataclass fields.

Keep generated results and caches outside the installed package. The output-directory variables, XDG defaults, and script options are documented in [the usage guide](docs/usage.md). If a change affects those interfaces, update the guide and README with it.

## Checks

Run the following from the repository root:

```bash
python -m compileall -q src/phy_mac_isac
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

The argument-routing regression tests use the Python standard library and lightweight stand-ins for the environment. They run without importing Sionna or initializing CUDA.

The packaging workflow runs these checks on Python 3.11 and checks that the wheel contains the import package and semantic scenario module. Generated meshes and radio maps are runtime cache files and are not required wheel contents.

For changes to scientific algorithms, include a focused numerical check and record the configuration, device, and relevant dependency versions. State whether full simulations were run; syntax, routing, and packaging checks alone do not establish scientific correctness.
