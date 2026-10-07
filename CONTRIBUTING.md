# Contributing

Create a branch from `develop` for changes and describe the simulation configuration and results in a pull request. Use Python 3.10 or newer and install with `python -m pip install -e ".[dev]"`.

Keep internal imports relative to `phy_mac_isac`. Launch experiments with `python -m phy_mac_isac.semantic_isac.<module>` and avoid writing generated output into the installed package. Before submitting, run `python -m compileall -q src/phy_mac_isac`, `python -m build`, and `python -m twine check dist/*`. Include a focused verification of scientific behavior when changing an algorithm.
