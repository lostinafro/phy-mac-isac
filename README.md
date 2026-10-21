# phy-mac-isac

Research software for physical and MAC layer integrated sensing and communications (ISAC) simulations using Sionna.
This software is made under the under the Marie Skłodowska-Curie Action Postdoctoral Fellowship 
project [SCOPES](https://cordis.europa.eu/project/id/101204088) with grant agreement No. 101204088. 

The distribution name is `phy-mac-isac`; the Python import name is `phy_mac_isac`. Reusable models live in `phy_mac_isac.modules`, shared helpers in `phy_mac_isac.utils`, and semantic ISAC experiments in `phy_mac_isac.semantic_isac`.
Each subdirectory within ``src`` represents a scientific project, and it contains a README file with specifications on the experiment, 
and on the submitted and published scientific papers, if any.

## Install

Use **Python 3.11 or newer**. From a checkout of the `main` branch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Runtime requirements and their declared minimum versions are listed in [pyproject.toml](pyproject.toml). CUDA execution needs a compatible NVIDIA driver and a CUDA-enabled PyTorch installation. CPU runs still use the package's Sionna, Mitsuba, and spatial-index dependencies.

## Example of experiment run

Run experiment files as **Python modules** so package-relative imports resolve:

```bash
python -m phy_mac_isac.semantic_isac.simulate_vs_ue --help
python -m phy_mac_isac.semantic_isac.simulate_vs_regions --help
python -m phy_mac_isac.semantic_isac.simulate_qualitative --help
```

For a sequential CPU run using the round-robin method:

```bash
python -m phy_mac_isac.semantic_isac.simulate_vs_ue \
  --cpu --ra_method MAF --savesubdir trial
```

The sweep scripts use logical `cuda:0` unless `--cpu` is supplied. Add `--parallel` to run episodes in spawned worker processes. Omitting `--ra_method` runs all allocation methods.

The two quantitative sweeps currently use **100 episodes per sweep value and method**, with 100 frames per episode. The qualitative script uses one episode per sweep value and enables debug figures. Review [the experiment configuration](src/phy_mac_isac/semantic_isac/parameters.py) and each script's sweep settings before launching a full experiment.

## Results and generated assets

By default, results are written outside the checkout:

```text
~/.local/share/phy_mac_isac/data/<YYYY-MM-DD>/<savesubdir>/
```

Set `PHY_MAC_ISAC_DATA_DIR` before starting Python to choose another root:

```bash
export PHY_MAC_ISAC_DATA_DIR="$HOME/OneDrive/data/semantic_isac"
```

Generated floor meshes and semantic ISAC radio maps default to `~/.cache/phy_mac_isac/meshes` and `~/.cache/phy_mac_isac/semantic_isac/rm`. XDG directory overrides and package-specific overrides are described in [the usage guide](docs/usage.md#directories-and-environment-variables). These generated files are created at runtime; the current repository contains no bundled scene meshes or radio-map archives.

To summarize quantitative sweep results and save plots, pass the experiment directory:

```bash
python -m phy_mac_isac.semantic_isac.print_results \
  --data "$PHY_MAC_ISAC_DATA_DIR/2026-10-21/trial"
```

Replace the date and subdirectory with those of your run. Plot rendering currently uses LaTeX, so a working LaTeX installation is required unless `latex_look()` is changed.

See [the usage guide](docs/usage.md) for CLI options, device selection, configuration, output layout, and VRAM profiling. Development checks are documented in [CONTRIBUTING.md](CONTRIBUTING.md). The project is available under the [MIT License](LICENSE).
