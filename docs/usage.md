# Usage and package layout

## Layout

| Previous path on `develop` | Import or module in this package |
| --- | --- |
| `src/modules/*.py` | `phy_mac_isac.modules.*` |
| `src/utils/*.py` | `phy_mac_isac.utils.*` |
| `src/semantic_isac/*.py` | `phy_mac_isac.semantic_isac.*` |
| `src/semantic_isac/scenes/` | `phy_mac_isac/semantic_isac/scenes/` package data |

The experiment entry points are Python modules. For example, `python -m phy_mac_isac.semantic_isac.simulate_vs_ue --help` and `python -m phy_mac_isac.semantic_isac.simulate_vs_regions --help`. The analysis scripts can be launched with `python -m phy_mac_isac.semantic_isac.check_results` and `python -m phy_mac_isac.semantic_isac.print_results`; review their configured data paths first.

## Configuration and data

The scripts currently take defaults from `phy_mac_isac.semantic_isac.parameters` and the shared modules take defaults from `phy_mac_isac.modules.params`. They are separate research configurations, retained for later consolidation. The default output directory is `~/OneDrive/data`. `PHY_MAC_ISAC_SCENE_DIR` overrides the scene directory for both parameter modules. The default scene directory contains bundled meshes and cached radio maps. Installed package directories may be read-only, so copy these assets into a writable directory and set the environment variable when a simulation needs to create or update scene files.

These scripts are computationally intensive and have external GPU, rendering, and spatial-index dependencies. Installing the distribution and compiling its modules verifies packaging; it does not establish scientific correctness or reproducibility of a full run.

## Build

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src/phy_mac_isac
python -m build
python -m twine check dist/*
```

The wheel includes `.ply` meshes and `.npz` radio maps. Inspect dependency and version compatibility on the target machine before publishing a release.
