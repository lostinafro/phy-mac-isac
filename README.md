# phy-mac-isac

Research code for physical and MAC layer integrated sensing and communications (ISAC) simulations using Sionna.

The distribution name is `phy-mac-isac`; the Python import name is `phy_mac_isac`. The package contains reusable simulation components in `phy_mac_isac.modules`, shared helpers in `phy_mac_isac.utils`, and experiment scripts in `phy_mac_isac.semantic_isac`.

## Install

Use Python 3.10 or newer. A working Sionna and Mitsuba setup, and system libraries required by Rtree, may be needed on your platform. From the repository root:

```bash
python -m pip install -e .
```

To install only the package build tools as well:

```bash
python -m pip install -e ".[dev]"
```

## Run the experiments

The experiment modules retain their existing command-line arguments. Run them as modules so their package imports resolve:

```bash
python -m phy_mac_isac.semantic_isac.simulate_vs_ue --help
python -m phy_mac_isac.semantic_isac.simulate_vs_regions --help
python -m phy_mac_isac.semantic_isac.simulate_qualitative --help
```

For example:

```bash
python -m phy_mac_isac.semantic_isac.simulate_vs_ue --cpu --savesubdir trial
```

The defaults in `semantic_isac.parameters` are research-scale (including 100 episodes); review them before a full run. Results default to `~/OneDrive/data`. Set `PHY_MAC_ISAC_SCENE_DIR` to a writable scene directory if a run generates meshes or radio maps; the bundled scenes are provided as starting data.

See [usage and package layout](docs/usage.md) for the migration map and build commands. The source is offered under the [MIT License](LICENSE).
