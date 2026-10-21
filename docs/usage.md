# Usage guide

## Package layout

| Repository path | Purpose |
| --- | --- |
| `src/phy_mac_isac/modules/` | Shared environment, parameter dataclasses, allocation, transmission, theory, and tracking models |
| `src/phy_mac_isac/utils/` | Geometry, plotting, statistics, argument routing, execution, and system helpers |
| `src/phy_mac_isac/semantic_isac/` | Semantic ISAC configuration, scenario, experiments, and result analysis |
| `tests/` | Argument-routing regression tests |
| `pyproject.toml` | Build backend, metadata, dependency requirements, and package discovery |

The import package is `phy_mac_isac`, while the distribution installed by pip is named `phy-mac-isac`. The current repository contains no scene asset directory or radio-map archives. Generated assets are stored in the runtime cache.

## Installation and launching modules

Use Python 3.11 or newer. From the repository root:

```bash
python -m pip install -e .
```

An editable installation makes source changes available to the selected interpreter. Use `python -m pip install .` for a regular installation.

Launch experiments with `python -m phy_mac_isac.semantic_isac.<module>`. Running a file such as `scenario.py` directly does not give its relative imports a package context.

For PyCharm Run/Debug, select the virtual environment where the package is installed and configure the target as a module, for example `phy_mac_isac.semantic_isac.simulate_vs_ue`. Put arguments such as `--cpu --ra_method MAF --savesubdir trial` in the script parameters, and set directory overrides in the run configuration's environment variables.

## Experiment entry points

(Copied from the [semantic_isac submodule](src/phy_mac_isac/semantic_isac/README.md))

| Module under `phy_mac_isac.semantic_isac` | Current experiment |
| --- | --- |
| `simulate_vs_ue` | UE counts 6, 12, 18, 27, and 36; `NUM_EP` episodes per value and method |
| `simulate_vs_regions` | Region counts 2, 4, 8, 16, 32, and 64; `NUM_EP` episodes per value and method |
| `simulate_qualitative` | UE counts 2 through 5; one episode per value, one slot, 60 subcarriers, three explicit regions, and debug figures |
| `print_results` | Quantitative sweep statistics and empirical CDF plots; requires `--data` |
| `run_vram_test` | A spawned, single-frame CUDA probe that reports process and PyTorch memory peaks |

The three simulation scripts share these options:

| Option | Behavior |
| --- | --- |
| `--ra_method LABEL` | Run one allocation method; omitting it runs all available methods |
| `--cpu` | Hide CUDA devices before the sweep's Sionna imports and select CPU execution |
| `--parallel` | Use a pool of spawned worker processes; execution is sequential by default |
| `--savesubdir NAME` | Append a subdirectory below the dated results directory |
| `--quiet` | Disable the episode frame progress bar |
| `--overwrite` | Recompute episode results rather than reuse existing summary archives |
| `--concurrent N` | Budget CPU/GPU resources for N scripts sharing the machine; default 1 |

Use a relative name for `--savesubdir`. The flag does not select the root output directory; use `PHY_MAC_ISAC_DATA_DIR` for that.

Available allocation labels are:

| Label | Class |
| --- | --- |
| `MAF` | `RoundRobinAllocation` |
| `SRI` | `SemanticRiskAllocation` |
| `SRL` | `SemanticRiskAllocationNoComm` |
| `PBI` | `PebBasedAllocation` |
| `PBL` | `PebBasedAllocationNoComm` |
| `QBC` | `QueueBasedAllocation` |

For example:

```bash
python -m phy_mac_isac.semantic_isac.simulate_vs_regions \
  --cpu --parallel --ra_method SRI --savesubdir trial
```

## Configuration

Shared defaults and dataclasses are defined in [modules/params.py](../src/phy_mac_isac/modules/params.py). Semantic ISAC inherits those dataclasses and overrides selected defaults in [semantic_isac/parameters.py](../src/phy_mac_isac/semantic_isac/parameters.py).

Current semantic defaults include 3 APs, 2 antennas per AP, 12 UEs, 16 regions, 5 slots per frame, 108 subcarriers, and 100 frames per episode. `NUM_EP` is 100. Each script can override these settings; the qualitative experiment has its own geometry and radio configuration.

Sweep arrays and episode counts are Python variables near the beginning of each simulation script. They are not CLI options. Ray-tracing settings are configured through the `rays` field and its `RT_PARAM` defaults.

For programmatic runs, `semantic_isac.scenario.run_episode()` accepts:

- `env_kwargs`: constructor parameters for the scenario's shared/wireless dataclasses, base environment options, and scenario setup arguments.
- `ra_kwargs`: additional constructor arguments for the selected allocation method.
- `step_kwargs`: per-frame options such as `cto`, `cfo`, and `sync`.

Unknown environment/step argument names raise `TypeError` before environment construction. Derived dataclass fields declared with `init=False` cannot be supplied as constructor options. Use `num_regions`, for example, rather than `sweep_value`.

In shared utilities, `filter_call_kwargs()` selects arguments accepted by a callable or constructor. `filter_kwargs()` retains all dataclass fields for restoration, including derived fields.

## Directories and environment variables

The following defaults apply when the environment variables are unset or empty:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PHY_MAC_ISAC_DATA_DIR` | `~/.local/share/phy_mac_isac/data` | Results root |
| `PHY_MAC_ISAC_CACHE_DIR` | `~/.cache/phy_mac_isac` | Generated assets and lock files |
| `PHY_MAC_ISAC_SCENE_DIR` | `<installed phy_mac_isac package>/scenes` | Optional input scene assets |
| `PHY_MAC_ISAC_RM_DIR` | `<CACHE_FOLDER>/semantic_isac/rm` | Semantic ISAC radio maps |

`XDG_DATA_HOME` replaces `~/.local/share` in the default data root. `XDG_CACHE_HOME` replaces `~/.cache` in the default cache root. Package-specific overrides take precedence. Generated floor meshes use `<CACHE_FOLDER>/meshes`; the base environment's radio maps use `<CACHE_FOLDER>/rm`.

Directory values expand environment variables and `~`, then resolve to absolute paths. Relative overrides are resolved against the process working directory. Set overrides before importing the package; restart a Python console after changing them.

For Bash:

```bash
export PHY_MAC_ISAC_DATA_DIR="$HOME/OneDrive/data/semantic_isac"
export PHY_MAC_ISAC_CACHE_DIR="$HOME/.cache/phy_mac_isac"
```

For a PyCharm environment variable:

```text
PHY_MAC_ISAC_DATA_DIR=$HOME/OneDrive/data/semantic_isac/
```

Changing the scene input directory does not relocate generated meshes or radio maps. A default generated-floor experiment does not require a populated scene directory. Named scenes are selected programmatically with `env_kwargs["scene_name"]`; the loader first searches the configured scene directory, then tries a built-in `sionna.rt.scene` name.

## Results and analysis

The scripts write beneath `<DATA_FOLDER>/<YYYY-MM-DD>/<savesubdir>/`; the final component is omitted when `--savesubdir` is empty.

| Output | Filename or subdirectory |
| --- | --- |
| UE sweep summary | `<LABEL>_vs_ue.csv` |
| UE sweep episodes | `<LABEL>_vs_ue_U<COUNT:03d>/<LABEL>_ep<EPISODE:04d>.npz` |
| Region sweep summary | `<LABEL>_vs_regions.csv` |
| Region sweep episodes | `<LABEL>_vs_regions_R<COUNT:03d>/<LABEL>_ep<EPISODE:04d>.npz` |
| Qualitative summary | `<LABEL>_qualitative.csv` |
| Qualitative episodes and debug PNGs | `<LABEL>_qualitative_U<COUNT:03d>/` |

CSV summaries include the sweep value and averaged metrics. Episode archives store flattened parameter, geometry, estimation, and metric fields. Time-indexed arrays use the final row for initialization; summary calculations omit that row.

Existing readable episode summaries are reused unless `--overwrite` is supplied. Use a separate results subdirectory when changing an experiment configuration. `--overwrite` does not clear the generated-asset cache.

To analyze results, pass the parent experiment directory containing the quantitative sweep episode subdirectories:

```bash
python -m phy_mac_isac.semantic_isac.print_results \
  --data "$HOME/.local/share/phy_mac_isac/data/2026-10-21/trial"
```

Replace the path with your actual run directory. The script processes both quantitative sweep families using the configured method labels, sweep values, and `NUM_EP`; it is not the qualitative experiment viewer. Keep those settings aligned with the saved run. It writes metric statistics as CSV files and CDF plots as JPEG files into the supplied directory. Its plotting configuration enables LaTeX rendering through `latex_look()`.

Generated meshes, radio maps, episode archives, and sweep CSVs use temporary files followed by replacement of the completed destination. Mesh/map locks live in `<CACHE_FOLDER>/locks`, so completed data retain their `.ply` and `.npz` extensions.

## CPU, CUDA, and parallel execution

The sweep scripts select logical `cuda:0` by default. To use another physical GPU, set visibility before starting Python:

```bash
CUDA_VISIBLE_DEVICES=1 python -m phy_mac_isac.semantic_isac.simulate_vs_ue \
  --parallel --ra_method MAF --savesubdir gpu_trial
```

The selected physical GPU becomes logical `cuda:0` inside the process. The sweep CLI currently selects one logical device; it does not distribute workers across multiple GPUs.

`--cpu` is handled before the sweep scripts import Sionna. Programmatic callers should likewise set GPU visibility before importing GPU-dependent modules and pass a consistent device to the episode runner and worker executor.

`run_simulations()` executes jobs sequentially or through a process pool using the `spawn` start method. Resource limits determine `max_workers`; a per-episode GPU semaphore is not used by this runner. `--concurrent` adjusts each script's resource estimate and does not coordinate a global scheduler.

## VRAM profiling

Run the supplied probe with:

```bash
python -m phy_mac_isac.semantic_isac.run_vram_test
```

Its current configuration runs one frame for 12 UEs on logical `cuda:0`, with environment replay and episode saving disabled. Scene setup can still generate mesh and radio-map cache files.

`profile_episode_vram()` measures the spawned process with NVML and reports process memory peaks, PyTorch allocation/reservation peaks, and `suggested_episode_mib`. The default safety factor is 1.2.

The sweep scripts currently use the estimator defaults in `get_processing_resources()`. To use a measured budget in a custom caller, pass it as `measured_episode_mib` and keep the probe configuration representative of the actual workload. The helper does not automatically update sweep settings.

## Development

See [CONTRIBUTING.md](../CONTRIBUTING.md) for tests, package builds, and distribution checks. Packaging checks and the isolated argument-routing tests do not run a full Sionna/CUDA simulation or validate scientific results.
