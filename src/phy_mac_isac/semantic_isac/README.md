# Semantic ISAC project
Code implementing the performance evaluation of the paper "User Scheduling for Integrated Communication and Semantic Localization in D-MIMO"

## Abstract
> We focus on user scheduling for integrated communication and semantic localization in distributed multiple-input multiple-output systems with limited uplink resources. The scheduling metric is based on semantic misclassification risk, linking localization uncertainty to the geometry of the spatial decision. Communication requirements are enforced through probabilistic queue constraints, while remaining resources are allocated according to the predicted reduction in semantic risk. This shifts localization-aware scheduling from accuracy-based allocation toward task-relevant information acquisition. Numerical results show that semantic risk-based scheduling consistently outperforms scheduling based on traditional metrics.

The paper is submitted to IEEE ICASSP 2027, and its preprint is available [here](https://zenodo.org/records/23338277)


## Experiment entry points

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