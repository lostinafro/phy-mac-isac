# Anomaly-tracking PHY
Code implementing the PHY layer part of the performance evaluation of the paper "Push-Pull Medium Access for Digital Twin Alignment and Low-Latency Anomaly Reporting"

## Abstract
> A digital twin (DT) contains a set of virtual models of real systems and processes that are synchronized with their physical counterparts. In a setup in which contact with the physical world is maintained through sensors and actuators that are wirelessly connected to the DT’s computing engine, DT alignment requires periodic status updates, while safety-critical messages and fault conditions call for low-latency anomaly reporting, creating a fundamental trade-off in how wireless resources are used. We present a medium access framework combining pull-based updates, centrally scheduled according to goal-oriented principles, with urgent push-based updates, for which transmission decisions are made directly by the sensors. This enables the system to quickly detect and recover from anomalies while maintaining DT alignment. We thus design a push-pull scheduler (PPS) that strikes a balance in the trade-off between DT alignment in normal conditions and anomaly reporting, optimizing resource usage and reducing DT drift by 20 − 30% with respect to state-of-the-art solutions while maintaining the same anomaly detection guarantees, or reducing worst-case anomaly detection times by 30 − 70% while meeting the same DT alignment conditions.

The paper is in preparation to be submitted to IEEE Transactions on Communications. A preprint version is available on [arXiv](https://arxiv.org/pdf/2508.21516).

## Code documentation

`AnomalyTrackingScenario` follows `SimulationEnvironment`: construct it with
shared/wireless parameter dictionaries, call `setup_scenario()`, wire modules
with `setup_modules()`, and then call `step(allocation)` once per frame.

```python
import numpy as np
from phy_mac_isac.anomaly_tracking.scenario import AnomalyTrackingScenario

scenario = AnomalyTrackingScenario(
    episode=42,
    shared_params={"num_ue": 100, "num_slots": 4, "num_clusters": 10, "cluster_size":5, "num_frames": 100},
    wireless_params={"ce_error_variance": 0.0},
    load_env=False,
)
scenario.setup_scenario()  # Random UE positions in the base environment polygon
scenario.setup_modules(device="cpu")

# One ordered vector per frame, supplied by the external MAC scheduler.
# [0, num_ue) represents UE IDs, -1 is a silent slot, -2 is a collision
decoded, successful_ues, bler = scenario.step(np.array([3, 7, -1, 12]))
print(env.frame)                       # 0; initial geometry is at frame -1
print(successful_ues)                  # [3, 7, 12]
print(decoded)                         # Four flags; idle slot is False
print(env.metrics.mcs_index[env.frame])      # One MCS per occupied slot
print(env.metrics.harq_feedback[env.frame])  # One TB ACK/NACK per occupied slot
```

Input configuration uses `SharedParams`, `WirelessParams`, and `Metrics` from this subproject, which override the shared classes in `modules/params.py`. `GeometryParams` remains the common geometry class. `Metrics` is an output container in `anomaly_tracking/parameters.py`; no separate scenario-input configuration dataclass is used. Its import from `scenario` also remains valid.

### Geometry and channel

- This scenario requires one AP with one antenna and enforces `load_scene=False`.
  It uses base area generation, random UE placement, AP placement (default on the origin), geometric initialization and optional trajectory restoration. No RT scene or radio map is generated. The common package still imports Sionna RT dependencies.
- UE speed and velocity covariance default to exactly zero. The initial geometry  is stored at frame -1. Each step increments the frame, calls base `move_ue()`, and samples the channel from **`self.geo.ue_location[self.frame, ue_ids]`**. Nonzero `ue_vel_mean`/`ue_vel_cov` and restored trajectories use the same flow.
- The `ue_ids` vector is extracted from the `allocation` `step()` argument, which requires a vector of size `num_slots` containing indices of the UEs `[0, num_ue)` allocated, `-1` for silent slots, and `-2` for collided slots (collision must be checked beforehand). A single UE can be allocated once per frame.
- Sionna PHY's TR 38.901 **UMi** model is used for channel. Its topology contains the actual UE position and velocity and the AP position. Outdoor links use distance-dependent LOS probability unless `force_los` is set to True/False. Path loss is enabled and OFDM channel normalization is  disabled, so geometry-dependent gain is retained. Shadow fading defaults to
  False and can be enabled with `enable_shadow_fading`.
- Each occupied slot is simulated as a separate batch example with one SISO UE/AP pair, by assuming **orthogonal UE scheduling**. Changing the number of occupied slots resets topology buffers as necessary. Idle slots are omitted from the channel batch but retained in slot-result mapping. This is not a persistent spatially consistent channel process and does not model TDD reciprocity or switching guards.
- Both antennas are single vertically polarized omnidirectional PHY elements.
- Default AP placement follows the shared perimeter placement, with its height  set to 10 m for UMi. UE height keeps this subproject's `UE_HEIGHT` value (1.5 m). You can supply explicit `init_ue_pos` and `ap_pos` or adjust `UE_HEIGHT` for generated positions.
- The resource grid derives T, F, subcarrier spacing and CP from shared wireless
  parameters. A single-transmitter grid reserves pilot symbols [2,T-3] (or [0]
  for short slots).

### Estimation error and PHY abstraction

`wireless.ce_error_variance` defaults to **0.0**. It is an
absolute complex channel-coefficient MSE **after path loss**.

```text
h_hat = h + e
E[abs(e)**2] = ce_error_variance
Var(real(e)) = Var(imag(e)) = ce_error_variance / 2
```

At zero variance the estimate equals the channel exactly and no estimation-error random samples are drawn. To change it later:

```python
scenario.wireless.ce_error_variance = 1e-10  # Example absolute MSE
```

Choose the variance relative to the attenuated channel power in your scenario.
The noise is independent per CFR coefficient.

`SisoPhyAbstraction` uses `InnerLoopLinkAdaptation` to choose MCS from the
estimated channel and `PHYAbstraction` to sample transport-block ACK/NACK from
the physical channel. Nonzero error uses an additional uncorrelated-noise
approximation in both the prediction and reception SINRs:

```text
prediction SINR = P * abs(h_hat)**2 / N0B 
reception SINR  = P * abs(h)**2     / N0B
```

At zero variance it reduces to perfect-CSI SISO reception. 

Total UE transmit power and AP thermal noise come from `WirelessParams`. Optional Downlink direction uses AP transmit power and UE noise and sets the SYS MCS category accordingly (not used in the paper simulations). The shared parameter class currently computes both receiver noise powers using a 9 dB noise figure (check and edit `anomaly_tracking.parameters` to change the default values).

### External scheduler and episode helper

`step()` accepts zero-based UE IDs, -1 for idle slots, and -2 for collided slots. The input array must have size `num_slots`. Resolve eventual random access contention before calling this PHY; budget pairs and contention thresholds are not allocation vectors.
The scenario does not run a MAC scheduler or update its beliefs/queues.

```python
allocation = pull_scheduler.schedule_pps(pull_resources)
decoded, _, _ = scenario.step(allocation)
observations = np.full(len(allocation), -1, dtype=int)
observations[decoded] = node_states[allocation[decoded]]
```

Check the application payload against the SYS-reported transport capacity
before treating a full packet as delivered. Retransmissions and segmentation
remain with the caller; there is no automatic HARQ loop.


### Helping function
For a simple generation of the default environment, the following snippet is enough.

```python
from phy_mac_isac.anomaly_tracking import generate_environment

num_ue = 50
num_slots = 10
num_clusters = 10
cluster_size = 4
env_kwargs = {"num_ue": num_ue, "num_slots": num_slots, "num_frames": 10,
              "num_clusters": num_clusters, "cluster_size": cluster_size,
              "perfect_csi": False}
env = generate_environment(1, env_kwargs=env_kwargs, device="cpu")

print(env.c)                # SharedParams
print(env.wireless)         # WirelessParams
print(env.geo)              # GeometryParams
print(env.metrics)          # SharedParams
```

It returns the generated environment which contains `SharedParams`, `WirelessParams`, `GeometryParams` `Metrics` to assess simulations parameters and results.
