# Anomaly-tracking PHY

`AnomalyTrackingScenario` follows `SimulationEnvironment`: construct it with
shared/wireless parameter dictionaries, call `setup_scenario()`, wire modules
with `setup_modules()`, and then call `step(allocation)` once per frame.

```python
import numpy as np
from phy_mac_isac.anomaly_tracking import AnomalyTrackingScenario

scenario = AnomalyTrackingScenario(
    episode=42,
    shared_params={"num_ue": 100, "num_slots": 20, "num_frames": 100},
    wireless_params={"channel_estimation_error_variance": 0.0},
    load_env=False,
)
scenario.setup_scenario()  # Random UE positions in the base environment polygon
scenario.setup_modules(device="cpu")

# One ordered vector per frame, supplied by the external MAC scheduler.
result = scenario.step(np.array([3, 7, -1, 12]))
assert result.channel.shape == (3, 1, 1, 1, 1, 14, 72)
print(result.frame)                       # 0; initial geometry is at frame -1
print(result.slot_indices)                # [0, 1, 3]
print(result.ue_ids)                      # [3, 7, 12]
print(result.decoded)                     # Four flags; idle slot is False
print(result.transmission.mcs_index)      # One MCS per occupied slot
print(result.transmission.harq_feedback)  # One TB ACK/NACK per occupied slot
```

Input configuration uses `SharedParams` and `WirelessParams` from this
subproject, which override the shared classes in `modules/params.py`.
`GeometryParams` remains the common geometry class. `StepResult` is an output
container in `anomaly_tracking/parameters.py`; no separate scenario-input
configuration dataclass is used. Its import from `scenario` also remains valid.
The package exports `AnomalyTrackingScenario`, `SharedParams`, `WirelessParams`,
`StepResult`, and the allocation-driven `run_episode` helper.

## Geometry and channel

- This scenario requires one AP with one antenna and enforces `load_scene=False`.
  It uses base area generation, random UE placement, AP placement, geometric
  initialization and optional trajectory restoration. No RT scene or radio map
  is generated. The common package still imports Sionna RT dependencies.
- UE speed and velocity covariance default to exactly zero. The initial geometry
  is stored at frame -1. Each step increments the frame, calls base `move_ue()`,
  and samples the channel from **`self.geo.ue_location[self.frame, ue_ids]`**.
  Nonzero `ue_vel_mean`/`ue_vel_cov` and restored trajectories use the same flow.
- Sionna PHY's TR 38.901 **UMi** model replaces the geometry-free TDL prototype.
  Its topology contains the actual UE position and velocity and the AP position.
  Outdoor links use distance-dependent LOS probability unless `force_los` is
  set to True/False. Path loss is enabled and OFDM channel normalization is
  disabled, so geometry-dependent gain is retained. Shadow fading defaults to
  False and can be enabled with `enable_shadow_fading`.
- Each occupied slot is a separate batch example with one SISO UE/AP pair.
  Changing the number of occupied slots resets topology buffers as necessary.
  Idle slots are omitted from the channel batch but retained in slot-result
  mapping. Repeated UE IDs are permitted; fast fading is resampled per occupied
  slot and across steps. This is not a persistent spatially consistent channel
  process and does not model TDD reciprocity or switching guards.
- Both antennas are single vertically polarized omnidirectional PHY elements
  with zero array orientations. The base RT array rotation is not reused as a
  PHY PanelArray rotation because their boresight conventions differ.
- Default AP placement follows the shared perimeter placement, with its height
  set to 10 m for UMi. UE height keeps this subproject's `UE_HEIGHT` value (1 m).
  For simulations within the standard UMi calibration range, use UE height at
  least 1.5 m and appropriate link distances/heights. You can supply explicit
  `init_ue_pos` and `ap_pos` or adjust `UE_HEIGHT` for generated positions.
- The resource grid derives T, F, subcarrier spacing and CP from shared wireless
  parameters. A single-transmitter grid reserves pilot symbols [2,T-3] (or [0]
  for short slots); no many-UE pilot grid or dummy allocator is needed.

## Estimation error and PHY abstraction

`wireless.channel_estimation_error_variance` defaults to **0.0**. It is an
absolute complex channel-coefficient MSE **after path loss**, not a relative
error percentage:

```text
h_hat = h + e
E[abs(e)**2] = channel_estimation_error_variance
Var(real(e)) = Var(imag(e)) = channel_estimation_error_variance / 2
```

`result.channel` always contains the physical propagation channel.
`result.channel_estimate` contains the noisy estimate, and
`result.channel_error_variance` records the configured variance. At zero
variance the estimate equals the channel exactly and no estimation-error
random samples are drawn. To change it later:

```python
scenario.wireless.channel_estimation_error_variance = 1e-10  # Example absolute MSE
```

Choose the variance relative to the attenuated channel power in your scenario.
The noise is independent per CFR coefficient; it is an artificial estimator
model, not an implemented pilot estimator.

`SisoPhyAbstraction` uses `InnerLoopLinkAdaptation` to choose MCS from the
estimated channel and `PHYAbstraction` to sample transport-block ACK/NACK from
the physical channel. Nonzero error uses an additional uncorrelated-noise
approximation in both the prediction and reception SINRs:

```text
prediction SINR = P_RE * abs(h_hat)**2 / (N_RE + P_RE * error_variance)
reception SINR  = P_RE * abs(h)**2     / (N_RE + P_RE * error_variance)
```

This is a system-level approximation of estimation mismatch, not a waveform
simulation of equalization or CRC decoding. At zero variance it reduces to
perfect-CSI SISO reception. The optional `channel_estimate` and `error_variance`
arguments also work when using `SisoPhyAbstraction` directly.

Total UE transmit power and AP thermal noise come from `WirelessParams`; both
are divided by F before being passed as per-subcarrier quantities. Downlink
uses AP transmit power and UE noise and sets the SYS MCS category accordingly.
The shared parameter class currently computes both receiver noise powers using
a 9 dB noise figure.

## External scheduler and episode helper

`step` accepts zero-based UE IDs and -1 for idle. It accepts up to `num_slots`
entries and rejects invalid IDs, collisions encoded as multiple users in a
slot, and calls beyond `num_frames`. Resolve push contention before calling
this PHY; budget pairs and contention thresholds are not allocation vectors.
The scenario does not run a MAC scheduler or update its beliefs/queues.

```python
allocation = pull_scheduler.schedule_pps(pull_resources)
result = scenario.step(allocation)
observations = np.full(len(allocation), -1, dtype=int)
observations[result.decoded] = node_states[allocation[result.decoded]]
```

Check the application payload against the SYS-reported transport capacity
before treating a full packet as delivered. Retransmissions and segmentation
remain with the caller; there is no automatic HARQ loop.

For a precomputed allocation sequence:

```python
from phy_mac_isac.anomaly_tracking import run_episode
results = run_episode(
    1, [[0, -1, 1], [2, 0]],
    env_kwargs={"num_ue": 3, "num_frames": 2, "num_slots": 3},
    device="cpu",
)
```

This helper replaces the copied semantic-ISAC runner: it returns a list of
`StepResult` values and does not create unrelated semantic risk metrics or save
files automatically. Manage the scenario directly when you need to retain and
save its `c`, `wireless`, and `geo` dataclasses together with PHY results.

## Validation

```bash
python -m unittest discover -s tests -v
MPLBACKEND=Agg PYTHONPATH=src python -m unittest discover -s tests/integration -v
```

The integration suite uses real Sionna PHY/SYS on CPU. It checks the inherited
setup/mobility flow, random placement, stationary and moving UEs, restored
geometry, allocation order, geometry-dependent path loss, variable batches,
idle frames, uplink/downlink, power scaling, zero/nonzero estimation error,
separate prediction/reception channels and singular mobility covariances.
