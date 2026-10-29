"""Real CPU integration tests for the shared environment and Sionna PHY/SYS."""
from copy import deepcopy
from unittest.mock import patch
import unittest

import numpy as np
import torch
from sionna.phy import config as sionna_config

from phy_mac_isac.anomaly_tracking import (
    AnomalyTrackingScenario, SharedParams, WirelessParams, Metrics, generate_environment,
)
from phy_mac_isac.modules.environment import SimulationEnvironment
from phy_mac_isac.modules.params import GeometryParams
from phy_mac_isac.utils.stats import sample_multivariate_gaussian


class AnomalyTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sionna_config.device = "cpu"
        torch.set_num_threads(2)

    def scenario(self, wireless=None, frames=4, slots=4, velocity=0.0):
        env = AnomalyTrackingScenario(
            17, {"num_ue": 3, "num_slots": slots, "num_frames": frames},
            {"force_los": True, **(wireless or {})}, load_env=False)
        env.setup_scenario(
            ap_pos=np.array([[0., 0., 10.]]),
            init_ue_pos=np.array([[20., 0., 1.5], [40., 20., 1.5], [80., 0., 1.5]]),
            ue_vel_mean=velocity, ue_vel_cov=0.)
        env.setup_modules(device="cpu")
        return env

    def test_shared_lifecycle_stationary_geometry_and_results(self):
        env = self.scenario()
        self.assertIsInstance(env, SimulationEnvironment)
        self.assertIsInstance(env.c, SharedParams)
        self.assertIsInstance(env.wireless, WirelessParams)
        self.assertIsInstance(env.geo, GeometryParams)
        self.assertIsNone(env.scene)
        self.assertFalse(env._load_scene)
        self.assertEqual(env.frame, -1)
        initial = env.geo.ue_location[-1].copy()
        for frame in range(3):
            result = env.step([2, -1, 0])
            self.assertIsInstance(result, Metrics)
            self.assertEqual(result.frame, frame)
            np.testing.assert_array_equal(env.geo.ue_location[frame], initial)
            torch.testing.assert_close(result.channel, result.channel_estimate, rtol=0, atol=0)
            self.assertEqual(tuple(result.channel.shape), (2, 1, 1, 1, 1, 14, 72))
            np.testing.assert_array_equal(result.ue_ids, [2, 0])
            np.testing.assert_array_equal(result.slot_indices, [0, 2])
            self.assertFalse(result.decoded[1])
            np.testing.assert_array_equal(result.decoded[[0, 2]], result.transmission.decoded.numpy())
            self.assertFalse(torch.allclose(result.channel[0], result.channel[1], atol=1e-14))

    def test_random_placement_uses_area_and_anomaly_height(self):
        env = AnomalyTrackingScenario(5, {"num_ue": 5}, {}, load_env=False)
        env.setup_scenario()
        self.assertEqual(env.geo.ue_location[-1].shape, (5, 3))
        from shapely import Point
        for position in env.geo.ue_location[-1]:
            self.assertTrue(env.geo.environment_shape.contains(Point(position[:2])))
            self.assertEqual(position[2], 1.)
        np.testing.assert_array_equal(env.geo.ue_vel_cov, np.zeros((5, 3, 3)))

    def test_topology_uses_current_frame_and_allocation_order(self):
        velocity = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 0.]])
        env = self.scenario(velocity=velocity)
        initial = env.geo.ue_location[-1].copy()
        with patch.object(env.channel_model, "set_topology", wraps=env.channel_model.set_topology) as setter:
            env.step([2, 0, 2])
            for frame in (1, 2):
                env.step([1, 0])
                expected = initial + (frame + 1) * velocity * env.wireless.frame_time
                np.testing.assert_allclose(env.geo.ue_location[frame], expected)
                kwargs = setter.call_args.kwargs
                np.testing.assert_allclose(kwargs["ut_loc"].numpy()[:, 0], expected[[1, 0]], rtol=1e-6)
                np.testing.assert_allclose(kwargs["ut_velocities"].numpy()[:, 0], velocity[[1, 0]])
                np.testing.assert_allclose(kwargs["bs_loc"].numpy()[:, 0], [[0, 0, 10]] * 2)

    def test_restored_current_geometry_is_used_without_resampling(self):
        env = self.scenario()
        current = np.array([[25., 5., 1.5], [35., -5., 1.5], [55., 0., 1.5]])
        env.geo.ue_location[0] = current
        env.geo.ue_velocity[0] = 0
        env._load_env = True
        with patch.object(env.channel_model, "set_topology", wraps=env.channel_model.set_topology) as setter:
            env.step([1, 2])
            np.testing.assert_array_equal(setter.call_args.kwargs["ut_loc"].numpy()[:, 0], current[[1, 2]])

    def test_changing_batch_size_idle_frames_and_frame_limit(self):
        env = self.scenario(frames=4)
        for allocation, size in (([0, 1, 2], 3), ([-1, -1], 0), ([], 0), ([2], 1)):
            result = env.step(allocation)
            self.assertEqual(result.channel.shape[0], size)
            self.assertEqual(result.transmission.harq_feedback.numel(), size)
        self.assertEqual(env.frame, 3)
        with self.assertRaisesRegex(RuntimeError, "frames"):
            env.step([0])
        self.assertEqual(env.frame, 3)

    def test_invalid_inputs_do_not_advance_frame(self):
        env = self.scenario()
        for allocation in ([3], [-2], [1.0], [True], [[0]], [0] * 5):
            with self.subTest(allocation=allocation), self.assertRaises(ValueError):
                env.step(allocation)
            self.assertEqual(env.frame, -1)
        with self.assertRaises(ValueError):
            AnomalyTrackingScenario(0, {}, {}, load_scene=True)
        for shared in ({"num_ap": 2}, {"num_ant_h": 2}):
            with self.assertRaises(ValueError):
                SharedParams(**shared)
        with self.assertRaises(ValueError):
            WirelessParams(SharedParams(), ce_error_variance=-1)
        with self.assertRaises(RuntimeError):
            AnomalyTrackingScenario(0, {}, {}, load_env=False).step([0])

    def test_estimation_error_zero_default_and_nonzero_complex_variance(self):
        env = self.scenario()
        h = torch.zeros(100000, dtype=torch.complex64)
        state = deepcopy(env.rng.bit_generator.state)
        estimate = env._estimate_channel(h)
        self.assertTrue(torch.equal(h, estimate))
        self.assertEqual(env.rng.bit_generator.state, state)
        env.wireless.ce_error_variance = 1e-8
        error = env._estimate_channel(h) - h
        self.assertAlmostEqual(error.abs().square().mean().item() / 1e-8, 1., delta=.02)
        self.assertAlmostEqual(error.real.var().item() / 5e-9, 1., delta=.02)
        self.assertAlmostEqual(error.imag.var().item() / 5e-9, 1., delta=.02)
        result = env.step([0, 1])
        self.assertFalse(torch.equal(result.channel, result.channel_estimate))
        self.assertEqual(result.channel_error_variance, 1e-8)
        self.assertTrue(torch.isfinite(result.transmission.sinr).all())

    def test_estimate_drives_mcs_but_does_not_replace_physical_channel(self):
        env = self.scenario()
        h = torch.ones((1, 1, 1, 1, 1, 14, 72), dtype=torch.complex64)
        tx = env.transmission_module
        ordinary = tx(h, tx_power=1., no=1.)
        optimistic = tx(h, tx_power=1., no=1., channel_estimate=10 * h)
        torch.testing.assert_close(ordinary.sinr, optimistic.sinr)
        self.assertGreater(optimistic.mcs_index.item(), ordinary.mcs_index.item())
        uncertain = tx(h, tx_power=1., no=1., channel_estimate=h, error_variance=1.)
        torch.testing.assert_close(uncertain.sinr, ordinary.sinr / 2)
        with self.assertRaises(ValueError):
            tx(h, channel_estimate=h[..., :-1], error_variance=0.)
        with self.assertRaises(ValueError):
            tx(h, error_variance=-1.)

    def test_geometry_dependent_pathloss_survives_ofdm_conversion(self):
        env = self.scenario(slots=64)
        result = env.step([0] * 32 + [2] * 32)
        energy = result.channel.abs().square().mean(dim=(1, 2, 3, 4, 5, 6))
        # Independent fast fading is averaged over 32 links at each distance.
        # 20 m and 80 m outdoor LOS links must not normalize to identical gain.
        self.assertGreater(energy[:32].mean().item(), 2 * energy[32:].mean().item())
        self.assertLess(energy.max().item(), 1e-5)

    def test_downlink_and_ue_specific_power_units(self):
        env = self.scenario(wireless={"ue_power": np.array([.001, .002, .003])})
        with patch.object(env, "transmission_module", wraps=env.transmission_module) as tx:
            env.step([2, 0])
            np.testing.assert_allclose(tx.call_args.kwargs["tx_power"], np.array([.003, .001]) / env.F)
            self.assertEqual(tx.call_args.kwargs["no"], env.wireless.nob_ap / env.F)
        downlink = self.scenario(wireless={"direction": "downlink", "ap_power": .1})
        result = downlink.step([1, 2])
        self.assertEqual(result.channel.shape, (2, 1, 1, 1, 1, 14, 72))
        self.assertTrue(torch.isfinite(result.transmission.tbler).all())

    def test_allocation_driven_runner(self):
        results = generate_environment(2, [[0, -1], [1]], env_kwargs={"num_ue": 2, "num_frames": 2})
        self.assertEqual([r.frame for r in results], [0, 1])

    def test_singular_mobility_covariances_and_legacy_cholesky(self):
        covariance = np.stack([np.zeros((3, 3)), np.diag([1., 1., 0.])])
        sample = sample_multivariate_gaussian(covariance, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(sample[0], [0, 0, 0])
        self.assertEqual(sample[1, 2], 0.)
        positive = np.array([[[2., .1], [.1, 1.]]])
        actual = sample_multivariate_gaussian(positive, rng=np.random.default_rng(5))
        expected = (np.linalg.cholesky(positive)[0] @ np.random.default_rng(5).standard_normal(2))[None]
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(np.linalg.LinAlgError):
            sample_multivariate_gaussian(np.array([np.diag([1., -1.])]))


if __name__ == "__main__":
    unittest.main()
