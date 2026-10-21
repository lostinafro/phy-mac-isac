"""Argument routing regressions that run without importing Sionna or initializing CUDA."""

import __future__
import ast
from dataclasses import InitVar, dataclass, field
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "phy_mac_isac"
SPEC = importlib.util.spec_from_file_location("argument_helpers", PACKAGE / "utils" / "common.py")
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


@dataclass
class Shared:
    num_ue: int = 12
    num_ant: int = field(init=False, default=2)


@dataclass
class ExtendedShared(Shared):
    custom_parameter: int = 3


@dataclass
class Wireless:
    c: Shared
    traffic_avg: float = 10.0
    gen_bit: object = field(init=False, default=None)


class SetupReached(Exception):
    """Stop a valid episode immediately after argument routing."""


def runner_fixture(relative_path):
    """Execute the actual episode entry point with lightweight environment stand-ins."""
    calls = []

    class Environment:
        shared_type = Shared
        wireless_type = Wireless

        def __init__(self, episode, shared_params, wireless_params,
                     env_dirpath="default", load_env=True):
            calls.append({
                "episode": episode,
                "shared": shared_params,
                "wireless": wireless_params,
                "env_dirpath": env_dirpath,
                "load_env": load_env,
            })

        def setup_scenario(self, ap_pos=None, ue_vel_mean=0.0):
            calls[-1]["ap_pos"] = ap_pos
            calls[-1]["ue_vel_mean"] = ue_vel_mean
            raise SetupReached

        def step(self, cto=True, cfo=True, sync=False):
            pass

    class SemanticEnvironment(Environment):
        shared_type = ExtendedShared

    tree = ast.parse((PACKAGE / relative_path).read_text())
    entry = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "run_episode")
    namespace = {
        "ut": helpers,
        "params": SimpleNamespace(DATA_FOLDER="default", SharedParams=Shared,
                                  WirelessParams=Wireless),
        "SimulationEnvironment": Environment,
        "SemanticISACScenario": SemanticEnvironment,
        "torch": SimpleNamespace(device=lambda value: SimpleNamespace(type="cpu")),
    }
    code = compile(ast.Module(body=[entry], type_ignores=[]), str(relative_path), "exec",
                   flags=__future__.annotations.compiler_flag)
    exec(code, namespace)
    return namespace["run_episode"], calls


class HelperTests(unittest.TestCase):
    def test_constructor_excludes_derived_fields(self):
        selected = helpers.filter_call_kwargs(Shared, num_ue=5, num_ant=99)
        self.assertEqual(selected, {"num_ue": 5})
        self.assertEqual(Shared(**selected).num_ant, 2)

    def test_restoration_keeps_derived_fields(self):
        selected = helpers.filter_kwargs(Wireless(Shared()), gen_bit=[1], unknown=2)
        self.assertEqual(selected, {"gen_bit": [1]})

    def test_constructor_includes_initvar_and_inherited_fields(self):
        @dataclass
        class Child(Shared):
            seed: InitVar[int] = 0

        selected = helpers.filter_call_kwargs(Child, num_ue=4, seed=7, num_ant=99)
        self.assertEqual(selected, {"num_ue": 4, "seed": 7})

    def test_custom_dataclass_constructor_signature(self):
        @dataclass(init=False)
        class Custom:
            stored: int = 0

            def __init__(self, supplied):
                self.stored = supplied

        self.assertEqual(helpers.filter_call_kwargs(Custom, supplied=3, stored=4),
                         {"supplied": 3})

    def test_positional_only_and_variadic_names_are_excluded(self):
        def target(position, /, accepted=1, *args, flag=False, **kwargs):
            pass

        self.assertEqual(helpers.keyword_parameters(target), {"accepted", "flag"})

    def test_union_validation_and_reserved_names(self):
        helpers.validate_kwargs({"num_ue": 2, "traffic_avg": 3}, Shared, Wireless,
                                exclude={"c"})
        with self.assertRaisesRegex(TypeError, "Unsupported env_kwargs argument.*c"):
            helpers.validate_kwargs({"c": object()}, Wireless, exclude={"c"},
                                    name="env_kwargs")


class EpisodeRoutingTests(unittest.TestCase):
    paths = (Path("modules/environment.py"), Path("semantic_isac/scenario.py"))

    def test_unknown_and_derived_arguments_fail_before_construction(self):
        for path in self.paths:
            for key in ("sweep_value", "num_ant", "gen_bit", "episode", "c", "self"):
                with self.subTest(path=path, key=key):
                    runner, calls = runner_fixture(path)
                    with self.assertRaisesRegex(TypeError, f"env_kwargs argument.*{key}"):
                        runner(0, object, env_kwargs={key: 1})
                    self.assertEqual(calls, [])

    def test_unknown_step_argument_fails_before_construction(self):
        for path in self.paths:
            with self.subTest(path=path):
                runner, calls = runner_fixture(path)
                with self.assertRaisesRegex(TypeError, "step_kwargs argument.*cfp"):
                    runner(0, object, step_kwargs={"cfp": False})
                self.assertEqual(calls, [])

    def test_supported_inputs_reach_the_correct_stage(self):
        for path in self.paths:
            with self.subTest(path=path):
                runner, calls = runner_fixture(path)
                with self.assertRaises(SetupReached):
                    runner(7, object, env_kwargs={"num_ue": 4, "traffic_avg": 8,
                           "load_env": False, "ue_vel_mean": 5},
                           step_kwargs={"cfo": False}, save_dirpath="results")
                self.assertEqual(calls, [{"episode": 7, "shared": {"num_ue": 4},
                    "wireless": {"traffic_avg": 8}, "env_dirpath": "results",
                    "load_env": False, "ap_pos": None, "ue_vel_mean": 5}])

    def test_explicit_environment_directory_does_not_duplicate_argument(self):
        for path in self.paths:
            with self.subTest(path=path):
                runner, calls = runner_fixture(path)
                with self.assertRaises(SetupReached):
                    runner(0, object, env_kwargs={"env_dirpath": "replay"},
                           save_dirpath="results")
                self.assertEqual(calls[0]["env_dirpath"], "replay")

    def test_scenario_specific_dataclass_fields_are_supported(self):
        runner, calls = runner_fixture(Path("semantic_isac/scenario.py"))
        with self.assertRaises(SetupReached):
            runner(0, object, env_kwargs={"custom_parameter": 17})
        self.assertEqual(calls[0]["shared"], {"custom_parameter": 17})


if __name__ == "__main__":
    unittest.main()
