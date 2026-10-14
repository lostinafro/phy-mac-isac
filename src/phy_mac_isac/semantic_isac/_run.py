"""Start an experiment with device visibility configured before heavy imports."""

import os
import runpy
import sys

MODULES = {
    "vs-ue": "phy_mac_isac.semantic_isac.simulate_vs_ue",
    "vs-regions": "phy_mac_isac.semantic_isac.simulate_vs_regions",
    "qualitative": "phy_mac_isac.semantic_isac.simulate_qualitative",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        raise SystemExit(
            "Usage: python -m phy_mac_isac._run "
            "{vs-ue|vs-regions|qualitative} [experiment options]"
        )

    module = MODULES[sys.argv[1]]
    experiment_args = sys.argv[2:]

    if "--cpu" in experiment_args:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sys.argv = [module, *experiment_args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()