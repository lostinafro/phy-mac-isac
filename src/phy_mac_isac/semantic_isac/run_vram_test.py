from phy_mac_isac.modules import resource_allocation as ra
from phy_mac_isac.semantic_isac.scenario import run_episode
from phy_mac_isac.utils.simulation import profile_episode_vram


if __name__ == "__main__":
    device = "cuda:0"

    report = profile_episode_vram(
        run_episode,
        {
            "episode": 0,
            "ra_method": ra.RoundRobinAllocation,
            "env_kwargs": {
                "num_ue": 12,
                "num_frames": 1,
                "debugging": False,
                "load_env": False,
            },
            "device": device,
            "gpu_guard": None,
            "save": False,
            "quiet": True,
        },
        device=device,
    )

    print(report)