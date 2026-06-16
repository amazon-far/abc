"""Live Viser viewer over an ABC-DiT sim rollout."""

import tyro

from abc_minimal.config import VizPolicyConfig
from abc_minimal.viz_policy import main as viz_policy


def main():
    viz_policy(tyro.cli(VizPolicyConfig))


if __name__ == "__main__":
    main()
