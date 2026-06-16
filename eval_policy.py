"""Launch MuJoCo-Warp put-bottles eval."""

import tyro

from abc_minimal.config import SimEvalConfig
from abc_minimal.eval_policy import main as eval_policy


def main():
    eval_policy(tyro.cli(SimEvalConfig))


if __name__ == "__main__":
    main()
