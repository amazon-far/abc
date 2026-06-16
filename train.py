"""Launch ABC-DiT training."""

import tyro

from abc_minimal.config import TrainConfig
from abc_minimal.train_loop import main as train


def main():
    train(tyro.cli(TrainConfig))


if __name__ == "__main__":
    main()
