"""Entry point for OpenI batch training on the instance's CPU."""

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "src" / "clasher_new"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="persistent output directory in the training container")
    parser.add_argument("--timesteps", type=int, help="training steps; default is 5,000,000")
    args = parser.parse_args()
    if args.timesteps is not None:
        os.environ["TRAIN_TOTAL_TIMESTEPS"] = str(args.timesteps)
    if args.output_dir:
        os.environ["TRAIN_OUTPUT_DIR"] = args.output_dir
    if "TRAIN_OUTPUT_DIR" not in os.environ:
        # OpenI deployments have used both mount points; check the active task's
        # sample or logs rather than assuming either one is always available.
        for base in (Path("/model"), Path("/result")):
            if base.is_dir():
                os.environ["TRAIN_OUTPUT_DIR"] = str(base / "clash-royale")
                break
        else:
            raise RuntimeError(
                "No known persistent output mount found. Set TRAIN_OUTPUT_DIR or "
                "pass --output-dir with the output path shown in this task's sample."
            )
    os.chdir(SOURCE_DIR)  # card_utils.py loads JSON data relative to the cwd.
    sys.path.insert(0, str(SOURCE_DIR))
    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
