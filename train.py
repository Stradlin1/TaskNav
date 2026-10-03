import argparse

from ultralytics import YOLO
from ultralytics.cfg import get_cfg


DEFAULT_CFG = "ultralytics/cfg/default.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Train TaskNav with the default or a reproducible experiment config.")
    parser.add_argument("--cfg", default=DEFAULT_CFG, help="Training YAML used to build and train the model.")
    return parser.parse_args()


def main():
    cli = parse_args()
    args = get_cfg(cli.cfg)
    model = YOLO(args.model, task=args.task)
    model.train(cfg=cli.cfg)


if __name__ == "__main__":
    main()
