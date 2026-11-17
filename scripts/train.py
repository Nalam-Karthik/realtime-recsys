"""Run the offline training pipeline: python -m scripts.train [--epochs N] [--index flat|hnsw]"""
import argparse

from recsys.config import PipelineConfig
from recsys.pipeline import run_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/ml-latest-small")
    p.add_argument("--artifact-dir", default="artifacts")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--index", choices=["hnsw", "flat"], default=None)
    args = p.parse_args()
    cfg = PipelineConfig(data_dir=args.data_dir, artifact_dir=args.artifact_dir)
    if args.epochs:
        cfg.tower.epochs = args.epochs
    if args.index:
        cfg.index.kind = args.index
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
