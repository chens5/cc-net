import argparse
import inspect
import os

import torch
import yaml
from torch_geometric.loader import DataLoader

import datasets.datasets as datasets
import models.models as models


def load_test_dataset(dataset_cfg: dict, test_seed: int | None):
    params = dict(dataset_cfg["params"])
    if test_seed is not None:
        params["seed"] = test_seed
    constructor = getattr(datasets, dataset_cfg["type"])
    return constructor(params)


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["config"]
    model_class = getattr(models, model_cfg["model"])
    model = model_class(**model_cfg["cfg"]).float().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_cfg


def forward_batch(model, batch):
    kwargs = {
        "h": batch.x.float(),
        "e": (batch.x[batch.edge_index[0]] - batch.x[batch.edge_index[1]]).float(),
        "edge_index": batch.edge_index,
        "w": batch.edge_attr,
    }
    if "x" in inspect.signature(model.forward).parameters:
        kwargs["x"] = batch.x.float()
    return model(**kwargs)


def run(checkpoint_dir: str, checkpoint_step: int, dataset_cfg: dict, batch_size: int,
        device: str, test_seed: int | None):
    ckpt_dir = checkpoint_dir
    if os.path.isdir(os.path.join(ckpt_dir, "checkpoints")):
        ckpt_dir = os.path.join(ckpt_dir, "checkpoints")
    checkpoint_path = os.path.join(ckpt_dir, f"{checkpoint_step}.pt")

    test_dataset = load_test_dataset(dataset_cfg, test_seed=test_seed)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model, model_cfg = load_model_from_checkpoint(checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Model: {model_cfg['model']} with {len(test_dataset)} test graphs")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            h, e = forward_batch(model, batch)
            if i == 0:
                print(f"batch0 h shape: {tuple(h.shape)} e shape: {tuple(e.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        help="path to experiment yaml (for dataset config)")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="wandb run dir or checkpoints dir")
    parser.add_argument("--checkpoint-step", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test-seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.experiment, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 1)

    run(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        dataset_cfg=dataset_cfg,
        batch_size=batch_size,
        device=args.device,
        test_seed=args.test_seed,
    )


if __name__ == "__main__":
    main()
