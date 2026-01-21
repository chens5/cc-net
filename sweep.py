# sweep.py
import argparse
import copy
import yaml
import wandb

import datasets.datasets as datasets
from train import train 

from datasets.dataset_utils import save_dataset, convert_cfgdict_to_str

import os
from utils.globals import DATA_OUTPUT

import torch


TOP_LEVEL_KEYS = {"lr", "batch_size", "epochs", "loss_function", "device"}

def apply_sweep_overrides(base_cfg: dict, sweep_cfg: dict) -> dict:
    """
    Merge wandb sweep parameters into your experiment config.
    Supported overrides:
      - top-level: lr, batch_size, epochs, loss_function, device
      - model_config.cfg.* : hidden_dim, num_layers, lam, tau, sigma, etc
      - dataset.params.* : n_samples, noise, etc
    """
    cfg = copy.deepcopy(base_cfg)

    # top-level overrides
    for k in TOP_LEVEL_KEYS:
        if k in sweep_cfg:
            cfg[k] = sweep_cfg[k]

    # nested model_config.cfg overrides
    for k, v in sweep_cfg.items():
        if "model_config" in cfg and "cfg" in cfg["model_config"] and k in cfg["model_config"]["cfg"]['processor_cfg']['cfg']:
            cfg["model_config"]["cfg"]['processor_cfg']['cfg'][k] = v

    return cfg


def make_dataset(dataset_cfg: dict):
    """
    dataset_cfg format:
      dataset:
        type: create_two_moons
        params: { ... }
    """
    fn = getattr(datasets, dataset_cfg["type"])
    params = dataset_cfg["params"]
    return fn(**params)


def sweep_run(base_cfg: dict):
    run = wandb.init(entity='primal-dual', project="primal-dual", dir="/data/sam/wandb")
    cfg = apply_sweep_overrides(base_cfg, dict(wandb.config))

    # build dataset
    dataset_cfg = base_cfg['dataset']
    print(dataset_cfg)
    dataset_str = convert_cfgdict_to_str(dataset_cfg)
    train_filepth = f'{DATA_OUTPUT}/{dataset_str}-train.pt'
    val_filepth = f'{DATA_OUTPUT}/{dataset_str}-val.pt'
    if os.path.isfile(train_filepth) and os.path.isfile(val_filepth):
        print("Dataset exists, using cached dataset at:", train_filepth, val_filepth)
        train_dataset = torch.load(train_filepth)
        val_dataset = torch.load(val_filepth)
    else:
        # Create dataset
        # if the datasets are already created, the new datasets get saved under a uuid
        constructor = getattr(datasets, dataset_cfg['type'])
        train_dataset = constructor(dataset_cfg['params'])
        if dataset_cfg['validation']['use_train']:
            val_dataset = train_dataset
        else:
            val_dataset = constructor(dataset_cfg['params'])
        save_dataset(dataset_cfg, train_dataset, which='train')
        save_dataset(dataset_cfg, val_dataset, which='val')
    # train() will reuse the existing wandb run (because of your tiny change)
    print(cfg)
    train(train_dataset, val_dataset, dataset_str, **cfg)

    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, required=True, help="path to wandb sweep yaml")
    parser.add_argument("--count", type=int, default=1, help="number of runs for this agent")
    args = parser.parse_args()

    wandb.login()

    with open(args.sweep, "r") as f:
        full_cfg = yaml.safe_load(f)

    base_cfg = full_cfg["experiment"]

    sweep_cfg = full_cfg["sweep"]

    sweep_id = wandb.sweep(sweep_cfg, project=sweep_cfg.get("project", "primal-dual"))

    if args.count is None:
        wandb.agent(sweep_id, function=lambda: sweep_run(base_cfg))
    else:
        wandb.agent(sweep_id, function=lambda: sweep_run(base_cfg), count=args.count)


if __name__ == "__main__":
    main()
