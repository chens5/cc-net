from train import *

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

def get_dataset(dataset_name):
    return 

def get_loss_function(loss_name):
    return 


# Sweep entry point
def sweep_train():
    run = wandb.init()
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(dataset_name=cfg.dataset_name, batch_size=int(cfg.batch_size))

    model_config = {
        "model": cfg.model_type,
        "cfg": {
            "layer_type": cfg.layer_type,
            "in_node_dim": int(cfg.in_node_dim),
            "in_edge_dim": int(cfg.in_edge_dim),
            "hidden_dim": int(cfg.hidden_dim)
            "num_layers": float(cfg.dropout),
        }
    }

    loss_fn = get_loss_function(cfg.loss_name)

    wandb.run.name = f"{cfg.model_type}-{cfg.layer_type}-L{cfg.num_layers}-H{cfg.hidden_dim}-lr{cfg.lr:.1e}"
    wandb.run.save()

    train(
        dataset=dataset,
        model_config=model_config,
        device=device,
        epochs=int(cfg.epochs),
        loss_function=loss_fn,
        lr=float(cfg.lr),
    )


if __name__ == "__main__":
    """
    Usage:
      # 1) Create sweep:
      wandb sweep sweep.yaml

      # 2) Run agents (locally):
      wandb agent <entity>/<project>/<sweep_id>

    Or programmatically you can do:
      sweep_id = wandb.sweep(sweep="sweep.yaml", project="YOUR_PROJECT")
      wandb.agent(sweep_id, function=sweep_train, count=20)
    """
    sweep_train()
