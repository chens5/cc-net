import models.models as models
import losses.losses as losses
import datasets.datasets as datasets
import wandb
import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam, AdamW
from tqdm import tqdm, trange
import yaml
import os
import argparse

# os.environ['WANDB_API_KEY'] = '395d2fa6b086e2f1063586bbcd6a65f8a14eca9c'

GLOBAL_OUTPUT = '/data/sam/primal-dual'

def make_modelstring(cfg: dict) -> str:
    return (
        f"{cfg['layer_type']}"
        f"_L{cfg['num_layers']}"
        f"_H{cfg['hidden_dim']}"
        f"_inN{cfg['in_node_dim']}"
        f"_inE{cfg['in_edge_dim']}"
        f"_lam{cfg['lam']}"
        f"_tau{cfg['tau']}"
        f"_sig{cfg['sigma']}"
    )

def compute_validation_loss(val_dataloader, model, loss_func, lam):
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            e_init = batch.x[src] - batch.x[dst]

            h, e = model(h=batch.x.float(), e=e_init.float(), edge_index=batch.edge_index,w=batch.edge_attr)

            loss = loss_func(h, batch.x, src,dst,batch.edge_attr,lam=lam,)

            val_loss += loss.item()

    val_loss /= len(val_dataloader)

    return val_loss

def compute_kkt_residuals(val_dataloader, model, lam, eps=1e-8):
    return_dict = {'stat_rel': 0.0, 'feas_rel': 0.0, 'align_rel': 0.0, 'kkt_rel': 0.0}
    with torch.no_grad():
        for batch in val_dataloader:
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            e_init = batch.x[src] - batch.x[dst]

            h, e = model(h=batch.x.float(), e=e_init.float(), edge_index=batch.edge_index,w=batch.edge_attr)
            kkt_dict = losses.kkt_residuals(h, e, batch.x, src, dst, batch.edge_attr, lam)
            for key in return_dict:
                return_dict[key] += kkt_dict[key]
    for key in return_dict:
        return_dict[key] /= len(val_dataloader)
    return return_dict

def train(train_dataset, val_dataset, model_config, device, epochs, loss_function, lr, batch_size=1, checkpoint_epoch=10, **kwargs):
    '''
    Training pipeline for model
    Follow the below format for your model config. 
    model_config = {model: <modeltype>, 
                    cfg: {layer_type: <layer_type>, 
                    in_node_dim: <int>, 
                    in_edge_dim: <int>,
                    num_layers: <int>, 
                    lam: <float>,
                    etc. }
                    }
      
    '''

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # initialize model
    model_class = getattr(models, model_config['model'])
    model = model_class(**model_config['cfg'])
    model = model.float()
    
    # Set config for lambda
    assert 'lam' in model_config['cfg']
    lam = model_config['cfg']['lam']

    # initialize wandb
    run = wandb.init(
        entity='primal-dual',
        project='primal-dual',
        dir='/data/sam/wandb',
        config={
            "model_config": model_config,
            "epochs": epochs,
            "loss_function": loss_function,
            "learning_rate": lr
        }
    ) if wandb.run is None else wandb.run

    wandb.config.update(
    {
        "model_config": model_config,
        "epochs": epochs,
        "loss_function": loss_function,
        "learning_rate": lr,
        "batch_size": batch_size,
    },
    allow_val_change=True,
    )

    # arrange save file
    # OUTPUT_FILE/dataset/model/{modelstring}
    wandb_id = wandb.run.id
    modelstring = make_modelstring(model_config['cfg'])
    filepth = os.path.join(GLOBAL_OUTPUT, 
                           kwargs['dataset']['type'], 
                           model_config['model'], 
                           modelstring, 
                           wandb_id)
    if not os.path.exists(f'{filepth}/checkpoints'):
        os.makedirs(f'{filepth}/checkpoints')

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_func = getattr(losses, loss_function)
    print("saving checkpoints and model in:", filepth)

    for epoch in trange(epochs):
        train_loss = 0.0
        for batch in train_dataloader: 
            # run model here
            optimizer.zero_grad()
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            e_init = batch.x[src] - batch.x[dst]
            h, e = model(h=batch.x.float(), 
                         e=e_init.float(), 
                         edge_index = batch.edge_index, 
                         w=batch.edge_attr)
            
            loss = loss_func(h, 
                             batch.x, 
                             src, 
                             dst, 
                             batch.edge_attr, 
                             lam=lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        validation_loss = compute_validation_loss(val_dataloader, model, loss_func, lam=lam)
        kkt_res_dict = compute_kkt_residuals(val_dataloader, model, lam)
        wandb.log({
        "train/loss": train_loss,
        "val/loss": validation_loss,
        "val/stationarity": kkt_res_dict['stat_rel'], 
        "val/dual-feasibility": kkt_res_dict['feas_rel'],
        "val/alignment": kkt_res_dict['align_rel'],
        'val/relative-kkt-residual': kkt_res_dict['kkt_rel'], 
        "epoch": epoch,
        })

        ## checkpoint   
        if epoch % checkpoint_epoch == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    "config": model_config,
                    "epoch": epoch,
                },
                f'{filepth}/checkpoints/{epoch}.pt',
            )          

    # save final model
    torch.save(model.state_dict(), f'{filepth}/final.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, help='yaml file with experiment configs')
    args = parser.parse_args()

    with open(args.experiment, "r") as f:
        cfg = yaml.safe_load(f)
        print(cfg)

    # Add dataset saving
    dataset_cfg = cfg['dataset']
    data = datasets.create_knn_dataset_from_base(dataset_cfg)

    train_dataset = [data]
    val_dataset = [data]

    train(train_dataset, val_dataset, **cfg)


