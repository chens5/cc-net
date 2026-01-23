import models.models as models
import losses.losses as losses
import datasets.datasets as datasets
from datasets.dataset_utils import save_dataset, convert_cfgdict_to_str
import wandb
import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam, AdamW
from tqdm import tqdm, trange
import yaml
import os
import argparse
from utils.globals import GLOBAL_OUTPUT, DATA_OUTPUT


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

def compute_validation_loss(val_dataloader, model, loss_func, lam, device):
    val_loss = 0.0
    primal_obj = 0.0
    avg_fidelity = 0.0
    avg_fusion = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            e_init = batch.x[src] - batch.x[dst]

            h, e = model(h=batch.x.float(), e=e_init.float(), edge_index=batch.edge_index,w=batch.edge_attr,x=batch.x.float())
            loss_terms = {'U': h, 'X': batch.x, 'src': src, 'dst': dst, 'P': e, 'w': batch.edge_attr, 'lam': lam}
            loss, fidelity, fusion = loss_func(**loss_terms, return_parts=True)
            primal_obj_ = losses.energy(**loss_terms)
            avg_fidelity += fidelity.item()
            avg_fusion += fusion.item()
            # loss = loss_func(h, batch.x, src,dst,batch.edge_attr,lam=lam,)

            val_loss += loss.item()
            primal_obj += primal_obj_.item()

    val_loss /= len(val_dataloader)
    primal_obj /=len(val_dataloader)
    avg_fidelity = avg_fidelity/len(val_dataloader)
    avg_fusion = avg_fusion/len(val_dataloader)
    return val_loss, primal_obj, avg_fidelity, avg_fusion

def compute_kkt_residuals(val_dataloader, model, lam, device, eps=1e-8):
    return_dict = {'stat_rel': 0.0, 'feas_rel': 0.0, 'align_rel': 0.0, 'kkt_rel': 0.0}
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            e_init = batch.x[src] - batch.x[dst]

            h, e = model(h=batch.x.float(), e=e_init.float(), edge_index=batch.edge_index,w=batch.edge_attr, x=batch.x.float())
            kkt_dict = losses.kkt_residuals(h, e, batch.x, src, dst, batch.edge_attr, lam)
            for key in return_dict:
                return_dict[key] += kkt_dict[key]
    for key in return_dict:
        return_dict[key] /= len(val_dataloader)
    return return_dict

def train(train_dataset, val_dataset,dataset_str, model_config, device, 
          epochs, loss_function, lr, batch_size=1, checkpoint_epoch=10, 
          **kwargs):
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
    Check the example.yaml files for how for format the required function parameters
      
    '''

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # initialize model

    # TODO: fix the model initialization here for EncodeProcessDecode model.
    # TODO: fix the yaml to be compatible
    model_class = getattr(models, model_config['model'])
    model = model_class(**model_config['cfg'])
    model = model.float()
    model = model.to(device)

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
    if model_config['model']=='EncodeProcessDecode':
        processor_cfg = model_config['cfg']['processor_cfg']
        processor_cfg['cfg']['in_node_dim'] = model_config['cfg']['embedding_dim']
        processor_cfg['cfg']['in_edge_dim'] = model_config['cfg']['embedding_dim']
        processor_cfg['cfg']['lam'] =model_config['cfg']['lam']
        modelstring = f"{processor_cfg['model']}/{make_modelstring(processor_cfg['cfg'])}_resid={model_config['cfg']['residual_stream']}_steps={model_config['cfg']['recurrent_steps']}_featDim={model_config['cfg']['in_node_dim']}"
    else:
        modelstring = make_modelstring(model_config['cfg'])
    filepth = os.path.join(GLOBAL_OUTPUT, 
                           loss_function,
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
            batch = batch.to(device)
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            e_init = batch.x[src] - batch.x[dst]
            h, e = model(h=batch.x.float(), 
                         e=e_init.float(), 
                         edge_index = batch.edge_index, 
                         w=batch.edge_attr,
                         x=batch.x.float())
            loss_terms = {'U': h, 'X': batch.x, 'src': src, 'dst': dst, 'P': e, 'w': batch.edge_attr, 'lam': lam}
            loss = loss_func(**loss_terms)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        validation_loss, primal_objective, fidelity, fusion = compute_validation_loss(val_dataloader, model, loss_func, lam=lam, device=device)
        kkt_res_dict = compute_kkt_residuals(val_dataloader, model, lam, device=device)
        wandb.log({
        "train/loss": train_loss,
        "val/loss": validation_loss,
        "val/primal_objective": primal_objective,
        "val/fidelity": fidelity,
        "val/fusion": fusion,
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
    parser.add_argument('--no-cached-data', action='store_true', help='regenerate dataset and ignore previously cached data')
    args = parser.parse_args()

    with open(args.experiment, "r") as f:
        cfg = yaml.safe_load(f)
        print(cfg)

    
    # Simple loading and caching data 
    dataset_cfg = cfg['dataset']
    dataset_str = convert_cfgdict_to_str(dataset_cfg)
    train_filepth = f'{DATA_OUTPUT}/{dataset_str}-train.pt'
    val_filepth = f'{DATA_OUTPUT}/{dataset_str}-val.pt'
    if not args.no_cached_data and os.path.isfile(train_filepth) and os.path.isfile(val_filepth):
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
    # Train network
    # print(train_dataset)
    train(train_dataset=train_dataset, val_dataset=val_dataset, dataset_str=dataset_str, **cfg)


