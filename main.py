from typing import List, Dict, Tuple, Optional, Union
from simulation.server_app import run_dropout_experiment
from simulation.client_app import create_client_fn, create_lightning_client_fn
from data.data_split import iid_client_split, same_distribution_client_split
from data.dataset import MRIDataset
import torch.nn as nn 
import hydra 
from omegaconf import DictConfig, OmegaConf
import logging 
import wandb 
from dotenv import load_dotenv
import os 
import warnings
import lightning as pl 


warnings.filterwarnings('ignore')
load_dotenv()

WANDB_APIKEY = os.getenv("WANDB_APIKEY")


logger = logging.getLogger(__name__)



@hydra.main(config_path="config", config_name="main")
def run_experiment(cfg: DictConfig) -> None:


    logger.info(f"Running experiment with config: {cfg.experiment.name}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    device = cfg.device
    epochs = cfg.train.epochs 
    wandb.login(
        key =  WANDB_APIKEY if WANDB_APIKEY else cfg.tracking.api_key
    ) 
    wandb.init(
        project="federated-mri-server_torch",
        name=f"{cfg.experiment.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        group="server"
    )



    logger.info("Loading dataset")

    full_dataset = MRIDataset(root_dir=cfg.data.root_path, label_path=cfg.data.label_path)

    logger.info(f"Dataset loaded successfully with len is {len(full_dataset)}")
    logger.info(f"Splitting dataset into {cfg.num_clients} clients")

    if cfg.data.distribution == "iid":
        client_datasets = iid_client_split(full_dataset, num_client=cfg.num_clients, val_ratio=cfg.val_ratio)

    elif cfg.data.distribution == "same":
        client_datasets = same_distribution_client_split(full_dataset, num_client=cfg.num_clients, val_ratio=cfg.data.val_ratio, overlap_ratio=cfg.data.overlap_ratio, root_dir = cfg.data.root_path)
    
    else: 
        raise ValueError(f"Unknown distribution type: {cfg.data.distribution}")
    
    logger.info(f"Client datasets created successfully with {len(client_datasets)} clients")
    logger.info(f"Client datasets: {[len(client_datasets[i][0]) for i in range(len(client_datasets))]}")
    logger.info(f"Dataset split successfully")

    resources = {
        "client_datasets": client_datasets,
        "device": device,
        "epochs": epochs, 
        "batch_size": cfg.train.batch_size,
        "learning_rate": cfg.train.learning_rate,
        "num_workers": cfg.train.num_workers

    }

    model: nn.Module = hydra.utils.instantiate(cfg.model.net)


    model.to(device)


    logger.info("Running experiments")

    results, history = run_dropout_experiment(
        client_fn_creator=create_client_fn,
        model=model,
        num_clients=cfg.num_clients,
        num_rounds=cfg.num_rounds,
        dropout_rate_eval=cfg.experiment.dropout_rate_eval,
        dropout_rate_training=cfg.experiment.dropout_rate_training,
        dropout_pattern=cfg.experiment.pattern,
        experiment_name=cfg.experiment.name,
        num_gpus=cfg.gpus,
        resource_config=resources
    )

    print(f"Result is {results}")

    for idx in range(len(results["rounds"])):
        wandb.log({
            "round": idx,
            "global_accuracy": results["accuracy"][idx],
            "loss": results["loss"][idx], 
        })

    # Log dropout history
    for round_idx, dropped in history[0].items():
        wandb.log({
            "dropped_clients_train_count": len(dropped),
            "dropped_clients": wandb.Table(
                columns=["client_id"],
                data=[[client_id] for client_id in dropped]
            )
        })

    for round_idx, dropped in history[1].items():
        wandb.log({
            "dropped_clients_eval_count": len(dropped),
            "dropped_clients": wandb.Table(
                columns=["client_id"],
                data=[[client_id] for client_id in dropped]
            )
        })

    
    


    wandb.finish()


    logger.info("Experiments completed successfully")
    logger.info(f"Client Dropout History: {history}")

    return results, history



@hydra.main(config_path="config", config_name="main")
def run_experiment_with_lightning(cfg: DictConfig) -> None:
    logger.info(f"Running experiment with config: {cfg.experiment.name}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    wandb.login(
        key = WANDB_APIKEY if WANDB_APIKEY else cfg.tracking.api_key
    )

    wandb.init(
        project="federated-mri-server",
        name=f"{cfg.experiment.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        group="server"
    )

    device = cfg.device
    epochs = cfg.train.epochs 

    logger.info("Loading dataset")
    full_dataset = MRIDataset(root_dir=cfg.data.root_path, label_path=cfg.data.label_path)
    logger.info(f"Dataset loaded successfully with len is {len(full_dataset)}")
    logger.info(f"Splitting dataset into {cfg.num_clients} clients")

    if cfg.data.distribution == "iid":
        client_datasets = iid_client_split(full_dataset, num_client=cfg.num_clients, val_ratio=cfg.data.val_ratio)
    elif cfg.data.distribution == "same":
        client_datasets = same_distribution_client_split(
            full_dataset, 
            num_client=cfg.num_clients, 
            val_ratio=cfg.data.val_ratio, 
            overlap_ratio=cfg.data.overlap_ratio,
            root_dir=cfg.data.root_path
        )
    else: 
        raise ValueError(f"Unknown distribution type: {cfg.data.distribution}")
    
    logger.info(f"Client datasets created successfully with {len(client_datasets)} clients")
   
    resources = {
        "client_datasets": client_datasets,
        "device": device,
        "epochs": epochs, 
        "batch_size": cfg.train.batch_size,
        "learning_rate": cfg.train.learning_rate,
        "num_workers": cfg.train.num_workers
    }

    logger.info("Running experiments with PyTorch Lightning")

    net : nn.Module = hydra.utils.instantiate(cfg.model.net)
    pl_model : pl.LightningModule = hydra.utils.instantiate(cfg.model, net=net, learning_rate=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay, batch_size=cfg.train.batch_size)

    results, history = run_dropout_experiment(
        client_fn_creator=create_lightning_client_fn,
        pl_model=pl_model,
        num_clients=cfg.num_clients,
        num_rounds=cfg.num_rounds,
        dropout_rate_training=cfg.experiment.dropout_rate_training,
        dropout_rate_eval=cfg.experiment.dropout_rate_eval,

        dropout_pattern=cfg.experiment.pattern,
        experiment_name=cfg.experiment.name,
        num_gpus=cfg.gpus,
        resource_config=resources
    )
    logger.info("Run successfully + wandb tracking")
    print(f"Result is {results}")


    for idx in range(len(results["rounds"])):
        wandb.log({
            "round": idx,
            "train_accuracy": results["train_accuracy"][-1] if idx > len(results["train_accuracy"]) - 1 else results["train_accuracy"][idx],
            "train_loss": results["train_loss"][-1] if idx > len(results["train_loss"]) - 1 else results["train_loss"][idx],
            "test_accuracy": results["test_accuracy"][-1] if idx > len(results["test_accuracy"]) - 1 else results["test_accuracy"][idx],
            "test_loss": results["test_loss"][-1] if idx > len(results["test_loss"]) - 1 else results["test_loss"][idx],
            "test_f1": results["test_f1"][-1] if idx > len(results["test_f1"]) - 1 else results["test_f1"][idx],
            "test_precision": results["test_precision"][-1] if idx > len(results["test_precision"]) - 1 else results["test_precision"][idx],
            "test_recall": results["test_recall"][-1] if idx > len(results["test_recall"]) - 1 else results["test_recall"][idx],
        })


    # Log dropout history
    for round_idx, dropped in history[0].items():
        wandb.log({
            "dropped_clients_train_count": len(dropped),
            "dropped_clients": wandb.Table(
                columns=["client_id"],
                data=[[client_id] for client_id in dropped]
            )
        })

    for round_idx, dropped in history[1].items(): 
        wandb.log({
            "dropped_clients_eval_count": len(dropped),
            "dropped_clients": wandb.Table(
                columns=["client_id"],
                data=[[client_id] for client_id in dropped]
            )
        })




    wandb.finish()
    
    logger.info("Experiments completed successfully")
    logger.info(f"Client Dropout History: {history}")

    return results, history



if __name__ == "__main__":

    run_experiment_with_lightning()