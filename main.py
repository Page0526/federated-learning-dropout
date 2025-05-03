from typing import List, Dict, Tuple, Optional, Union
from simulation.server_app import run_dropout_experiment
from simulation.client_app import create_client_fn
from data.data_split import iid_client_split, same_distribution_client_split
from data.dataset import MRIDataset
import torch 
import hydra 
from omegaconf import DictConfig, OmegaConf
import logging 




logger = logging.getLogger(__name__)



@hydra.main(config_path="config", config_name="main")
def run_experiment(cfg: DictConfig) -> None:


    logger.info(f"Running experiment with config: {cfg.experiment.name}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    device = cfg.device
    epochs = cfg.train.epochs 

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


    logger.info("Running experiments")

    results, history = run_dropout_experiment(
        client_fn_creator=create_client_fn,
        num_clients=cfg.num_clients,
        num_rounds=cfg.num_rounds,
        dropout_rate=cfg.experiment.dropout_rate,
        dropout_pattern=cfg.experiment.pattern,
        experiment_name=cfg.experiment.name,
        resource_config=resources
    )


    logger.info("Experiments completed successfully")
    logger.info(f"Client Dropout History: {history}")

    return results, history



if __name__ == "__main__":

    run_experiment()