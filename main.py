from typing import List, Dict, Tuple, Optional, Union
from simulation.server_app import run_dropout_experiment
from simulation.client_app import create_client_fn
from data.data_split import iid_client_split, same_distribution_client_split
from data.dataset import MRIDataset
import torch 



def conduct_dropout_experiments(client_fn, resource_config, num_clients=3, num_rounds=5):
    """Run a series of experiments with different dropout configurations."""

    # Experiment 1: Random dropout with 30% rate
    exp1_results, exp1_history = run_dropout_experiment(
        client_fn_creator=client_fn,
        num_clients=num_clients,
        num_rounds=num_rounds,
        dropout_rate=0.3,
        dropout_pattern="random",
        experiment_name="random_dropout_30pct",
        resource_config = resource_config
    )

    # # Experiment 2: Alternate dropout (every other round)
    # exp2_results, exp2_history = run_dropout_experiment(
    #     client_fn=client_fn,
    #     num_clients=num_clients,
    #     num_rounds=num_rounds,
    #     dropout_pattern="alternate",
    #     experiment_name="alternate_dropout" ,
    #     resource_config= resouce_config
    # )

    # # Experiment 3: Fixed dropout (same clients always drop out)
    # exp3_results, exp3_history = run_dropout_experiment(
    #     client_fn=client_fn,
    #     num_clients=num_clients,
    #     num_rounds=num_rounds,
    #     dropout_rate=0.3,
    #     dropout_pattern="fixed",
    #     experiment_name="fixed_dropout_30pct", 
    #     resource_config= resouce_config
    # )

    # # Experiment 4: High dropout rate (70%)
    # exp4_results, exp4_history = run_dropout_experiment(
    #     client_fn=client_fn,
    #     num_clients=num_clients,
    #     num_rounds=num_rounds,
    #     dropout_rate=0.7,
    #     dropout_pattern="random",
    #     experiment_name="random_dropout_70pct", 
    #     resource_config= resouce_config
    # )

    # # Experiment 5: With fixed clients that never drop out
    # exp5_results, exp5_history = run_dropout_experiment(
    #     client_fn=client_fn,
    #     num_clients=num_clients,
    #     num_rounds=num_rounds,
    #     dropout_rate=0.5,
    #     dropout_pattern="random",
    #     fixed_clients=[0],  # Client 0 never drops out
    #     experiment_name="random_dropout_with_fixed", 
    #     resource_config= resouce_config
    # )

    return {
        "random_30pct": (exp1_results, exp1_history)
        # "alternate": (exp2_results, exp2_history),
        # "fixed_30pct": (exp3_results, exp3_history),
        # "random_70pct": (exp4_results, exp4_history),
        # "random_with_fixed": (exp5_results, exp5_history)
    }



if __name__ == "__main__":


    ROOT_PATH = "dataset/not_skull_stripped"
    LABEL_PATH = "dataset/label.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 1

    full_dataset = MRIDataset(root_dir=ROOT_PATH, label_path=LABEL_PATH)
    print("Pass initialization")

    client_datasets = same_distribution_client_split(full_dataset, num_client=3, val_ratio=0.2, overlap_ratio=0.2)
    resources = {"client_datasets": client_datasets, "device": DEVICE, "epochs": EPOCHS}
    print("Pass dataset creation")

    conduct_dropout_experiments(
        client_fn=create_client_fn,
        num_clients=len(client_datasets),
        num_rounds=1, 
        resource_config=resources
    )
    print("Pass experiment")
