from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
from simulation.strategy import DropoutFedAvg
from flwr.common import Context
import torch 
import lightning as pl
from typing import Union




def run_dropout_experiment(
    client_fn_creator,
    pl_model : Union[pl.LightningModule, torch.nn.Module], 
    num_clients: int,
    num_rounds: int = 5,
    dropout_rate_training: float = 0.3,
    dropout_rate_eval: float = 0.3,
    dropout_pattern_train: str = "random",
    dropout_pattern_eval: str = "random",
    fixed_clients: Optional[List[int]] = None,
    experiment_name: str = "dropout_experiment",
    save_dir: str = "model_weights",
    num_gpus : int = 0, 
    resource_config : Optional[Dict[str, float]] = None,

):
    
      # Configure client app
    print(f"\nStarting experiment: {experiment_name}")
    print(f"Dropout rate training: {dropout_rate_training}, Pattern: {dropout_pattern_train}")
    print(f"Dropout rate evaluation: {dropout_rate_eval}, Pattern: {dropout_pattern_eval   }")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Number of clients: {num_clients}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Fixed clients: {fixed_clients or []}")

    # Create strategy with dropout
    strategy = DropoutFedAvg(
        net=pl_model.model if isinstance(pl_model, pl.LightningModule) else pl_model,
        dropout_rate_training=dropout_rate_training,
        dropout_rate_eval=dropout_rate_eval,
        dropout_pattern_train=dropout_pattern_train,
        dropout_pattern_eval=dropout_pattern_eval,
        fixed_clients=fixed_clients or [],
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,

    )

    # Configure server with strategy
    def server_fn(server_context: Context):
        from flwr.server import ServerAppComponents, ServerConfig
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = resource_config.get("epochs", 1) if resource_config else 1
    client_datasets = resource_config.get("client_datasets", {}) if resource_config else {}

    
    batch_size = resource_config.get("batch_size", 32) if resource_config else 32
    learning_rate = resource_config.get("learning_rate", 0.001) if resource_config else 0.001
    num_workers = resource_config.get("num_workers", 1) if resource_config else 1
    client_fn = client_fn_creator(device=device, epochs=epochs, client_datasets=client_datasets
                                , batch_size=batch_size, pl_model=pl_model, num_workers=num_workers)
    
    # Create client and server apps
    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)

    # Configure backend
    backend_config = {
        "client_resources": {
            "num_cpus": 1,
            "num_gpus": num_gpus,
        }
    }
    history = strategy.get_dropout_history()
    # Run simulation
    try:
        run_simulation(
            client_app=client_app,
            server_app=server_app,
            num_supernodes=num_clients,
            backend_config=backend_config,
        )

        # Get metrics directly from strategy
        fit_metrics, eval_metrics = strategy.get_metrics_history()

        # Format metrics for plotting
        rounds = list(range(1, len(eval_metrics) + 1))

        train_accuracy_values = [metrics.get("train_accuracy", 0.0) for metrics in fit_metrics]
        train_loss_values = [metrics.get("train_loss", 0.0) for metrics in fit_metrics]
        

        test_accuracy_values = [metrics.get("test_accuracy", 0.0) for metrics in eval_metrics]
        test_loss_values = [metrics.get("test_loss", 0.0) for metrics in eval_metrics]
        test_f1_values = [metrics.get("test_f1", 0.0) for metrics in eval_metrics]
        test_precision_values = [metrics.get("test_precision", 0.0) for metrics in eval_metrics]
        test_recall_values = [metrics.get("test_recall", 0.0) for metrics in eval_metrics]    


        # cleanup_wandb_loggers()

        results = {
            "rounds": rounds,
            "train_accuracy": train_accuracy_values,
            "train_loss": train_loss_values,

            "test_accuracy": test_accuracy_values,
            "test_loss": test_loss_values,
            "test_f1": test_f1_values,
            "test_precision": test_precision_values,
            "test_recall": test_recall_values
        }


       


        return results, history
    
    except Exception as e:
        print(f"Error in dropout experiment: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    


