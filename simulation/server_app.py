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



def run_dropout_experiment(
    client_fn_creator,
    num_clients: int,
    num_rounds: int = 5,
    dropout_rate: float = 0.3,
    dropout_pattern: str = "random",
    fixed_clients: Optional[List[int]] = None,
    experiment_name: str = "dropout_experiment",
    save_results: bool = True,
    resource_config : Optional[Dict[str, float]] = None,
):
    
      # Configure client app
    print(f"\nStarting experiment: {experiment_name}")
    print(f"Dropout rate: {dropout_rate}, Pattern: {dropout_pattern}")
    print(f"Fixed clients: {fixed_clients or []}")

    # Create strategy with dropout
    strategy = DropoutFedAvg(
        dropout_rate=dropout_rate,
        dropout_pattern=dropout_pattern,
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
                                , batch_size=batch_size, learning_rate=learning_rate, num_workers=num_workers)
    
    # Create client and server apps
    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)

    # Configure backend
    backend_config = {
        "client_resources": {
            "num_cpus": 1,
            "num_gpus": 1.0
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
        dropout_history = strategy.get_dropout_history()

        # Format metrics for plotting
        rounds = list(range(1, len(eval_metrics) + 1))
        accuracy_values = [metrics.get("accuracy", 0.0) for metrics in eval_metrics]
        loss_values = [metrics.get("loss", 0.0) for metrics in eval_metrics]


        results = {
            "rounds": rounds,
            "accuracy": accuracy_values,
            "loss": loss_values,
            "dropout_history": dropout_history
        }

        # Visualize results
        # plt.figure(figsize=(15, 7))

        # # Plot accuracy
        # plt.subplot(1, 2, 1)
        # plt.plot(rounds, accuracy_values, 'o-', label=f'{dropout_pattern} dropout')
        # plt.title(f'Accuracy with {dropout_rate*100:.0f}% Dropout')
        # plt.xlabel('Round')
        # plt.ylabel('Accuracy')
        # plt.grid(True)
        # plt.ylim(0, 1)

        # # Plot loss
        # plt.subplot(1, 2, 2)
        # plt.plot(rounds, loss_values, 'o-', color='orange', label=f'{dropout_pattern} dropout')
        # plt.title(f'Loss with {dropout_rate*100:.0f}% Dropout')
        # plt.xlabel('Round')
        # plt.ylabel('Loss')
        # plt.grid(True)

        # plt.tight_layout()

        # if save_results:
        #     # Create results directory
        #     os.makedirs("results", exist_ok=True)

        #     # Save figure
        #     plt.savefig(f"results/{experiment_name}.png")

        #     # Save metrics to CSV
        #     metrics_df = pd.DataFrame({
        #         "round": rounds,
        #         "accuracy": accuracy_values,
        #         "loss": loss_values
        #     })
        #     metrics_df.to_csv(f"results/{experiment_name}_metrics.csv", index=False)

        #     # Save dropout history
        #     dropout_df = pd.DataFrame([
        #         {"round": round_num, "dropped_clients": ",".join(map(str, clients))}
        #         for round_num, clients in dropout_history.items()
        #     ])
        #     dropout_df.to_csv(f"results/{experiment_name}_dropout.csv", index=False)

        # plt.show()

        return results, history
    
    except Exception as e:
        print(f"Error in dropout experiment: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}