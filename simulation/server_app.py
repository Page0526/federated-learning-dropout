from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
from simulation.strategy import DropoutFedAvg
from flwr.common import Context



def run_dropout_experiment(
    client_fn,
    num_clients: int,
    num_rounds: int = 5,
    dropout_rate: float = 0.3,
    dropout_pattern: str = "random",
    fixed_clients: Optional[List[int]] = None,
    experiment_name: str = "dropout_experiment",
    save_results: bool = True,
    resource_config : Optional[Dict[str, float]] = None,
):
    """Run a federated learning experiment with client dropout.

    Args:
        client_fn: Function to create client applications
        num_clients: Number of clients to simulate
        num_rounds: Number of federated learning rounds
        dropout_rate: Probability of clients dropping out (0.0-1.0)
        dropout_pattern: Pattern for client dropout ("random", "alternate", "fixed")
        fixed_clients: List of client IDs that will never drop out
        experiment_name: Name for saving experiment results
        save_results: Whether to save results to disk

    Returns:
        Tuple containing results dict and history dict
    """
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

    # Create client and server apps
    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)

    # Configure backend
    backend_config = {
        "client_resources": {
            "num_cpus": 1,
            "num_gpus": 0.0
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
            resources = resource_config,
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
        plt.figure(figsize=(15, 7))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(rounds, accuracy_values, 'o-', label=f'{dropout_pattern} dropout')
        plt.title(f'Accuracy with {dropout_rate*100:.0f}% Dropout')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.ylim(0, 1)

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(rounds, loss_values, 'o-', color='orange', label=f'{dropout_pattern} dropout')
        plt.title(f'Loss with {dropout_rate*100:.0f}% Dropout')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.tight_layout()

        if save_results:
            # Create results directory
            os.makedirs("results", exist_ok=True)

            # Save figure
            plt.savefig(f"results/{experiment_name}.png")

            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                "round": rounds,
                "accuracy": accuracy_values,
                "loss": loss_values
            })
            metrics_df.to_csv(f"results/{experiment_name}_metrics.csv", index=False)

            # Save dropout history
            dropout_df = pd.DataFrame([
                {"round": round_num, "dropped_clients": ",".join(map(str, clients))}
                for round_num, clients in dropout_history.items()
            ])
            dropout_df.to_csv(f"results/{experiment_name}_dropout.csv", index=False)

        plt.show()

        return results, history
    except Exception as e:
        print(f"Error in dropout experiment: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}