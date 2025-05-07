import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg





class DropoutFedAvg(FedAvg):
    """FedAvg strategy with client dropout simulation and metrics tracking."""

    def __init__( self, dropout_rate: float = 0.3, fixed_clients: Optional[List[int]] = None, dropout_pattern: str = "random", **kwargs):
    
        if "fit_metrics_aggregation_fn" not in kwargs:
            kwargs["fit_metrics_aggregation_fn"] = self.weighted_average
        if "evaluate_metrics_aggregation_fn" not in kwargs:
            kwargs["evaluate_metrics_aggregation_fn"] = self.weighted_average

        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.fixed_clients = fixed_clients or []
        self.dropout_pattern = dropout_pattern
        self.current_round = 0
        self.dropped_clients_history: Dict[int, List[int]] = {}

        # For tracking metrics
        self.fit_metrics_history: List[Dict[str, float]] = []
        self.eval_metrics_history: List[Dict[str, float]] = []


    def weighted_average(self, metrics: List[Tuple[int, Dict]]) -> Dict:
        """Aggregate metrics using weighted average based on number of samples."""
        if not metrics:
            return {}

        total_examples = sum([num_examples for num_examples, _ in metrics])
        weighted_metrics = {}

        # Calculate weighted metrics
        for metric_key in metrics[0][1].keys():
            weighted_sum = sum(
                metric_dict[metric_key] * num_examples
                for num_examples, metric_dict in metrics
                if metric_key in metric_dict
            )
            weighted_metrics[metric_key] = weighted_sum / total_examples if total_examples > 0 else 0

        return weighted_metrics


    def configure_fit( self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with client dropout."""
        self.current_round = server_round

    
        client_fit_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        if not client_fit_instructions:
            return []


        available_clients = self._apply_dropout(client_fit_instructions)

        # Save dropout history for this round
        client_ids = [int(client.cid) for client, _ in client_fit_instructions]
        available_client_ids = [int(client.cid) for client, _ in available_clients]
        dropped_clients = [cid for cid in client_ids if cid not in available_client_ids]
        self.dropped_clients_history[server_round] = dropped_clients

        print(f"Round {server_round}: {len(dropped_clients)} clients dropped out of {len(client_ids)}")
        print(f"Dropped client IDs: {dropped_clients}")

        return available_clients
    



    # def configure_evaluate(self): 
    #     pass 




    def _apply_dropout( self, client_instructions: List[Tuple[ClientProxy, FitIns]]) -> List[Tuple[ClientProxy, FitIns]]:
        """Apply dropout to clients based on the specified pattern."""
        if len(client_instructions) == 0:
            return []

        # Get all client IDs
        all_clients = [(client, ins) for client, ins in client_instructions]
        all_client_ids = [int(client.cid) for client, _ in all_clients]

        # Determine which clients will drop out
        dropout_mask = [False] * len(all_clients)

        if self.dropout_pattern == "random":
           
            for i, cid in enumerate(all_client_ids):
                
                if cid in self.fixed_clients:
                    continue
            
                if random.random() < self.dropout_rate:
                    dropout_mask[i] = True

        elif self.dropout_pattern == "alternate":
         
            if self.current_round % 2 == 1:  
                for i, cid in enumerate(all_client_ids):
                    if cid not in self.fixed_clients:
                        dropout_mask[i] = True

        elif self.dropout_pattern == "fixed":
      
            n_dropout = int(len(all_clients) * self.dropout_rate)
            for i in range(n_dropout):
                if all_client_ids[i] not in self.fixed_clients:
                    dropout_mask[i] = True

        
        available_clients = [
            (client, ins) for i, (client, ins) in enumerate(all_clients)
            if not dropout_mask[i]
        ]

        return available_clients

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
        """Aggregate fit results and store metrics."""
        aggregated = super().aggregate_fit(server_round, results, failures)

        # Store metrics if available
        if results:
            metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.weighted_average(metrics)
            self.fit_metrics_history.append(aggregated_metrics)

            print(f"Round {server_round} training metrics: {aggregated_metrics}")

        return aggregated

    def aggregate_evaluate( self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],  failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if results:
            metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.weighted_average(metrics)
            self.eval_metrics_history.append(aggregated_metrics)

            print(f"Round {server_round} evaluation metrics: {aggregated_metrics}")

        return aggregated

    def get_dropout_history(self) -> Dict[int, List[int]]:
        return self.dropped_clients_history

    def get_metrics_history(self) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        return self.fit_metrics_history, self.eval_metrics_history