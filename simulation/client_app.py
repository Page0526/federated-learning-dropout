import torch 
from collections import OrderedDict
from flwr.client import NumPyClient
from model.helper import train, test
from flwr.common import Client, Context
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import BrainMRINet
import torch.nn as nn 


class FlowerClient(NumPyClient):
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, epochs,  device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.epochs = epochs

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        if not parameters:
            return

        state_dict = OrderedDict()
        params_dict = zip(self.model.state_dict().keys(), parameters)

        for k, v in params_dict:
            if v.ndim == 0:
                state_dict[k] = torch.tensor(v)
            else:
                state_dict[k] = torch.tensor(v)

        if state_dict:
            self.model.load_state_dict(state_dict, strict=False)



    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loss, train_acc = train(self.model, self.train_dataloader, self.optimizer, self.criterion, self.epochs, self.device)
        return self.get_parameters(config={}), len(self.train_dataloader.dataset), {"loss": train_loss, "accuracy": train_acc}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test(self.model, self.val_dataloader, self.criterion, self.device)
        return loss, len(self.val_dataloader.dataset), {"accuracy": acc,  "loss": loss}
    






def client_fn(context: Context):

    client_id = context.node_config['partition-id']
    DEVICE = context.resources['device']
    EPOCHS = context.resources['epochs']
    client_datasets = context.resources['client_datasets']

    train_dataset, val_dataset = client_datasets[client_id]

    train_dataloader = DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers = 2)
    val_dataloader = DataLoader(val_dataset, batch_size = 4, shuffle = False, num_workers = 2)

    model = BrainMRINet(num_classes = 2)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    return FlowerClient(model, train_dataloader, val_dataloader, optimizer, criterion, EPOCHS, DEVICE).to_client()






if __name__ == "__main__":


    def test_flower_client():
        print("Testing FlowerClient...")
       
    


