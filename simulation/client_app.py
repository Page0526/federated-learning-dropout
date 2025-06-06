import torch 
from collections import OrderedDict
from flwr.client import NumPyClient
from model.helper import train, test
from flwr.common import  Context
import torch.optim as optim
from torch.utils.data import DataLoader
from model.component.cnn import BrainMRINet
import torch.nn as nn 
import logging 
import lightning as pl
import warnings 
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from model.gender_module import BrainMRILightningModule

warnings.filterwarnings('ignore')



logger = logging.getLogger(__name__)

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
    



class FlowerLightningClient(NumPyClient):


    def __init__(self, model: pl.LightningModule, train_dataloader, val_dataloader, epochs, batch_size, device, client_id): 

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.device = device 
        self.client_id = client_id
        self.batch_size = batch_size


        


    def get_parameters(self, config):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        if not parameters:
            return

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.tensor(v)
            

        if state_dict:
            self.model.load_state_dict(state_dict, strict=False)

        
    def fit(self, parameters, config):

        self.set_parameters(parameters)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./checkpoints/client_{self.client_id}",
            filename=f"round_{config.get('round_num', 0)}" + "-{epoch:02d}",
            save_top_k=1,
            monitor="val/loss",
            mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False, 
            log_every_n_steps=1
        )

        trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

        callback_metrics = trainer.callback_metrics

        train_loss = callback_metrics.get("train/loss", 0)
        train_accuracy = callback_metrics.get("train/acc", 0)
        val_loss = callback_metrics.get("val/loss", 0)
        val_accuracy = callback_metrics.get("val/acc", 0)
        val_precision = callback_metrics.get("val/precision", 0)
        val_recall = callback_metrics.get("val/recall", 0)
        val_f1 = callback_metrics.get("val/f1", 0)



        metrics = {
            "train_loss": train_loss.item() if isinstance(train_loss, torch.Tensor) else float(train_loss),
            "train_accuracy": train_accuracy.item() if isinstance(train_accuracy, torch.Tensor) else float(train_accuracy),
            "val_loss": val_loss.item() if isinstance(val_loss, torch.Tensor) else float(val_loss),
            "val_accuracy": val_accuracy.item() if isinstance(val_accuracy, torch.Tensor) else float(val_accuracy),
            "val_precision": val_precision.item() if isinstance(val_precision, torch.Tensor) else float(val_precision),
            "val_recall": val_recall.item() if isinstance(val_recall, torch.Tensor) else float(val_recall),
            "val_f1": val_f1.item() if isinstance(val_f1, torch.Tensor) else float(val_f1)
        }



        return self.get_parameters(config={}), len(self.train_dataloader.dataset), metrics


    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

       
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1 ,
            enable_progress_bar=False
        )

        results = trainer.test(self.model, dataloaders=self.val_dataloader)
        
        callback_metrics = trainer.callback_metrics

        test_loss = callback_metrics.get("test/loss", 0)
        test_accuracy = callback_metrics.get("test/acc", 0)
        test_f1 = callback_metrics.get("test/f1", 0)
        test_precision = callback_metrics.get("test/precision", 0)
        test_recall = callback_metrics.get("test/recall", 0)
        
        # Additional metrics
        metrics = {
            "test_loss": test_loss.item() if isinstance(test_loss, torch.Tensor) else float(test_loss),
            "test_accuracy": test_accuracy.item() if isinstance(test_accuracy, torch.Tensor) else float(test_accuracy),
            "test_f1": test_f1.item() if isinstance(test_f1, torch.Tensor) else float(test_f1),
            "test_precision": test_precision.item() if isinstance(test_precision, torch.Tensor) else float(test_precision),
            "test_recall": test_recall.item() if isinstance(test_recall, torch.Tensor) else float(test_recall)
        }

       
        return float(test_loss), len(self.val_dataloader.dataset), metrics




def create_client_fn(device, epochs, client_datasets, batch_size, learning_rate, num_workers):

    def client_fn(context: Context):
    
        client_id = context.node_config['partition-id']
        

        train_dataset, val_dataset = client_datasets[client_id]

        

        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

        model = BrainMRINet()
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        criterion = nn.CrossEntropyLoss()
        return FlowerClient(model, train_dataloader, val_dataloader, optimizer, criterion, epochs, device).to_client()

    return client_fn




def create_lightning_client_fn(device, epochs, client_datasets, batch_size, num_workers, pl_model):

    def client_fn(context: Context):
        
        
        client_id = context.node_config['partition-id']
        train_dataset, val_dataset = client_datasets[client_id]

        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

        # model = BrainMRINet()
        # pl_model = BrainMRILightningModule(model, learning_rate=learning_rate, weight_decay=weight_decay, batch_size = batch_size )


        return FlowerLightningClient(pl_model, train_dataloader, val_dataloader, epochs, batch_size, device, client_id).to_client()

    return client_fn



if __name__ == "__main__":


    def test_flower_client():
        print("Testing FlowerClient...")
       
    


