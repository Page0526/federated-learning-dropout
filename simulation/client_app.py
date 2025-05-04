import torch 
from collections import OrderedDict
from flwr.client import NumPyClient
from model.helper import train, test
from flwr.common import  Context
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import BrainMRINet
import torch.nn as nn 
import logging 
import lightning as pl
import wandb 
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from model.gender_module import BrainMRILightningModule


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
            if v.ndim == 0:
                state_dict[k] = torch.tensor(v)
            else:
                state_dict[k] = torch.tensor(v)

        if state_dict:
            self.model.load_state_dict(state_dict, strict=False)

        
    def fit(self, parameters, config):

        self.set_parameters(parameters)

        wandb_logger = WandbLogger(
            project="brain_mri_classification",
            name=f"client_{self.client_id}_round_{config.get('round_num', 0)}",
            config={
                "learning_rate": self.model.hparams.learning_rate,
                "weight_decay": self.model.hparams.weight_decay,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
            },
            save_dir="wandb_logs",
        ) 


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
            devices=1 if torch.cuda.is_available() else None,
            logger=wandb_logger if self.use_wandb else None,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,  # Disable progress bar for cleaner logs
            log_every_n_steps=1
        )

        trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)


        metrics = {
            "loss": self.model.trainer.callback_metrics.get("train/loss", 0).item(),
            "accuracy": self.model.trainer.callback_metrics.get("train/acc", 0).item(),
            "val_loss": self.model.trainer.callback_metrics.get("val/loss", 0).item(),
            "val_accuracy": self.model.trainer.callback_metrics.get("val/acc", 0).item()
        }

        wandb.finish()

        return self.get_parameters(config={}), len(self.train_dataloader.dataset), metrics


    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        round_num = config.get("round_num", 0)
        wandb_logger = WandbLogger(
            project="brain_mri_classification",
            name=f"eval_client_{self.client_id}_round_{round_num}",
            config={
                "learning_rate": self.model.hparams.learning_rate,
                "weight_decay": self.model.hparams.weight_decay,
                "batch_size": self.model.hparams.batch_size,
                "epochs": self.epochs,
            },
            save_dir="wandb_logs",
        )

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            logger=wandb_logger if self.use_wandb else None,
            enable_progress_bar=False
        )

        results = trainer.test(self.model, dataloaders=self.val_dataloader)

        test_loss = results[0].get("test/loss", 0)
        test_accuracy = results[0].get("test/acc", 0)
        
        # Additional metrics
        metrics = {
            "loss": test_loss,
            "accuracy": test_accuracy,
        }

        wandb.finish()

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




def create_lightning_client_fn(device, epochs, client_datasets, batch_size, learning_rate, num_workers, weight_decay = 0.0001):

    def client_fn(context: Context):
        
        
        client_id = context.node_config['partition-id']

        train_dataset, val_dataset = client_datasets[client_id]

        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

        model = BrainMRINet()
        pl_model = BrainMRILightningModule(model, learning_rate=learning_rate, weight_decay=weight_decay)

        return FlowerLightningClient(pl_model, train_dataloader, val_dataloader, epochs, batch_size, device, client_id).to_client()

    return client_fn



if __name__ == "__main__":


    def test_flower_client():
        print("Testing FlowerClient...")
       
    


