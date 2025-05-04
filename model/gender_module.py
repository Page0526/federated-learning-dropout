import lightning as pl 
import torch.nn as nn 
from torchmetrics import Accuracy, F1Score, Precision, Recall, MeanMetric
import torch.optim as optim 
import torch 


class BrainMRILightningModule(pl.LightningModule): 
    
    def __init__(self, model: nn.Module, learning_rate: float, weight_decay: float):
        super().__init__()

        self.save_hyperparameters(ignore=['model'])

        self.model = model 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()


        self.train_acc = Accuracy(task="binary" if model.classifier[-1].out_features == 1 else "multiclass", 
                                 num_classes=model.classifier[-1].out_features)
        self.val_acc = Accuracy(task="binary" if model.classifier[-1].out_features == 1 else "multiclass",
                                 num_classes=model.classifier[-1].out_features)
        
        self.test_acc = Accuracy(task="binary" if model.classifier[-1].out_features == 1 else "multiclass",
                                    num_classes=model.classifier[-1].out_features)
        
        # F1 score 
        self.val_f1 = F1Score(task="binary" if model.classifier[-1].out_features == 1 else "multiclass",
                              num_classes=model.classifier[-1].out_features)
        self.test_f1 = F1Score(task="binary" if model.classifier[-1].out_features == 1 else "multiclass",
                                num_classes=model.classifier[-1].out_features)
        # Precision
        self.val_precision = Precision(task="binary" if model.classifier[-1].out_features == 1 else "multiclass",
                                       num_classes=model.classifier[-1].out_features)   
        self.test_precision = Precision(task="binary" if model.classifier[-1].out_features == 1 else "multiclass",
                                        num_classes=model.classifier[-1].out_features)
        # Recall
        self.val_recall = Recall(task="binary" if model.classifier[-1].out_features == 1 else "multiclass",
                                 num_classes=model.classifier[-1].out_features)
        self.test_recall = Recall(task="binary" if model.classifier[-1].out_features == 1 else "multiclass",
                                  num_classes=model.classifier[-1].out_features)


    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        
        loss = nn.CrossEntropyLoss()(outputs, y)
        preds = torch.argmax(outputs, dim=1)
        
        # Update metrics
        self.train_loss(loss)
        self.train_acc(preds, y)
        
        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        
        loss = nn.CrossEntropyLoss()(outputs, y)
        preds = torch.argmax(outputs, dim=1)
        
        # Update metrics
        self.val_loss(loss)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        
        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        
        loss = nn.CrossEntropyLoss()(outputs, y)
        preds = torch.argmax(outputs, dim=1)
        
        # Update metrics
        self.test_loss(loss)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        
        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        train_acc = self.train_acc.compute()

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc
        }
        self.log_dict(metrics, prog_bar=True)




        self.train_loss.reset()
        self.train_acc.reset()


    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()

        metrics = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall
        }
        self.log_dict(metrics, prog_bar=True)

        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def on_test_epoch_end(self):
        test_loss = self.test_loss.compute()
        test_acc = self.test_acc.compute()
        test_f1 = self.test_f1.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()

        metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall
        }
        self.log_dict(metrics, prog_bar=True)

        self.test_loss.reset()
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()


    

        
    







