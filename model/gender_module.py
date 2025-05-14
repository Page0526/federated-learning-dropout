import lightning as pl 
import torch.nn as nn 
from torchmetrics import Accuracy, F1Score, Precision, Recall, MeanMetric
import torch.optim as optim 
import torch 


class BrainMRILightningModule(pl.LightningModule): 
    
    def __init__(self, net: nn.Module, learning_rate: float, weight_decay: float, batch_size: int ):
        super().__init__()

        self.save_hyperparameters(ignore=['net'])

        self.model = net 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size


        self.train_acc = Accuracy(task= "multiclass",  num_classes=net.classifier[-1].out_features)
        self.val_acc = Accuracy(task= "multiclass", num_classes=net.classifier[-1].out_features)
        
        self.test_acc = Accuracy(task= "multiclass",  num_classes=net.classifier[-1].out_features)
        
        # F1 score 
        self.val_f1 = F1Score(task="multiclass", num_classes=net.classifier[-1].out_features)
        self.test_f1 = F1Score(task="multiclass", num_classes=net.classifier[-1].out_features)
        # Precision
        self.val_precision = Precision(task="multiclass", num_classes=net.classifier[-1].out_features)   
        self.test_precision = Precision(task="multiclass", num_classes=net.classifier[-1].out_features)
        # Recall
        self.val_recall = Recall(task="multiclass", num_classes=net.classifier[-1].out_features)
        self.test_recall = Recall(task="multiclass", num_classes=net.classifier[-1].out_features)

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,  # Adjust value as needed
        }
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        
        loss = nn.CrossEntropyLoss()(outputs, y)
        preds = torch.argmax(outputs, dim=1)
        
        # Update metrics
        acc = self.train_acc(preds, y)
        

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        
        loss = self.criterion(outputs, y)
        preds = torch.argmax(outputs, dim=1)
        
        # Update metrics
       
        acc =  self.val_acc(preds, y)
        f1 =  self.val_f1(preds, y)
        precision =  self.val_precision(preds, y)
        recall =  self.val_recall(preds, y)
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", recall, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        
        loss = self.criterion(outputs, y)
        preds = torch.argmax(outputs, dim=1)
        
        # Update metrics
        acc = self.test_acc(preds, y)
        f1 = self.test_f1(preds, y)
        precision = self.test_precision(preds, y)
        recall = self.test_recall(preds, y)
        
        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", recall, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    



class DenseNetModule(pl.LightningModule):
    def __init__(self, net, learning_rate=1e-3, weight_decay = 1e-2, batch_size = 32):
        super().__init__()
        self.model = net
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        # how confidence model is in it prediction
        # tức model có thể rất tự tin trong quyết định nhưng thực tế lại sai
        # BCE = y*log(y_pred) + (1 - y)*log(1 - y_pred)
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.train_accuracy = Accuracy(task="binary", num_classes=1)
        self.val_accuracy = Accuracy(task="binary", num_classes=1)
        self.test_accuracy = Accuracy(task="binary", num_classes=1)


        self.val_precision = Precision(task="binary", num_classes=1)
        self.test_precision = Precision(task="binary", num_classes=1)

        self.val_recall = Recall(task="binary", num_classes=1)
        self.test_recall = Recall(task="binary", num_classes=1)

        self.val_f1 = F1Score(task="binary", num_classes=1)
        self.test_f1 = F1Score(task="binary", num_classes=1)

    

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x.shape  = (batch_size, in_channel, height, width, depth), y.shape = (batch_size)
        logits = self(x.unsqueeze(1))
        
        loss = self.criterion(logits, y.float().unsqueeze(1))
        
        acc = self.train_accuracy((torch.sigmoid(logits) > 0.5).float(), y.unsqueeze(1))

        
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.unsqueeze(1))
        
        loss = self.criterion(logits, y.float().unsqueeze(1))
        acc = self.val_accuracy((torch.sigmoid(logits) > 0.5).float(), y.unsqueeze(1))
        f1 = self.val_f1((torch.sigmoid(logits) > 0.5).float(), y.unsqueeze(1))
        precision = self.val_precision((torch.sigmoid(logits) > 0.5).float(), y.unsqueeze(1))
        recall = self.val_recall((torch.sigmoid(logits) > 0.5).float(), y.unsqueeze(1))
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recall', recall, on_step=False, on_epoch=True, prog_bar=True)

        return loss



    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.unsqueeze(1))
        
        loss = self.criterion(logits, y.float().unsqueeze(1))
        acc = self.test_accuracy((torch.sigmoid(logits) > 0.5).float(), y.unsqueeze(1))
        f1 = self.test_f1((torch.sigmoid(logits) > 0.5).float(), y.unsqueeze(1))
        precision = self.test_precision((torch.sigmoid(logits) > 0.5).float(), y.unsqueeze(1))
        recall = self.test_recall((torch.sigmoid(logits) > 0.5).float(), y.unsqueeze(1))

        



        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/precision', precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/recall', recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss




    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

        return {
           "optimizer": optimizer,
           "lr_scheduler": {
               "scheduler": scheduler,
               "monitor": "val_loss",
           },
        }
    

   

    

        
    







