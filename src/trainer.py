import torch.nn as nn
from torch.optim import Adam, AdamW
import src.config_training as config_training
from torch.optim.lr_scheduler import OneCycleLR
from lightning.pytorch import LightningModule
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection


class WasteClassifier(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.train_loss_fn = nn.CrossEntropyLoss()
        self.val_loss_fn = nn.CrossEntropyLoss()
        self.test_loss_fn = nn.CrossEntropyLoss()

        self.test_metrics = MetricCollection(
            [
                Precision(average="weighted", num_classes=config_training.OUT_CHANNELS),
                Recall(average="weighted", num_classes=config_training.OUT_CHANNELS),
                F1Score(average="weighted", num_classes=config_training.OUT_CHANNELS),
            ]
        )
        self.val_acc_fn = Accuracy(average="weighted", num_classes=config_training.OUT_CHANNELS)

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch):
        images, labels = batch

        logits = self.model(images)
        loss = self.train_loss_fn(logits, labels)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch

        logits = self.model(images)
        loss = self.test_loss_fn(logits, labels)

        pred = nn.functional.softmax(logits, dim=-1)

        self.test_metrics.update(pred, labels, on_epoch=True)
        self.log("test_loss", loss)
        self.log_dict(self.test_metrics)
        

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self.model(images)
        pred = nn.functional.softmax(logits, dim=-1)
        
        loss = self.val_loss_fn(logits, labels)

        self.val_acc_fn.update(pred, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        val_acc = self.val_acc_fn.compute()
        self.log("val_acc", val_acc, prog_bar=True)
        self.val_acc_fn.reset()

    def configure_optimizers(self):
        param_lst = [
            {"params": self.model.backbone.parameters(), "lr": config_training.LR['BACKBONE']},
            {"params": self.model.classifier.parameters(), "lr": config_training.LR['CLASSIFIER']},
        ]

        if config_training.OPTIMIZER == 'Adam':
            optimizer = Adam(
                param_lst,
                lr=config_training.LR['CLASSIFIER'],
                eps=1e-6,
                weight_decay=0.01,
            )
        else:
            optimizer = AdamW(
                param_lst,
                lr=config_training.LR['CLASSIFIER'],
                eps=1e-6,
                weight_decay=0.01,
            )
        
        scheduler = OneCycleLR(optimizer=optimizer, max_lr = config_training.LR['CLASSIFIER'], total_steps=self.trainer.estimated_stepping_batches)
        
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
        }