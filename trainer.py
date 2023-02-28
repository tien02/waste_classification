import config
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection
from pytorch_lightning import LightningModule

class WasteClassifier(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_metrics = MetricCollection(
            [
                Precision(average="weighted", num_classes=config.OUT_CHANNELS),
                Recall(average="weighted", num_classes=config.OUT_CHANNELS),
                F1Score(average="weighted", num_classes=config.OUT_CHANNELS),
            ]
        )
        self.val_acc_fn = Accuracy(average="weighted", num_classes=config.OUT_CHANNELS)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        images, labels = batch

        logits = self.model(images)
        loss = self.loss_fn(logits, labels)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch

        logits = self.model(images)
        loss = self.loss_fn(logits, labels)

        pred = nn.functional.softmax(logits, dim=1)
        pred = torch.argmax(pred, dim=1)

        metrics = self.test_metrics(pred, labels)
        self.log("test_loss", loss)
        self.log_dict(metrics)

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self.model(images)
        pred = nn.functional.softmax(logits, dim=1)
        
        loss = self.loss_fn(logits, labels)

        self.val_acc_fn.update(pred, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        val_acc = self.val_acc_fn.compute()
        self.log("val_acc", val_acc, prog_bar=True)
        self.val_acc_fn.reset()

    def configure_optimizers(self):
        if config.OPTIMIZER == 'Adam':
            optimizer = Adam(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                eps=1e-6,
                weight_decay=0.01,
            )
        else:
            optimizer = AdamW(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                eps=1e-6,
                weight_decay=0.01,
            )
        return {
            "optimizer": optimizer,
        }
