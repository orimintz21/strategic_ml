import torch
import lightning.pytorch as pl
from torch import nn
from lightning.pytorch.callbacks import Callback
from torch.optim import Adam

class StrategicClassificationModule(pl.LightningModule):
    def __init__(self, model, strategic_regularization, loss_fn, gsc, lr=0.001):
        super(StrategicClassificationModule, self).__init__()
        self.model = model
        self.strategic_regularization = strategic_regularization
        self.loss_fn = loss_fn
        self.gsc = gsc
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        assert inputs.ndim == 2, f"Expected inputs to be 2D (batch_size, n_features), got shape {inputs.shape}"
        assert inputs.shape[1] == self.model.model.in_features, (
            f"Expected input features {inputs.shape[1]} to match model's input features {self.model.model.in_features}"
        )
        assert targets.ndim == 2 and targets.shape[1] == 1, (
            f"Expected targets to be 2D with shape (batch_size, 1), got shape {targets.shape}"
        )

        x_prime = self.gsc(inputs, targets)  # Compute strategic delta
        assert x_prime.shape == inputs.shape, (
            f"Expected x_prime to have shape {inputs.shape}, but got {x_prime.shape}"
        )

        outputs = self.model(x_prime)
        assert outputs.shape == targets.shape, (
            f"Expected model outputs to have shape {targets.shape}, but got {outputs.shape}"
        )
        x_prime = self.gsc(inputs, targets)  # Compute strategic delta
        outputs = self.model(x_prime)
        loss = self.loss_fn(outputs, targets)
        
        if self.strategic_regularization:
            reg_loss = self.strategic_regularization(inputs, x_prime, targets, outputs)
            loss += reg_loss
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

class StrategicAdjustmentCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Example: Adjust the cost weight or other parameters at the end of each epoch
        new_cost_weight = pl_module.gsc.get_cost_weight() * 0.9
        pl_module.gsc.set_cost_weight(new_cost_weight)
        print(f"Adjusted cost weight to {new_cost_weight}")