from lightning.pytorch.callbacks import Callback

class StrategicAdjustmentCallback(Callback):
    def __init__(self, adjustment_factor=0.9, min_cost_weight=0.1):
        """
        Initializes the StrategicAdjustmentCallback.

        Args:
            adjustment_factor (float): Factor by which to multiply the cost weight at the end of each epoch.
            min_cost_weight (float): The minimum cost weight value allowed after adjustment.
        """
        super().__init__()
        self.adjustment_factor = adjustment_factor
        self.min_cost_weight = min_cost_weight

    def on_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each epoch. Adjusts the cost weight of the GSC component.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The current LightningModule being trained.
        """
        # Get the current cost weight from the GSC component
        current_cost_weight = pl_module.gsc.get_cost_weight()
        
        # Adjust the cost weight
        new_cost_weight = max(self.min_cost_weight, current_cost_weight * self.adjustment_factor)
        pl_module.gsc.set_cost_weight(new_cost_weight)
        
        # Log or print the adjustment
        trainer.logger.log_metrics({"cost_weight": new_cost_weight}, step=trainer.current_epoch)
        print(f"Epoch {trainer.current_epoch + 1}: Adjusted cost weight to {new_cost_weight:.4f}")
