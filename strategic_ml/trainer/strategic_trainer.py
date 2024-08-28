import lightning.pytorch as pl
from strategic_ml.trainer.strategic_classification_module import StrategicClassificationModule
from strategic_ml.trainer.strategic_callbacks import StrategicAdjustmentCallback
import torch

def create_trainer(model, strategic_regularization, loss_fn, gsc, training_params, callbacks=None):
    """
    Creates and configures a PyTorch Lightning Trainer instance.

    Args:
        model: The model to be trained (e.g., a PyTorch `nn.Module`).
        strategic_regularization: An instance of the strategic regularization to apply.
        loss_fn: The loss function to use during training.
        gsc: The generalized strategic delta function or module.
        training_params (dict): A dictionary of training parameters including:
            - 'lr': Learning rate.
            - 'max_epochs': Maximum number of epochs.
            - 'devices': Number of devices to use (0 for CPU).
            - 'precision': Precision to use (e.g., 16 for mixed precision).
            - 'accelerator': Accelerator type ('cpu', etc.).
            - 'logger': Logger instance (optional).
            - 'early_stopping': Early stopping configuration (optional).
            - 'checkpoint_callback': Checkpoint callback configuration (optional).
        callbacks (list): Additional custom callbacks to use during training (optional).

    Returns:
        pl.Trainer: A configured PyTorch Lightning Trainer instance.
        StrategicClassificationModule: The Lightning module with the model and configurations.
    """

    # Extract training parameters from the provided dictionary
    lr = training_params.get('lr', 0.001)
    max_epochs = training_params.get('max_epochs', 20)
    devices = training_params.get('devices', 1)  # Ensure it's set to 0 for CPU
    precision = training_params.get('precision', 32)
    accelerator = training_params.get('accelerator', 'cpu')  # Set accelerator to 'cpu'
    logger = training_params.get('logger', None)
    early_stopping_config = training_params.get('early_stopping', None)
    checkpoint_callback_config = training_params.get('checkpoint_callback', None)

    # Create the LightningModule instance
    strategic_module = StrategicClassificationModule(
        model=model,
        strategic_regularization=strategic_regularization,
        loss_fn=loss_fn,
        gsc=gsc,
        lr=lr
    )

    # Prepare callbacks
    callbacks = callbacks or []
    if early_stopping_config:
        early_stopping = pl.callbacks.EarlyStopping(**early_stopping_config)
        callbacks.append(early_stopping)
    
    if checkpoint_callback_config:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(**checkpoint_callback_config)
        callbacks.append(checkpoint_callback)

    # Optionally add more custom callbacks
    callbacks.append(StrategicAdjustmentCallback())  # Example of adding the strategic adjustment callback

    # Create the Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,  # Set devices to 0 for CPU
        precision=precision,
        accelerator=accelerator,  # Set accelerator to 'cpu'
        logger=logger,
        callbacks=callbacks
    )

    return trainer, strategic_module
