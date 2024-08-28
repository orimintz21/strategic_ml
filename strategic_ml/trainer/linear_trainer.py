import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any

from strategic_ml.models.linear_strategic_model import LinearStrategicModel
from strategic_ml.strategic_regularization.strategic_regularization import _StrategicRegularization
from strategic_ml.loss_functions.loss import _Loss
from strategic_ml.loss_functions.stratigic_hinge_loss import StrategicHingeLoss
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.linear_gp import _LinearGP

class LinearTrainer:
    """
        A class that represents a basic trainer for a strategic machine learning model.

        Args:
            model (LinearStrategicModel): The strategic machine learning model to be trained.
            strategic_regularization (_StrategicRegularization): The strategic regularization method to be used.
            loss (_Loss): The loss function to be used during training.
            cost (_CostFunction): The cost function to be used during training.
            gsc (_LinearGP): The gradient similarity constraint to be applied during training.
            device (str): The device to be used for training. Defaults to "cpu".
            training_params (dict): Additional parameters for training. Defaults to None.
                The `training_params` dictionary should have the following structure:
                {
                    'optimizer': None,  # The optimizer to be used for training
                    'learning_rate': 0.001,  # The learning rate for the optimizer
                    'scheduler': None,  # The learning rate scheduler
                    'optimizer_params': {},  # Additional parameters for the optimizer
                    'scheduler_params': {},  # Additional parameters for the scheduler
                    'early_stopping': {
                        'patience': 10,  # The number of epochs with no improvement after which training will be stopped
                        'min_delta': 0.001,  # The minimum change in the monitored quantity to qualify as an improvement
                        'monitor': 'val_loss',  # The quantity to be monitored for early stopping
                        'restore_best_weights': True  # Whether to restore the weights of the best model found during training
                    }
                }
        """

    def __init__(self,
        model: LinearStrategicModel,
        strategic_regularization: _StrategicRegularization,
        loss: _Loss,
        cost: _CostFunction,
        gsc: _LinearGP,
        device: str = "cpu",
        *args: Any,
        training_params: Dict[str, Any],
    ) -> None:
        self.model: LinearStrategicModel = model
        self.strategic_regularization: _StrategicRegularization = strategic_regularization
        self.loss: _Loss = loss
        self.cost: _CostFunction = cost
        self.gsc: _LinearGP = gsc
        self.device: str = device

        # Store the training parameters
        self.training_params: Dict[str, Any] = training_params
        
        # Create the optimizer with the specified learning rate and parameters
        self.optimizer = self.training_params['optimizer'](
            self.model.parameters(), 
            lr=self.training_params['learning_rate'], 
            **self.training_params['optimizer_params']
        )
        
        # Check if a learning rate scheduler is provided
        if self.training_params['scheduler']:
            self.scheduler = self.training_params['scheduler'](
                self.optimizer, 
                **self.training_params['scheduler_params']
            )
        else:
            self.scheduler = None

        # Early stopping parameters
        self.early_stopping_params = self.training_params.get('early_stopping', {})

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10) -> None:
        for epoch in range(epochs):
            self.train(train_loader)
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                self._check_early_stopping(val_metrics)
    
    def train(self, train_loader: DataLoader) -> None:
        self.model.train()
        for batch in train_loader:
            self.train_batch(batch)
        if self.scheduler:
            self.scheduler.step()

    def train_batch(self, batch: Any) -> None:
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        

        # Compute the strategic delta
        x_prime = self.gsc(inputs, targets)

        self.optimizer.zero_grad()
        outputs = self.model(x_prime)
        loss = self.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def evaluate(self, val_loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                all_outputs.append(outputs)
                all_targets.append(targets)

        # Concatenate all outputs and targets into single tensors
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute and return metrics
        metrics = self._compute_metrics(all_outputs, all_targets)
        return metrics
    
    def _compute_metrics(self, outputs: Any, targets: Any) -> Dict[str, Any]:
        loss = self.loss(outputs, targets).item()
        correct = (outputs.round() == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total

        return {
            "loss": loss,
            "accuracy": accuracy
        }   

    def _check_early_stopping(self, metrics: Dict[str, Any]) -> None:
        # Implement early stopping logic based on validation metrics
        pass
