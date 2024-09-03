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
    """

    def __init__(self,
        model: LinearStrategicModel,
        strategic_regularization: _StrategicRegularization,
        loss: _Loss,
        cost: _CostFunction,
        gsc: _LinearGP,
        device: str = "cpu",
        training_params: Dict[str, Any] = None,
        *args: Any,
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
        self.early_stopping_counter = 0
        self.best_metric = None
        self.best_weights = None
        

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10, checkpoints: Optional[str] = None) -> None:
        for epoch in range(epochs):
            # Training step
            self.train(train_loader)
            
            # Validation step
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                print(f"Epoch {epoch+1}/{epochs} - Validation {self.early_stopping_params.get('monitor', 'val_loss')}: {val_metrics[self.early_stopping_params.get('monitor', 'val_loss')]:.4f}")

                # Early stopping check
                if self._check_early_stopping(val_metrics):
                    print(f"Early stopping at epoch {epoch + 1}")
                    if self.early_stopping_params.get('restore_best_weights', True) and self.best_weights is not None:
                        self.model.load_state_dict(self.best_weights)
                    break

            if checkpoints is not None and self.best_weights is not None:
                torch.save(self.model.state_dict(), f"{checkpoints}.pt")
                print(f"*** Saved checkpoint {checkpoints} at epoch {epoch+1}")

        print("Training completed.")
    
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

        # Compute the primary loss
        loss = self.loss(outputs, targets)
        
        # Apply strategic regularization if defined
        if self.strategic_regularization:
            reg_loss = self.strategic_regularization(inputs, x_prime, targets, outputs)
            loss += reg_loss
        
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

    def _check_early_stopping(self, metrics: Dict[str, Any]) -> bool:
        """
        Checks if early stopping should be triggered based on the specified monitoring metric.

        This method compares the current value of the monitored metric to the best value observed so far. 
        If the metric has not improved by at least `min_delta` for a number of epochs specified by `patience`, 
        early stopping is triggered.

        Args:
            metrics (Dict[str, Any]): A dictionary containing the evaluation metrics for the current epoch. 
                                    The key should correspond to the metric being monitored (e.g., 'val_loss').

        Returns:
            bool: True if early stopping should be triggered, False otherwise.

        Early Stopping Parameters:
            - patience (int): Number of epochs with no improvement after which training will be stopped. 
                            Default is 10 if not specified.
            - min_delta (float): Minimum change in the monitored metric to qualify as an improvement. 
                                Default is 0.001 if not specified.
            - monitor (str): The metric name to monitor (e.g., 'val_loss', 'val_accuracy'). 
                            This metric should be a key in the `metrics` dictionary. Default is 'val_loss' if not specified.
            - restore_best_weights (bool): Whether to restore the model weights from the epoch with the best 
                                        monitored metric when early stopping is triggered. Default is True.

        Example:
            early_stopping_params = {
                'patience': 5,
                'min_delta': 0.001,
                'monitor': 'val_loss',
                'restore_best_weights': True
            }

        This method will monitor 'val_loss' and trigger early stopping if it does not improve 
        by at least 0.001 for 5 consecutive epochs.
        """

        current_metric = metrics[self.early_stopping_params.get('monitor', 'val_loss')]

        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_weights = self.model.state_dict()
            return False

        # Check if there is improvement greater than min_delta
        if (current_metric - self.best_metric) < -self.early_stopping_params.get('min_delta', 0.001):
            self.best_metric = current_metric
            self.best_weights = self.model.state_dict()
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        # Check if early stopping should be triggered
        if self.early_stopping_counter >= self.early_stopping_params.get('patience', 10):
            return True
        
        return False