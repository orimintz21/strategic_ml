import numpy as np

class _Loss:
    def __init__(self, delta_h: Callable[[np.ndarray], np.ndarray], 
                    regularization_lambda: float = 0.01, 
                    model: Optional[np.ndarray] = None):
        """
        Initialize the base loss class.
        
        :param delta: Function to modify features strategically.
        :param regularization_lambda: Regularization parameter.
        :param model: Initial model parameters.
        """
        raise NotImplementedError()


    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the loss for given inputs and labels.
        
        :param X: Input features.
        :param y: True labels.
        :return: Loss value.
        """
 
        raise NotImplementedError()


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass.
        
        :param X: Input features.
        :return: Predicted scores.
        """
        if self.w is None:
            raise ValueError("Model is not initialized. Please set the model weights.")
        return np.dot(X, self.w)


    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss function.
        
        :param X: Input features.
        :param y: True labels.
        :return: Gradient value.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    

    @property
    def delta(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._delta

    @delta.setter
    def delta(self, value: Callable[[np.ndarray], np.ndarray]) -> None:
        self._delta = value

    @property
    def regularization_lambda(self) -> float:
        return self._regularization_lambda

    @regularization_lambda.setter
    def regularization_lambda(self, value: float) -> None:
        self._regularization_lambda = value

    @property
    def model(self) -> Optional[np.ndarray]:
        return self._w

    @model.setter
    def model(self, value: Optional[np.ndarray]) -> None:
        self._w = value

    