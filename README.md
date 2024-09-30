# Strategic Machine Learning Library

Welcome to the **Strategic Machine Learning Library**! This library provides tools and frameworks for modeling, training, and evaluating machine learning models in strategic settings where agents may manipulate their features.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Key Concepts](#key-concepts)
- [Library Components](#library-components)
  - [Models](#models)
  - [Deltas (Strategic Behavior)](#deltas-strategic-behavior)
  - [Cost Functions](#cost-functions)
  - [Regularization Methods](#regularization-methods)
  - [Loss Functions](#loss-functions)
  - [Model suit](#model-suit)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [License](#license)

## Introduction

In many real-world scenarios, individuals (agents) may alter their observable features to receive better outcomes from predictive models (e.g., loan approvals, college admissions). This phenomenon is known as **strategic behavior**. This library also support an **advasieral user**, which is a user that tries to manipulate the model to give a wrong prediction.

The Strategic Machine Learning Library provides a comprehensive framework to model such strategic interactions between agents and predictive models. It allows you to:

- Build models that anticipate strategic behavior.
- Define and use various cost functions representing the effort or cost for agents to change their features.
- Implement strategic regularization techniques to encourage desirable outcomes.
- Train and evaluate models in settings where agents act strategically.

## Installation

To install the library, you can clone the repository and install the required packages:

```bash
git clone https://github.com/orimintz21/strategic_ml.git
cd strategic_ml
pip install -r requirements.txt
```

Alternatively, if the library is available on PyPI, you can install it via pip:

```bash
pip install strategic-ml
```

## Getting Started

Here's a simple example to get you started with training a strategic classification model:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
import strategic_ml as sml

# Define the model
model = sml.LinearModel(in_features=IN_FEATURES)

# Define the cost function
cost_fn = sml.CostNormL2(dim=1)

# Define the strategic delta
delta = sml.LinearStrategicDelta(strategic_model=model, cost=cost_fn, cost_weight=1.0)

# Define the loss function
loss_fn = sml.loss_functions.StrategicHingeLoss(model=model, delta=delta)

# Set up training parameters
training_params = {
    "optimizer_class": torch.optim.SGD,
    "optimizer_params": {"lr": 0.01},
}

# Initialize the ModelSuit
model_suit = sml.ModelSuit(
    model=model,
    delta=delta,
    loss_fn=loss_fn,
    train_loader=train_loader,
    validation_loader=validation_loader,
    test_loader=test_loader,
    training_params=training_params,
)

# Train the model
trainer = sml.Trainer(max_epochs=10)
trainer.fit(model_suit)
```

## Key Concepts

### Strategic Classification

**Strategic classification** considers scenarios where agents can manipulate their features to receive favorable predictions. Traditional machine learning models may become less effective if they do not account for such behavior.

### Modeling Strategic Behavior

The library models strategic behavior using **strategic deltas** (`delta`), which represent the changes agents make to their features. The cost of these changes is quantified using **cost functions**, and models can be regularized to promote or discourage certain strategic behaviors.

## Library Components

### Models

#### `LinearModel`

A simple linear model suitable for binary classification tasks:

```python
from strategic_ml.models import LinearModel

model = LinearModel(in_features=10)
```
#### Linear Regularizations
We also added regulation that are specific to the linear model, for example:

- **L1 Regularization** (`LinearL1Regularization`): Encourages sparsity in the model weights.
- **L2 Regularization** (`LinearL2Regularization`): Penalizes large weights to prevent overfitting.

```python
from strategic_ml.regularization import LinearL1Regularization, LinearL2Regularization

l1_reg = LinearL1Regularization(lambda_=0.01)
l2_reg = LinearL2Regularization(lambda_=0.01)
```
You are welcome to add more regularizations to the library.


### Deltas (Strategic Behavior)

Deltas represent the strategic modifications agents make to their features.
There are two basic type of delta, 'linear delta' and 'non-linear delta'. 
The 'linear delta' assumes that the model is linear and the cost is L2 norm.
By making those assumptions, the delta can be calculated by a closed form solution and therefore is faster to compute. If you can't make those assumptions, you can use the 'non-linear delta' which is more general and can be used with any model and cost function. This delta is calculated using gradient descent. 
There is also the IdentityDelta, which is used when the agents don't change their features (i.e., we are in a non-strategic scenario). We have implemented strategic and adversarial deltas for both linear and non-linear deltas.
You can add more deltas by inheriting from the appropriate class. Example:

#### `LinearStrategicDelta`

Assumes agents make linear adjustments to their features:

```python
from strategic_ml.gsc import LinearStrategicDelta

delta = LinearStrategicDelta(
    strategic_model=model,
    cost=cost_fn,
    cost_weight=1.0,
)
```

### Cost Functions

Cost functions quantify the effort or cost for agents to change their features.
For example:

#### `CostNormL2`

Computes the L2 norm (Euclidean distance) between the original and modified features:

```python
from strategic_ml.cost_functions import CostNormL2

cost_fn = CostNormL2(dim=1)
```


### Regularization Methods

Regularization methods encourage or discourage certain strategic behaviors.

#### Strategic Regularizations

- **Social Burden**: Minimizes the total effort required by positively labeled agents.
- **Recourse**: Measures the ability of negatively labeled agents to achieve positive outcomes.
- **Expected Utility**: Maximizes the expected utility of strategic agents.


### Loss Functions

Loss functions define the optimization objective during training.

#### `StrategicHingeLoss`

A modified hinge loss that accounts for strategic behavior:

```python
from strategic_ml.loss_functions import StrategicHingeLoss

loss_fn = StrategicHingeLoss(model=model, delta=delta)
```
## Model Suit

The `ModelSuit` class serves as a comprehensive module that encapsulates the model, strategic behavior (delta), loss functions, regularization methods, and training configurations. It streamlines the training, validation, and testing processes by integrating all necessary components and it is implemented using lightning torch.

### **Why Use ModelSuit?**

- **Simplifies Training Workflow**: By encapsulating all components, `ModelSuit` reduces boilerplate code and potential errors.
- **Strategic Integration**: Automatically handles strategic behavior during training and evaluation.
- **Flexibility**: Supports various models, deltas, cost functions, and regularization methods.
- **Leverages PyTorch Lightning**: Benefits from advanced features like automatic batching, GPU acceleration, and logging.

### **Usage**

```python
from strategic_ml import (
        LinearModel,
        LinearStrategicDelta,
        CostNormL2,
        StrategicHingeLoss,
        LinearL1Regularization,
        ModelSuit
)

# Define the components
model = LinearModel(in_features=10)
cost_fn = CostNormL2(dim=1)
delta = LinearStrategicDelta(strategic_model=model, cost=cost_fn, cost_weight=1.0)
test_delta = LinearStrategicDelta(strategic_model=model, cost=cost_fn, cost_weight=1.5)
loss_fn = StrategicHingeLoss(model=model, delta=delta)
l1_reg = LinearL1Regularization(lambda_=0.01)

training_params = {
    "optimizer_class": optim.SGD,
    "optimizer_params": {"lr": 0.01},
}

# Initialize the ModelSuit
model_suit = ModelSuit(
    model=model,
    delta=delta,
    test_delta=test_delta,
    loss_fn=loss_fn,
    train_loader=train_loader,
    validation_loader=val_loader,
    test_loader=test_loader,
    training_params=training_params,
    linear_regularization=[l1_reg],
)

# Train the model using PyTorch Lightning's Trainer
from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=10)
trainer.fit(model_suit)
output = trainer.test(model_suit)
```

### **Key Parameters**

- **model**: The predictive model (e.g., `LinearModel`).
- **delta**: The strategic delta modeling agents' behavior.
- **loss_fn**: The loss function used for optimization.
- **regularization**: (Optional) Strategic regularization methods.
- **linear_regularization**: (Optional) Linear regularization methods (e.g., L1, L2).
- **training_params**: Dictionary containing optimizer and scheduler configurations.

### **Advanced Features**

- **Device Compatibility**: Works seamlessly on CPU and GPU.
- **Logging and Metrics**: Integrates with PyTorch Lightning's logging for tracking training progress.
- **Strategic Evaluation**: Automatically accounts for strategic behavior during validation and testing.
## Hyperparameter Optimization

You can use **Optuna** for hyperparameter optimization to find the best settings for your model.

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer

def objective(trial):
    # Define hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 256)
    epochs = trial.suggest_int("epochs", 10, 100)
    
    # Set up data loaders with the suggested batch size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Define the model, loss function, and optimizer with suggested parameters
    model = LinearModel(in_features=10)
    loss_fn = StrategicHingeLoss(model=model, delta=delta)
    training_params = {
        "optimizer_class": optim.SGD,
        "optimizer_params": {"lr": lr},
    }
    
    # Initialize ModelSuit
    model_suit = ModelSuit(
        model=model,
        delta=delta,
        loss_fn=loss_fn,
        train_loader=train_loader,
        validation_loader=val_loader,
        test_loader=test_loader,
        training_params=training_params,
    )
    
    # Set up the trainer with Optuna's pruning callback
    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss_epoch")],
    )
    
    # Train the model
    trainer.fit(model_suit)
    
    # Return the validation loss
    return trainer.callback_metrics["val_loss_epoch"].item()

# Create an Optuna study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for using the Strategic Machine Learning Library! We hope it aids in your research and applications.
