# External Imports
import os
import sys
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping

# Internal Imports
import strategic_ml as sml
from .data_handle import get_data_set

# Constants
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(THIS_DIR, "logs/find_hyper_params")
VISUALIZATION_DIR = os.path.join(THIS_DIR, "/visualizations/")
DATA_DIR = os.path.join(THIS_DIR, "data")
DATA_NAME = "creditcard.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_NAME)
DATA_ROW_SIZE = 29
OUTPUT_DIR = os.path.join(THIS_DIR, "output")
COSTS = [0.1, 0.5, 1, 2, 10, float("inf")] 

class BCEWithLogitsLossPNOne(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLossPNOne, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        target = (target + 1) / 2
        input = (input + 1) / 2
        return self.loss(input, target)


class MSEPNOne(nn.Module):
    def __init__(self):
        super(MSEPNOne, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        target = (target + 1) / 2
        input = (input + 1) / 2
        return self.loss(input, target)


TRAIN_DATASET, VAL_DATASET, TEST_DATASET = get_data_set(
    data_path=DATA_PATH,
    seed=42,
    test_frac=0.2,
    val_frac_from_train=0.2,
    dtype=torch.float32,
)


def objective(trial: optuna.trial.Trial) -> float:
    """
    The objective function for the hyperparameter optimization using Optuna.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
        float: The computed loss.
    """

    # Define the hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 256)
    epochs = trial.suggest_int("epochs", 40, 140)
    early_stopping = trial.suggest_int("early_stopping", 10, 40)
    loss_fn = trial.suggest_categorical("loss_fn", ["hinge", "mse", "bce"])
    optimizer_str = trial.suggest_categorical("optimizer", ["adam", "sgd", "adagrad"])
    linear_reg_str = trial.suggest_categorical("linear_reg", ["none", "l1", "l2"])
    linear_reg_lambda = trial.suggest_float("linear_reg_lambda", 1e-5, 1e-1, log=True)

    # Load the data
    train_dataloader = DataLoader(
        TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = DataLoader(
        VAL_DATASET, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_dataloader = DataLoader(
        TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=0
    )

    if loss_fn == "hinge":
        loss_fn = nn.HingeEmbeddingLoss()
    elif loss_fn == "mse":
        loss_fn = MSEPNOne()
    elif loss_fn == "bce":
        loss_fn = BCEWithLogitsLossPNOne()
    else:
        raise ValueError(f"Invalid loss function {loss_fn}")
    # Define the model

    if optimizer_str == "adam":
        optimizer_class = torch.optim.Adam
    elif optimizer_str == "sgd":
        optimizer_class = torch.optim.SGD
    elif optimizer_str == "adagrad":
        optimizer_class = torch.optim.Adagrad
    else:
        raise ValueError(f"Invalid optimizer {optimizer_str}")

    model = sml.models.LinearModel(DATA_ROW_SIZE)
    if linear_reg_str == "none":
        linear_reg: Optional[List[sml._LinearRegularization]] = None
    elif linear_reg_str == "l1":
        linear_reg = [sml.LinearL1Regularization(lambda_=linear_reg_lambda)]
    elif linear_reg_str == "l2":
        linear_reg = [sml.LinearL2Regularization(lambda_=linear_reg_lambda)]
    else:
        raise ValueError(f"Invalid linear regularization {linear_reg_str}")
    cost_fn = sml.cost_functions.CostNormL2(dim=1)

    sum_loss = 0
    for cost_weight in COSTS:
        if cost_weight == float("inf"):
            delta = sml.gsc.IdentityDelta(strategic_model=model, cost=cost_fn)
        else:
            delta = sml.gsc.LinearStrategicDelta(
                strategic_model=model, cost=cost_fn, cost_weight=cost_weight
            )
        

        training_params = {
            "optimizer_class": optimizer_class,
            "optimizer_param": lr,
        }

        model_suit = sml.ModelSuit(
            model=model,
            delta=delta,
            loss_fn=loss_fn,
            train_loader=train_dataloader,
            validation_loader=val_dataloader,
            test_loader=test_dataloader,
            training_params=training_params,
            linear_regularization=linear_reg,
        )
        trail_number = trial.number
        early_stopping_callback = EarlyStopping(monitor="val_zero_one_loss_epoch", patience=early_stopping)
        trainer = pl.Trainer(
            logger=CSVLogger(save_dir=LOG_DIR, name=f"trial_{trail_number}"),
            max_epochs=epochs,
            enable_checkpointing=False,
            accelerator="auto",
            callbacks=[early_stopping_callback],
        )

        # Train the model
        trainer.fit(model_suit)
        out = trainer.test(model_suit)
        loss = out[0]["test_zero_one_loss_epoch"]
        if cost_weight == 0.1:
            loss = loss * 0.5

        sum_loss += loss

    return sum_loss 


def create_study():
    study = optuna.create_study(
        direction="minimize",
    )

    study.optimize(objective, n_trials=200, n_jobs=5)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    print("Task ID: ", task_id)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Trial number: ", trial.number)
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # write the results to file:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(
        os.path.join(
            OUTPUT_DIR,
            f"task_{task_id}_best_values.txt",
        ),
        "w",
    ) as f:
        f.write(f"Best trial: {trial.number}\n")
        f.write(f"Value: {trial.value}\n")
        f.write("Params:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")


if __name__ == "__main__":
    create_study()
