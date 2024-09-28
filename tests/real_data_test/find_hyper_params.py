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

# Internal Imports
import strategic_ml as sml
from .data_handle import get_data_set

# Constants
global COST_WEIGHT, COST_WEIGHT_TEST
COST_WEIGHT = 1.0
COST_WEIGHT_TEST = 1.0
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(THIS_DIR, "logs/find_hyper_params")
VISUALIZATION_DIR = os.path.join(THIS_DIR, "/visualizations/")
DATA_DIR = os.path.join(THIS_DIR, "data")
DATA_NAME = "creditcard.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_NAME)
DATA_ROW_SIZE = 29
OUTPUT_DIR = os.path.join(THIS_DIR, "output")


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


class CustomPruningCallback(Callback):
    def __init__(self, trial, monitor):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        self.trial.report(current_score, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()


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

    # Define the model
    model = sml.models.LinearModel(DATA_ROW_SIZE)
    cost_fn = sml.cost_functions.CostNormL2(dim=1)
    if COST_WEIGHT == float("inf"):
        delta = sml.gsc.IdentityDelta(strategic_model=model, cost=cost_fn)
    else:
        delta = sml.gsc.LinearStrategicDelta(
            strategic_model=model, cost=cost_fn, cost_weight=COST_WEIGHT
        )

    if COST_WEIGHT_TEST == float("inf"):
        test_delta = sml.gsc.IdentityDelta(strategic_model=model, cost=cost_fn)
    else:
        test_delta = sml.gsc.LinearStrategicDelta(
            strategic_model=model, cost=cost_fn, cost_weight=COST_WEIGHT_TEST
        )

    if loss_fn == "hinge":
        loss_fn = nn.HingeEmbeddingLoss()
    # elif loss_fn == "strategic_hinge":
    #     loss_fn = sml.loss_functions.StrategicHingeLoss(model=model, delta=delta)
    elif loss_fn == "mse":
        loss_fn = MSEPNOne()
    elif loss_fn == "bce":
        loss_fn = BCEWithLogitsLossPNOne()
    else:
        raise ValueError(f"Invalid loss function {loss_fn}")

    if optimizer_str == "adam":
        optimizer_class = torch.optim.Adam
    elif optimizer_str == "sgd":
        optimizer_class = torch.optim.SGD
    elif optimizer_str == "adagrad":
        optimizer_class = torch.optim.Adagrad
    else:
        raise ValueError(f"Invalid optimizer {optimizer_str}")

    if linear_reg_str == "none":
        linear_reg: Optional[List[sml._LinearRegularization]] = None
    elif linear_reg_str == "l1":
        linear_reg = [sml.LinearL1Regularization(lambda_=linear_reg_lambda)]
    elif linear_reg_str == "l2":
        linear_reg = [sml.LinearL2Regularization(lambda_=linear_reg_lambda)]
    else:
        raise ValueError(f"Invalid linear regularization {linear_reg_str}")

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
        test_delta=test_delta,
        training_params=training_params,
        linear_regularization=linear_reg,
    )
    trail_number = trial.number
    callback = CustomPruningCallback(trial, monitor="val_zero_one_loss")
    trainer = pl.Trainer(
        logger=CSVLogger(save_dir=LOG_DIR, name=f"trial_{trail_number}"),
        max_epochs=epochs,
        enable_checkpointing=False,
        accelerator="auto",
        # callbacks=[callback],
    )
    hyper_params = {
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "loss_fn": loss_fn,
        "optimizer": optimizer_str,
        "linear_reg": linear_reg_str,
        "linear_reg_lambda": linear_reg_lambda,
    }
    # Train the model
    trainer.fit(model_suit)
    out = trainer.test(model_suit)
    return out[0]["test_zero_one_loss_epoch"]


def create_study():
    study = optuna.create_study(
        direction="minimize",
    )

    study.optimize(objective, n_trials=150, n_jobs=1)
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
            f"task_{task_id}_cost_{COST_WEIGHT}_assume_{COST_WEIGHT_TEST}.txt",
        ),
        "w",
    ) as f:
        f.write(f"COST_WEIGHT: {COST_WEIGHT}\n")
        f.write(f"COST_WEIGHT_TEST: {COST_WEIGHT_TEST}\n")
        f.write(f"Best trial: {trial.number}\n")
        f.write(f"Value: {trial.value}\n")
        f.write("Params:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    assert len(args) <= 2, "Too many arguments"
    if len(args) > 0:
        # This is a global variable
        assert float(args[0]) >= 0, "Cost weight must be non-negative"
        COST_WEIGHT = float(args[0])
    else:
        COST_WEIGHT = 1.0

    if len(args) > 1:
        COST_WEIGHT_TEST = float(args[1])
    else:
        COST_WEIGHT_TEST = COST_WEIGHT

    create_study()
