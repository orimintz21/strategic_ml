
# External Imports
import os
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import optuna
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping

# Internal Imports
import strategic_ml as sml
from .data_handle import get_data_set

# Constants
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(THIS_DIR, "logs/find_hyper_params_non_linear")
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

    batch_size = 126
    epochs = 75
    model_optimizer = torch.optim.SGD
    lr =0.017
    linear_reg = sml.LinearL2Regularization(lambda_=0.03)
    loss_fn = BCEWithLogitsLossPNOne()
    

    # Define the hyperparameters to optimize
    delta_lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    delta_optimizer_str = trial.suggest_categorical("optimizer", ["adam", "sgd", "adagrad"])
    temp = trial.suggest_float("temp", 0.1, 50)
    num_delta_epochs = trial.suggest_int("num_delta_epochs", 40, 100)

    model_suit_training_params = {
        "optimizer_class": model_optimizer,
        "optimizer_params": {"lr": lr},
    }


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

    if delta_optimizer_str == "adam":
        optimizer_class = torch.optim.Adam
    elif delta_optimizer_str == "sgd":
        optimizer_class = torch.optim.SGD
    elif delta_optimizer_str == "adagrad":
        optimizer_class = torch.optim.Adagrad
    else:
        raise ValueError(f"Invalid optimizer {delta_optimizer_str}")
    
    delta_training_params = {
        "optimizer_class": optimizer_class,
        "optimizer_params": {"lr": delta_lr},
        "num_epochs": num_delta_epochs,
        "temp": temp,
    }

    linear_model_linear_delta = sml.models.LinearModel(
        DATA_ROW_SIZE
    )
    linear_model_non_linear_delta = sml.models.LinearModel(
        DATA_ROW_SIZE
    )
    cost = sml.CostNormL2(dim=1)

    linear_delta = sml.LinearStrategicDelta(cost=cost, strategic_model=linear_model_linear_delta)

    save_dir = os.path.join(LOG_DIR, f"trial_{trial.number}")
    non_linear_delta = sml.NonLinearStrategicDelta(cost=cost, strategic_model=linear_model_non_linear_delta, save_dir=save_dir, training_params=delta_training_params)

    model_suit_linear = sml.ModelSuit(
        model=linear_model_linear_delta,
        delta=linear_delta,
        loss_fn=loss_fn,
        train_loader=train_dataloader,
        validation_loader=val_dataloader,
        test_loader=test_dataloader,
        training_params=model_suit_training_params,
        linear_regularization=[linear_reg]
    )

    trainer_save_dir = os.path.join(LOG_DIR, f"linear_csv_{trial.number}")
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=CSVLogger(trainer_save_dir),
    )
    
    trainer.fit(model_suit_linear)
    output = trainer.test(model_suit_linear)
    linear_output = output[0]["test_loss_epoch"]

    model_suit_non_linear = sml.ModelSuit(
        model=linear_model_non_linear_delta,
        delta=non_linear_delta,
        loss_fn=loss_fn,
        train_loader=train_dataloader,
        validation_loader=val_dataloader,
        test_loader=test_dataloader,
        training_params=model_suit_training_params,
        linear_regularization=[linear_reg]
    )

    trainer_save_dir = os.path.join(LOG_DIR, f"non_linear_csv_{trial.number}")
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=CSVLogger(trainer_save_dir),
    )
    trainer.fit(model_suit_non_linear)
    model_suit_non_linear.train_delta_for_test()
    output = trainer.test(model_suit_non_linear)
    non_linear_output = output[0]["test_loss_epoch"]

    return abs(linear_output - non_linear_output)



def create_study():
    study = optuna.create_study(
        direction="minimize",
    )

    study.optimize(objective, n_trials=300, n_jobs=5)
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
            f"task_{task_id}_find_delta_best_values.txt",
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
