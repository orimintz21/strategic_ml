# External imports
import os
from typing import Dict, Tuple
import unittest
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger


# Internal imports
from strategic_ml import (
    ModelSuit,
    LinearModel,
    LinearStrategicDelta,
    CostNormL2,
    LinearAdvDelta,
    IdentityDelta,
)

from .data_handle import load_data

# Constants
LOG_DIR = "tests/real_data_test/logs/"
VISUALIZATION_DIR = "tests/real_data_test/visualizations/"
DATA_DIR = "tests/real_data_test/data"
DATA_NAME = "creditcard.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_NAME)
DATA_ROW_SIZE = 29


def visualize_cost_weight_test(
    cost_weight_assumed_to_tested_to_loss: Dict[
        float, Dict[float, Tuple[float, float]]
    ],
    save_dir: str = VISUALIZATION_DIR,
):
    """
    This function visualize the results of the cost weight test.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    data = cost_weight_assumed_to_tested_to_loss
    real_weights = sorted(
        set(key for subdict in data.values() for key in subdict.keys())
    )
    # create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Increase spacing between groups of bars
    group_spacing = 1.0
    bar_width = 0.2
    index = np.arange(len(real_weights)) * (bar_width * len(data) + group_spacing)

    # Plotting the loss
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (assumed_w, real_dict) in enumerate(sorted(data.items())):
        losses = [
            real_dict[real_w][0] if real_w in real_dict else 0
            for real_w in real_weights
        ]
        ax.bar(
            index + i * bar_width,
            losses,
            bar_width,
            label=f"Assumed Weight: {'no movement' if assumed_w == float('inf') else assumed_w}",
        )

    ax.set_xlabel("Real Weight", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title("Loss vs Real Weight for Different Assumed Weights", fontsize=16)
    ax.set_xticks(index + bar_width * (len(data) - 1) / 2)
    ax.set_xticklabels(
        [
            "no movement" if real_w == float("inf") else real_w
            for real_w in real_weights
        ],
        fontsize=12,
    )
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    path = os.path.join(save_dir, "cost_weight_test_loss.png")
    plt.savefig(path)

    # Plotting the zero-one loss
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (assumed_w, real_dict) in enumerate(sorted(data.items())):
        zero_one_losses = [
            real_dict[real_w][1] if real_w in real_dict else 0
            for real_w in real_weights
        ]
        ax.bar(
            index + i * bar_width,
            zero_one_losses,
            bar_width,
            label=f"Assumed Weight: {'no movement' if assumed_w == float('inf') else assumed_w}",
        )

    ax.set_xlabel("Real Weight", fontsize=14)
    ax.set_ylabel("Zero-One Loss", fontsize=14)
    ax.set_title(
        "Zero-One Loss vs Real Weight for Different Assumed Weights", fontsize=16
    )
    ax.set_xticks(index + bar_width * (len(data) - 1) / 2)
    ax.set_xticklabels(
        [
            "no movement" if real_w == float("inf") else real_w
            for real_w in real_weights
        ],
        fontsize=12,
    )
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    path = os.path.join(save_dir, "cost_weight_test_zero_one_loss.png")
    plt.savefig(path)


class CreditCardTest(unittest.TestCase):
    def setUp(self):
        seed = 0
        test_frac = 0.2
        val_frac_from_train = 0.2
        batch_size_train = 64
        batch_size_val = 64
        batch_size_test = 64
        dtype = torch.float32
        self.train_loader, self.val_loader, self.test_loader = load_data(
            data_path=DATA_PATH,
            seed=seed,
            test_frac=test_frac,
            val_frac_from_train=val_frac_from_train,
            batch_size_train=batch_size_train,
            batch_size_val=batch_size_val,
            batch_size_test=batch_size_test,
            dtype=dtype,
            train_num_workers=9,
            val_num_workers=9,
            test_num_workers=9,
        )
        self.fast_dev_run = False

    def test_cost_weighs_strategic(self):
        """
        In this test we check what a linear model do when it assume a cost weight
        and it is tested with different cost weights.
        When we have a cost weight of infinity, the model should not move from the base model.
        """
        print("Test cost weights")
        TESTED_COST_WEIGHTS = [0.1, 0.5, 1.0, 2.0, 10.0, float("inf")]
        MAX_EPOCHS = 50
        model = LinearModel(in_features=DATA_ROW_SIZE)
        loss_fn = nn.BCEWithLogitsLoss()
        cost = CostNormL2(dim=1)
        training_params = {
            "optimizer": torch.optim.Adam,
            "lr": 0.01,
        }

        cost_weight_assumed_to_tested_to_loss: Dict[
            float, Dict[float, Tuple[float, float]]
        ] = {}

        for assumed_cost_weight in TESTED_COST_WEIGHTS:
            print(f"Assumed cost weight: {assumed_cost_weight}")
            model = LinearModel(in_features=DATA_ROW_SIZE)
            if assumed_cost_weight == float("inf"):
                delta = IdentityDelta(cost=cost, strategic_model=model)
            else:
                delta = LinearStrategicDelta(
                    strategic_model=model,
                    cost=cost,
                    cost_weight=assumed_cost_weight,
                )
            model_suit = ModelSuit(
                model=model,
                delta=delta,
                loss_fn=loss_fn,
                train_loader=self.train_loader,
                validation_loader=self.val_loader,
                test_loader=self.test_loader,
                training_params=training_params,
            )

            trainer = pl.Trainer(
                fast_dev_run=self.fast_dev_run,
                max_epochs=MAX_EPOCHS,
                logger=CSVLogger(
                    LOG_DIR,
                    name=f"credit_card_test_cost_weight_{assumed_cost_weight}_train",
                ),
            )
            trainer.fit(model_suit)
            cost_weight_assumed_to_tested_to_loss[assumed_cost_weight] = {}

            for test_cost_weight in TESTED_COST_WEIGHTS:
                print(f"Test cost weight: {test_cost_weight}")
                if test_cost_weight == float("inf"):
                    model_suit.test_delta = IdentityDelta(
                        cost=cost, strategic_model=model
                    )
                else:
                    model_suit.test_delta = LinearStrategicDelta(
                        strategic_model=model,
                        cost=cost,
                        cost_weight=test_cost_weight,
                    )

                trainer = pl.Trainer(
                    fast_dev_run=self.fast_dev_run,
                    max_epochs=MAX_EPOCHS,
                    logger=CSVLogger(
                        LOG_DIR,
                        name=f"credit_card_test_cost_weight_{assumed_cost_weight}_to_{test_cost_weight}",
                    ),
                )
                output = trainer.test(model_suit)

                cost_weight_assumed_to_tested_to_loss[assumed_cost_weight][
                    test_cost_weight
                ] = (
                    output[0]["test_loss_epoch"],
                    output[0]["test_zero_one_loss_epoch"],
                )

        visualize_cost_weight_test(
            cost_weight_assumed_to_tested_to_loss,
            save_dir=os.path.join(VISUALIZATION_DIR, "strategic_cost_weight_test"),
        )

    def test_cost_weighs_adv(self):
        """
        In this test we check what a linear model do when it assume a cost weight
        and it is tested with different cost weights.
        When we have a cost weight of infinity, the model should not move from the base model.
        """
        print("Test cost weights")
        TESTED_COST_WEIGHTS = [0.1, 0.5, 1.0, 2.0, 10.0, float("inf")]
        MAX_EPOCHS = 50
        model = LinearModel(in_features=DATA_ROW_SIZE)
        loss_fn = nn.BCEWithLogitsLoss()
        cost = CostNormL2(dim=1)
        training_params = {
            "optimizer": torch.optim.Adam,
            "lr": 0.01,
        }

        cost_weight_assumed_to_tested_to_loss: Dict[
            float, Dict[float, Tuple[float, float]]
        ] = {}

        for assumed_cost_weight in TESTED_COST_WEIGHTS:
            print(f"Assumed cost weight: {assumed_cost_weight}")
            model = LinearModel(in_features=DATA_ROW_SIZE)
            if assumed_cost_weight == float("inf"):
                delta = IdentityDelta(cost=cost, strategic_model=model)
            else:
                delta = LinearAdvDelta(
                    strategic_model=model,
                    cost=cost,
                    cost_weight=assumed_cost_weight,
                )
            model_suit = ModelSuit(
                model=model,
                delta=delta,
                loss_fn=loss_fn,
                train_loader=self.train_loader,
                validation_loader=self.val_loader,
                test_loader=self.test_loader,
                training_params=training_params,
            )

            trainer = pl.Trainer(
                fast_dev_run=self.fast_dev_run,
                max_epochs=MAX_EPOCHS,
                logger=CSVLogger(
                    LOG_DIR,
                    name=f"credit_card_test_cost_weight_{assumed_cost_weight}_train",
                ),
            )
            trainer.fit(model_suit)
            cost_weight_assumed_to_tested_to_loss[assumed_cost_weight] = {}

            for test_cost_weight in TESTED_COST_WEIGHTS:
                print(f"Test cost weight: {test_cost_weight}")
                if test_cost_weight == float("inf"):
                    model_suit.test_delta = IdentityDelta(
                        cost=cost, strategic_model=model
                    )
                else:
                    model_suit.test_delta = LinearAdvDelta(
                        strategic_model=model,
                        cost=cost,
                        cost_weight=test_cost_weight,
                    )

                trainer = pl.Trainer(
                    fast_dev_run=self.fast_dev_run,
                    max_epochs=MAX_EPOCHS,
                    logger=CSVLogger(
                        LOG_DIR,
                        name=f"credit_card_test_cost_weight_{assumed_cost_weight}_to_{test_cost_weight}",
                    ),
                )
                output = trainer.test(model_suit)

                cost_weight_assumed_to_tested_to_loss[assumed_cost_weight][
                    test_cost_weight
                ] = (
                    output[0]["test_loss_epoch"],
                    output[0]["test_zero_one_loss_epoch"],
                )

        visualize_cost_weight_test(
            cost_weight_assumed_to_tested_to_loss,
            save_dir=os.path.join(VISUALIZATION_DIR, "adv_cost_weight_test"),
        )


if __name__ == "__main__":
    unittest.main()
