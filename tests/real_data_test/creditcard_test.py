# External imports
import os
from typing import Dict, Tuple, Union
import unittest
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping


# Internal imports
from strategic_ml import (
    ModelSuit,
    LinearModel,
    LinearStrategicDelta,
    CostNormL2,
    LinearL1Regularization,
    NonLinearStrategicDelta,
    LinearAdvDelta,
    StrategicHingeLoss,
    IdentityDelta,
    SocialBurden,
    ExpectedUtility,
    Recourse,
)

from .data_handle import load_data
from .visualization import (
    visualize_cost_weight_test,
    visualize_reg_weight_test,
    visualize_full_connected_2_layers,
    visualize_loss_test,
)

# Constants
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(THIS_DIR, "logs/")
DELTA_LOG_DIR = os.path.join(THIS_DIR, "delta_logs/")
VISUALIZATION_DIR = os.path.join(THIS_DIR, "visualizations/")
DATA_DIR = os.path.join(THIS_DIR, "data/")
DATA_NAME = "creditcard.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_NAME)
DATA_ROW_SIZE = 29


class FullyConnected2Layers(nn.Module):
    def __init__(self, in_features: int, hidden_size: int = 100):
        super(FullyConnected2Layers, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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


class CreditCardTest(unittest.TestCase):
    def setUp(self):
        seed = 0
        test_frac = 0.2
        val_frac_from_train = 0.2
        batch_size_train = 34
        batch_size_val = 34
        batch_size_test = 34
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
            train_num_workers=1,
            val_num_workers=1,
            test_num_workers=1,
        )
        self.fast_dev_run = False

    def test_loss_fn(self):
        print("Test loss functions")
        MAX_EPOCHS = 130
        output_dict: Dict[str, Tuple[float, float]] = {}
        loss_fn_names = ["bce", "mse", "hinge", "strategic_hinge"]
        for loss_fn_name in loss_fn_names:
            model = LinearModel(in_features=DATA_ROW_SIZE)
            delta = LinearStrategicDelta(
                strategic_model=model,
                cost=CostNormL2(dim=1),
                cost_weight=1.0,
            )
            if loss_fn_name == "bce":
                loss_fn = BCEWithLogitsLossPNOne()
            elif loss_fn_name == "mse":
                loss_fn = MSEPNOne()
            elif loss_fn_name == "hinge":
                loss_fn = nn.HingeEmbeddingLoss()
            elif loss_fn_name == "strategic_hinge":
                loss_fn = StrategicHingeLoss(model, delta)
            else:
                raise ValueError(f"Unknown loss function: {loss_fn_name}")
            linear_reg = LinearL1Regularization(0.05)

            model_suit = ModelSuit(
                model=model,
                delta=delta,
                loss_fn=loss_fn,
                train_loader=self.train_loader,
                validation_loader=self.val_loader,
                test_loader=self.test_loader,
                training_params={
                    "optimizer": torch.optim.SGD,
                    "lr": 0.0007,
                },
                linear_regularization=[linear_reg],
            )
            early_stopping_callback = EarlyStopping(
                monitor="val_zero_one_loss_epoch", patience=33
            )
            trainer = pl.Trainer(
                fast_dev_run=self.fast_dev_run,
                max_epochs=MAX_EPOCHS,
                logger=CSVLogger(
                    LOG_DIR,
                    name=f"credit_card_test_loss_fn_{loss_fn_name}",
                ),
                callbacks=[early_stopping_callback],
            )
            trainer.fit(model_suit)
            output = trainer.test(model_suit)
            output_dict[loss_fn_name] = (
                output[0]["test_loss_epoch"],
                output[0]["test_zero_one_loss_epoch"],
            )
        visualize_loss_test(
            output_dict,
            save_dir=os.path.join(VISUALIZATION_DIR, "loss_fn_test"),
        )

    def test_cost_weighs_strategic(self):
        """
        In this test we check what a linear model do when it assume a cost weight
        and it is tested with different cost weights.
        When we have a cost weight of infinity, the model should not move from the base model.
        """
        print("Test cost weights")
        TESTED_COST_WEIGHTS = [0.1, 0.5, 1.0, 2.0, 10.0, float("inf")]
        MAX_EPOCHS = 116
        model = LinearModel(in_features=DATA_ROW_SIZE)
        loss_fn = BCEWithLogitsLossPNOne()
        cost = CostNormL2(dim=1)
        l2_reg = LinearL1Regularization(0.011)
        training_params = {
            "optimizer": torch.optim.Adam,
            "lr": 0.00018,
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
                linear_regularization=[l2_reg],
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
                print(
                    f"Assumed cost weight: {assumed_cost_weight}Test cost weight: {test_cost_weight}"
                )
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
                mean_loss = np.mean(
                    [output[i]["test_loss_epoch"] for i in range(len(output))]
                ).item()
                mean_zero_one_loss = np.mean(
                    [output[i]["test_zero_one_loss_epoch"] for i in range(len(output))]
                ).item()

                cost_weight_assumed_to_tested_to_loss[assumed_cost_weight][
                    test_cost_weight
                ] = (mean_loss, mean_zero_one_loss)

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
        loss_fn = MSEPNOne()
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
                mean_loss = np.mean(
                    [output[i]["test_loss_epoch"] for i in range(len(output))]
                ).item()
                mean_zero_one_loss = np.mean(
                    [output[i]["test_zero_one_loss_epoch"] for i in range(len(output))]
                ).item()

                cost_weight_assumed_to_tested_to_loss[assumed_cost_weight][
                    test_cost_weight
                ] = (mean_loss, mean_zero_one_loss)

        visualize_cost_weight_test(
            cost_weight_assumed_to_tested_to_loss,
            save_dir=os.path.join(VISUALIZATION_DIR, "adv_cost_weight_test"),
        )

    def test_reg_weights(self):
        """ """
        print("Test reg weights")
        TESTED_COST_WEIGHTS = [0.1, 0.5, 1.0, 2.0, 10.0, float("inf")]
        MAX_EPOCHS = 116
        model = LinearModel(in_features=DATA_ROW_SIZE)
        loss_fn = BCEWithLogitsLossPNOne()
        cost = CostNormL2(dim=1)
        l2_reg = LinearL1Regularization(0.011)
        training_params = {
            "optimizer": torch.optim.Adam,
            "lr": 0.00018,
        }
        TESTED_REG_WEIGHTS = [0, 0.5, 1.0, 2.0, 10.0]
        reg_functions = ["expected_utility", "recourse", "social_burden"]
        cost_reg_loss: Dict[float, Dict[float, Tuple[float, float]]] = {}
        for reg_function in reg_functions:
            for cost_weight in TESTED_COST_WEIGHTS:
                cost_reg_loss[cost_weight] = {}
                print(f"test weight: {cost_weight}")
                for reg_weight in TESTED_REG_WEIGHTS:
                    print(f"Test reg weight: {reg_weight}")
                    model = LinearModel(in_features=DATA_ROW_SIZE)
                    if cost_weight == float("inf"):
                        delta: Union[IdentityDelta, LinearStrategicDelta] = (
                            IdentityDelta(cost=cost, strategic_model=model)
                        )
                    else:
                        delta = LinearStrategicDelta(
                            strategic_model=model,
                            cost=cost,
                            cost_weight=cost_weight,
                        )

                    if reg_weight == 0:
                        reg = None
                    else:
                        if reg_function == "social_burden":
                            reg = SocialBurden(linear_delta=delta)
                        elif reg_function == "expected_utility":
                            reg = ExpectedUtility(tanh_temp=10)
                        elif reg_function == "recourse":
                            reg = Recourse(sigmoid_temp=10)
                        else:
                            raise ValueError(f"Unknown reg function: {reg_function}")

                    model_suit = ModelSuit(
                        model=model,
                        delta=delta,
                        loss_fn=loss_fn,
                        train_loader=self.train_loader,
                        validation_loader=self.val_loader,
                        test_loader=self.test_loader,
                        training_params=training_params,
                        regularization=reg,
                        regularization_weight=reg_weight,
                    )

                    trainer = pl.Trainer(
                        fast_dev_run=self.fast_dev_run,
                        max_epochs=MAX_EPOCHS,
                        logger=CSVLogger(
                            LOG_DIR,
                            name=f"test_{reg_function}_reg_{reg_weight}_cost_{cost_weight}",
                        ),
                    )
                    trainer.fit(model_suit)
                    trainer.test(model_suit)

                    output = trainer.test(model_suit)
                    mean_loss = np.mean(
                        [output[i]["test_loss_epoch"] for i in range(len(output))]
                    ).item()
                    mean_zero_one_loss = np.mean(
                        [
                            output[i]["test_zero_one_loss_epoch"]
                            for i in range(len(output))
                        ]
                    ).item()

                    cost_reg_loss[cost_weight][reg_weight] = (
                        mean_loss,
                        mean_zero_one_loss,
                    )

            visualize_reg_weight_test(
                cost_reg_loss,
                save_dir=os.path.join(
                    VISUALIZATION_DIR, f"reg_{reg_function}_weight_test"
                ),
            )

    def test_fully_connected_2_layers(self):
        """We check the effect of learning with delta on a fully connected 2 layers model
        for different sizes of the hidden layer. We use the BCEWithLogitsLossPNOne loss function
        """

        MAX_EPOCHS = 116
        loss_fn = BCEWithLogitsLossPNOne()
        cost = CostNormL2(dim=1)
        training_params = {
            "optimizer": torch.optim.Adam,
            "lr": 0.00018,
        }
        delta_train_params = {
            "optimizer_class": torch.optim.SGD,
            "optimizer_param": {"lr": 0.0017},
            "temp": 27,
            "num_epochs": 61,
        }
        TRAIN_DELTA_DIR = os.path.join(DELTA_LOG_DIR, "train_delta")
        TEST_DELTA_DIR = os.path.join(DELTA_LOG_DIR, "test_delta")
        hidden_layer_sizes = [1, 3, 5, 10, 15, 50, 100]

        output_dict: Dict[int, Dict[bool, Tuple[float, float]]] = {}
        for hidden_layer_size in hidden_layer_sizes:
            print(f"Hidden layer size: {hidden_layer_size}")
            model_no_delta_train = FullyConnected2Layers(
                in_features=DATA_ROW_SIZE, hidden_size=hidden_layer_size
            )
            test_delta_non_strategic = NonLinearStrategicDelta(
                cost=cost,
                strategic_model=model_no_delta_train,
                save_dir=os.path.join(
                    TEST_DELTA_DIR, f"test_{hidden_layer_size}_no_delta"
                ),
                training_params=delta_train_params,
            )
            model_suited_no_delta = ModelSuit(
                model=model_no_delta_train,
                delta=IdentityDelta(cost=cost, strategic_model=model_no_delta_train),
                loss_fn=loss_fn,
                train_loader=self.train_loader,
                validation_loader=self.val_loader,
                test_loader=self.test_loader,
                training_params=training_params,
                train_delta_every=1,
            )
            trainer = pl.Trainer(
                fast_dev_run=self.fast_dev_run,
                max_epochs=MAX_EPOCHS,
                logger=CSVLogger(
                    LOG_DIR,
                    name=f"credit_card_test_fully_connected_2_layers_hidden_{hidden_layer_size}_no_delta",
                ),
            )
            trainer.fit(model_suited_no_delta)
            model_suited_no_delta.train_delta_for_test()
            output = trainer.test(model_suited_no_delta)
            output_dict[hidden_layer_size] = {}
            output_dict[hidden_layer_size][False] = (
                output[0]["test_loss_epoch"],
                output[0]["test_zero_one_loss_epoch"],
            )

            model_delta_train = FullyConnected2Layers(
                in_features=DATA_ROW_SIZE, hidden_size=hidden_layer_size
            )
            delta_strategic = NonLinearStrategicDelta(
                cost=cost,
                strategic_model=model_delta_train,
                save_dir=os.path.join(TRAIN_DELTA_DIR, f"{hidden_layer_size}_delta"),
                training_params=delta_train_params,
            )
            model_suit_delta = ModelSuit(
                model=model_delta_train,
                delta=delta_strategic,
                loss_fn=loss_fn,
                train_loader=self.train_loader,
                validation_loader=self.val_loader,
                test_loader=self.test_loader,
                training_params=training_params,
                train_delta_every=1,
            )
            trainer = pl.Trainer(
                fast_dev_run=self.fast_dev_run,
                max_epochs=MAX_EPOCHS,
                logger=CSVLogger(
                    LOG_DIR,
                    name=f"credit_card_test_fully_connected_2_layers_hidden_{hidden_layer_size}_delta",
                ),
            )
            trainer.fit(model_suit_delta)
            model_suit_delta.train_delta_for_test()
            output = trainer.test(model_suit_delta)
            output_dict[hidden_layer_size][True] = (
                output[0]["test_loss_epoch"],
                output[0]["test_zero_one_loss_epoch"],
            )

        save_dir = os.path.join(VISUALIZATION_DIR, "full_connected_2_layers")
        os.makedirs(save_dir, exist_ok=True)
        visualize_full_connected_2_layers(output_dict, save_dir=save_dir)


if __name__ == "__main__":
    unittest.main()
