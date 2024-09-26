# External imports
import os
from typing import Dict, Tuple, Union
import unittest
import torch
from torch import nn
import numpy as np
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
    SocialBurden,
    ExpectedUtility,
    Recourse
)

from .data_handle import load_data
from .visualization import visualize_cost_weight_test, visualize_reg_weight_test

# Constants
LOG_DIR = "tests/real_data_test/logs/"
VISUALIZATION_DIR = "tests/real_data_test/visualizations/"
DATA_DIR = "tests/real_data_test/data"
DATA_NAME = "creditcard.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_NAME)
DATA_ROW_SIZE = 29


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

    # def test_cost_weighs_strategic(self):
    #     """
    #     In this test we check what a linear model do when it assume a cost weight
    #     and it is tested with different cost weights.
    #     When we have a cost weight of infinity, the model should not move from the base model.
    #     """
    #     print("Test cost weights")
    #     TESTED_COST_WEIGHTS = [0.1, 0.5, 1.0, 2.0, 10.0, float("inf")]
    #     MAX_EPOCHS = 50
    #     model = LinearModel(in_features=DATA_ROW_SIZE)
    #     loss_fn = nn.BCEWithLogitsLoss()
    #     cost = CostNormL2(dim=1)
    #     training_params = {
    #         "optimizer": torch.optim.Adam,
    #         "lr": 0.01,
    #     }

    #     cost_weight_assumed_to_tested_to_loss: Dict[
    #         float, Dict[float, Tuple[float, float]]
    #     ] = {}

    #     for assumed_cost_weight in TESTED_COST_WEIGHTS:
    #         print(f"Assumed cost weight: {assumed_cost_weight}")
    #         model = LinearModel(in_features=DATA_ROW_SIZE)
    #         if assumed_cost_weight == float("inf"):
    #             delta = IdentityDelta(cost=cost, strategic_model=model)
    #         else:
    #             delta = LinearStrategicDelta(
    #                 strategic_model=model,
    #                 cost=cost,
    #                 cost_weight=assumed_cost_weight,
    #             )
    #         model_suit = ModelSuit(
    #             model=model,
    #             delta=delta,
    #             loss_fn=loss_fn,
    #             train_loader=self.train_loader,
    #             validation_loader=self.val_loader,
    #             test_loader=self.test_loader,
    #             training_params=training_params,
    #         )

    #         trainer = pl.Trainer(
    #             fast_dev_run=self.fast_dev_run,
    #             max_epochs=MAX_EPOCHS,
    #             logger=CSVLogger(
    #                 LOG_DIR,
    #                 name=f"credit_card_test_cost_weight_{assumed_cost_weight}_train",
    #             ),
    #         )
    #         trainer.fit(model_suit)
    #         cost_weight_assumed_to_tested_to_loss[assumed_cost_weight] = {}

    #         for test_cost_weight in TESTED_COST_WEIGHTS:
    #             print(f"Test cost weight: {test_cost_weight}")
    #             if test_cost_weight == float("inf"):
    #                 model_suit.test_delta = IdentityDelta(
    #                     cost=cost, strategic_model=model
    #                 )
    #             else:
    #                 model_suit.test_delta = LinearStrategicDelta(
    #                     strategic_model=model,
    #                     cost=cost,
    #                     cost_weight=test_cost_weight,
    #                 )

    #             trainer = pl.Trainer(
    #                 fast_dev_run=self.fast_dev_run,
    #                 max_epochs=MAX_EPOCHS,
    #                 logger=CSVLogger(
    #                     LOG_DIR,
    #                     name=f"credit_card_test_cost_weight_{assumed_cost_weight}_to_{test_cost_weight}",
    #                 ),
    #             )
    #             output = trainer.test(model_suit)
    #             mean_loss = np.mean(
    #                 [output[i]["test_loss_epoch"] for i in range(len(output))]
    #             ).item()
    #             mean_zero_one_loss = np.mean(
    #                 [output[i]["test_zero_one_loss_epoch"] for i in range(len(output))]
    #             ).item()

    #             cost_weight_assumed_to_tested_to_loss[assumed_cost_weight][
    #                 test_cost_weight
    #             ] = (mean_loss, mean_zero_one_loss)

    #     visualize_cost_weight_test(
    #         cost_weight_assumed_to_tested_to_loss,
    #         save_dir=os.path.join(VISUALIZATION_DIR, "strategic_cost_weight_test"),
    #     )

    # def test_cost_weighs_adv(self):
    #     """
    #     In this test we check what a linear model do when it assume a cost weight
    #     and it is tested with different cost weights.
    #     When we have a cost weight of infinity, the model should not move from the base model.
    #     """
    #     print("Test cost weights")
    #     TESTED_COST_WEIGHTS = [0.1, 0.5, 1.0, 2.0, 10.0, float("inf")]
    #     MAX_EPOCHS = 50
    #     model = LinearModel(in_features=DATA_ROW_SIZE)
    #     loss_fn = nn.BCEWithLogitsLoss()
    #     cost = CostNormL2(dim=1)
    #     training_params = {
    #         "optimizer": torch.optim.Adam,
    #         "lr": 0.01,
    #     }

    #     cost_weight_assumed_to_tested_to_loss: Dict[
    #         float, Dict[float, Tuple[float, float]]
    #     ] = {}

    #     for assumed_cost_weight in TESTED_COST_WEIGHTS:
    #         print(f"Assumed cost weight: {assumed_cost_weight}")
    #         model = LinearModel(in_features=DATA_ROW_SIZE)
    #         if assumed_cost_weight == float("inf"):
    #             delta = IdentityDelta(cost=cost, strategic_model=model)
    #         else:
    #             delta = LinearAdvDelta(
    #                 strategic_model=model,
    #                 cost=cost,
    #                 cost_weight=assumed_cost_weight,
    #             )
    #         model_suit = ModelSuit(
    #             model=model,
    #             delta=delta,
    #             loss_fn=loss_fn,
    #             train_loader=self.train_loader,
    #             validation_loader=self.val_loader,
    #             test_loader=self.test_loader,
    #             training_params=training_params,
    #         )

    #         trainer = pl.Trainer(
    #             fast_dev_run=self.fast_dev_run,
    #             max_epochs=MAX_EPOCHS,
    #             logger=CSVLogger(
    #                 LOG_DIR,
    #                 name=f"credit_card_test_cost_weight_{assumed_cost_weight}_train",
    #             ),
    #         )
    #         trainer.fit(model_suit)
    #         cost_weight_assumed_to_tested_to_loss[assumed_cost_weight] = {}

    #         for test_cost_weight in TESTED_COST_WEIGHTS:
    #             print(f"Test cost weight: {test_cost_weight}")
    #             if test_cost_weight == float("inf"):
    #                 model_suit.test_delta = IdentityDelta(
    #                     cost=cost, strategic_model=model
    #                 )
    #             else:
    #                 model_suit.test_delta = LinearAdvDelta(
    #                     strategic_model=model,
    #                     cost=cost,
    #                     cost_weight=test_cost_weight,
    #                 )

    #             trainer = pl.Trainer(
    #                 fast_dev_run=self.fast_dev_run,
    #                 max_epochs=MAX_EPOCHS,
    #                 logger=CSVLogger(
    #                     LOG_DIR,
    #                     name=f"credit_card_test_cost_weight_{assumed_cost_weight}_to_{test_cost_weight}",
    #                 ),
    #             )
    #             output = trainer.test(model_suit)
    #             mean_loss = np.mean(
    #                 [output[i]["test_loss_epoch"] for i in range(len(output))]
    #             ).item()
    #             mean_zero_one_loss = np.mean(
    #                 [output[i]["test_zero_one_loss_epoch"] for i in range(len(output))]
    #             ).item()

    #             cost_weight_assumed_to_tested_to_loss[assumed_cost_weight][
    #                 test_cost_weight
    #             ] = (mean_loss, mean_zero_one_loss)

    #     visualize_cost_weight_test(
    #         cost_weight_assumed_to_tested_to_loss,
    #         save_dir=os.path.join(VISUALIZATION_DIR, "adv_cost_weight_test"),
    #     )

    def test_reg_weights(self):
        """ """
        print("Test reg weights")
        TESTED_COST_WEIGHTS = [0.5, 1.0, 2.0, float("inf")]
        TESTED_REG_WEIGHTS = [0, 0.5, 1.0, 2.0, 10.0]
        MAX_EPOCHS = 50
        model = LinearModel(in_features=DATA_ROW_SIZE)
        loss_fn = nn.BCEWithLogitsLoss()
        cost = CostNormL2(dim=1)
        training_params = {
            "optimizer": torch.optim.Adam,
            "lr": 0.01,
        }
        reg_functions = [ 'expected_utility', 'recourse', 'social_burden']
        cost_reg_loss: Dict[float, Dict[float, Tuple[float, float]]] = {}
        for reg_function in reg_functions:
            for cost_weight in TESTED_COST_WEIGHTS:
                cost_reg_loss[cost_weight] = {}
                print(f"test weight: {cost_weight}")
                for reg_weight in TESTED_REG_WEIGHTS:
                    print(f"Test reg weight: {reg_weight}")
                    model = LinearModel(in_features=DATA_ROW_SIZE)
                    if cost_weight == float("inf"):
                        delta: Union[IdentityDelta, LinearStrategicDelta] = IdentityDelta(
                            cost=cost, strategic_model=model
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
                        if reg_function == 'social_burden':
                            reg = SocialBurden(linear_delta=delta)
                        elif reg_function == 'expected_utility':
                            reg = ExpectedUtility(tanh_temp=10)
                        elif reg_function == 'recourse':
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
                        [output[i]["test_zero_one_loss_epoch"] for i in range(len(output))]
                    ).item()

                    cost_reg_loss[cost_weight][reg_weight] = (mean_loss, mean_zero_one_loss)

            visualize_reg_weight_test(
                cost_reg_loss,
                save_dir=os.path.join(VISUALIZATION_DIR, f"reg_{reg_function}_weight_test"),
            )


if __name__ == "__main__":
    unittest.main()
