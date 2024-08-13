from typing import Any, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import unittest

from strategic_ml import (
    CostNormL2,
    CostNormLInf,
    LinearStrategicModel,
    NonLinearStrategicDelta,
    NonLinearAdvDelta,
    NonLinearNoisyLabelDelta,
)

VERBOSE: bool = True

TRAINING_PARAMS: Dict[str, Any] = {
    "num_epochs": 100,
    "optimizer_class": optim.Adam,
    "optimizer_params": {
        "lr": 0.15,
    },
    "early_stopping": 10,
    "temp": 9,
}


def print_if_verbose(message: str) -> None:
    global VERBOSE
    if VERBOSE:
        print(message)


def create_strategic_separable_data():
    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # Generate the first half of the points with the first index less than -5
    x1 = torch.cat((torch.randn(5, 1) - 10, torch.randn(5, 1)), dim=1)

    # Generate the second half of the points with the first index greater than 5
    x2 = torch.cat((torch.randn(5, 1) + 10, torch.randn(5, 1)), dim=1)

    # Concatenate both parts to create the dataset
    x = torch.cat((x1, x2), dim=0)

    # Create labels: 1 for the first half, -1 for the second half
    y1 = torch.ones(5, 1)
    y2 = -torch.ones(5, 1)
    y = torch.cat((y1, y2), dim=0)
    return x, y


def create_adv_need_movement():
    x_p = torch.Tensor([[1, -1], [1, 1]])
    y_p = torch.Tensor([[1], [1]])
    x_n = torch.Tensor([[-1, 10]])
    y_n = torch.Tensor([[-1]])

    x = torch.cat((x_p, x_n), dim=0)
    y = torch.cat((y_p, y_n), dim=0)
    return x, y


def create_strategic_need_movement():
    x_p = torch.Tensor([[1, -1], [1, 1]])
    y_p = torch.Tensor([[1], [1]])
    x_n = torch.Tensor([[-1, -1], [-1, 1]])
    y_n = torch.Tensor([[-1], [-1]])

    x = torch.cat((x_p, x_n), dim=0)
    y = torch.cat((y_p, y_n), dim=0)
    return x, y


class NonLinearModel(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(NonLinearModel, self).__init__()
        self.fc1 = nn.Linear(in_features, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TestLinearStrategicDelta(unittest.TestCase):

    # def test_demo(self) -> None:
    #     """This test just checks that the flow of the code is correct.
    #     It does not check the correctness of the code.
    #     """
    #     self.x, self.y = create_strategic_separable_data()
    #     # Create a strategic model
    #     strategic_model = LinearStrategicModel(in_features=2)

    #     # Create a cost function
    #     cost = CostNormLInf()

    #     # Create a strategic delta
    #     strategic_delta: NonLinearStrategicDelta = NonLinearStrategicDelta(
    #         cost, strategic_model, cost_weight=1.0, training_params=TRAINING_PARAMS
    #     )

    #     # Train the strategic model
    #     optimizer = torch.optim.SGD(strategic_model.parameters(), lr=0.001)
    #     loss = torch.nn.BCEWithLogitsLoss()
    #     strategic_model.train()
    #     for _ in range(200):
    #         delta_move: torch.Tensor = strategic_delta(self.x)
    #         x_prime = delta_move
    #         optimizer.zero_grad()
    #         prediction = strategic_model(x_prime)
    #         output = loss(prediction, self.y)
    #         output.backward()
    #         optimizer.step()
    #     print("The strategic model has been trained")

    #     # validate the the distance between the two points is less than 1
    #     for x, y in zip(self.x, self.y):
    #         x = x.unsqueeze(0)
    #         x_prime_test = strategic_delta.forward(x)
    #         print_if_verbose(
    #             f"""
    #             x = {x},
    #             delta = {x_prime_test},
    #             cost = {cost(x, x_prime_test)},
    #             y = {y},
    #             x pred {(strategic_model(x))},
    #             x_prime = {(strategic_model(x_prime_test))}
    #             """
    #         )
    #         self.assertTrue(cost(x, x_prime_test) < 1)
    #     print(f"{strategic_model.get_weights_and_bias( )}")

    def test_strategic_separable_needs_movement(self) -> None:
        self.x, self.y = create_strategic_need_movement()
        # Create a strategic model
        strategic_model = LinearStrategicModel(in_features=2)

        # Create a cost function
        cost = CostNormL2()

        # Create a strategic delta
        strategic_delta: NonLinearStrategicDelta = NonLinearStrategicDelta(
            cost, strategic_model, cost_weight=1.0, training_params=TRAINING_PARAMS
        )

        # Train the strategic model
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(strategic_model.parameters(), lr=0.1)

        strategic_model.train()
        for i in range(1401):
            optimizer.zero_grad()
            delta_move: torch.Tensor = strategic_delta(self.x)
            output = (strategic_model(delta_move ))
            loss = loss_fn(output, self.y)
            loss.backward()
            optimizer.step()
            print_if_verbose(f"""loss = {loss.item()}
                             model = {strategic_model.get_weights_and_bias()}""")
        print("The strategic model has been trained")
        successful = 0

        # validate the the distance between the two points is less than 1
        strategic_model.eval()
        for x, y in zip(self.x, self.y):
            x = x.unsqueeze(0)
            x_prime_test = strategic_delta.forward(x)
            print_if_verbose(
                f"""
                x = {x},
                delta = {x_prime_test},
                cost = {cost(x, x_prime_test)},
                y = {y},
                x pred {(strategic_model(x))},
                x_prime = {(strategic_model(x_prime_test))}
                """
            )
            # self.assertEqual(torch.sign(strategic_model(x_prime_test)), y)
            if torch.sign(strategic_model(x_prime_test)) == y:
                successful += 1
        print(f"Strategic: successful = {successful}")


# class TestLinearAdvDelta(unittest.TestCase):
#     def setUp(self) -> None:
#         pass

#     def test_adv_separable_needs_movement(self) -> None:
#         self.x, self.y = create_adv_need_movement()
#         # Create a strategic model
#         strategic_model = LinearStrategicModel(in_features=2)

#         # Create a cost function
#         cost = CostNormL2()

#         # Create a strategic delta
#         strategic_delta: LinearAdvDelta = LinearAdvDelta(
#             cost,
#             strategic_model,
#             cost_weight=1.0,
#         )

#         # Train the strategic model
#         loss_fn = torch.nn.BCEWithLogitsLoss()
#         optimizer = torch.optim.Adam(strategic_model.parameters(), lr=0.01)

#         strategic_model.train()
#         for _ in range(1401):
#             optimizer.zero_grad()
#             with torch.no_grad():
#                 delta_move: torch.Tensor = strategic_delta(self.x, self.y)
#             output = strategic_model(delta_move)
#             loss = loss_fn(output, self.y)
#             loss.backward()
#             optimizer.step()
#         print("The strategic model has been trained")
#         successful = 0

#         # validate the the distance between the two points is less than 1
#         strategic_model.eval()
#         for x, y in zip(self.x, self.y):
#             x = x.unsqueeze(0)
#             y = y.unsqueeze(0)
#             x_prime_test = strategic_delta.forward(x, y)
#             print_if_verbose(
#                 f"""
#                 x = {x},
#                 delta = {x_prime_test},
#                 cost = {cost(x, x_prime_test)},
#                 y = {y},
#                 x pred {(strategic_model(x))},
#                 x_prime = {(strategic_model(x_prime_test))}
#                 """
#             )
#             if torch.sign(strategic_model(x_prime_test)) == y:
#                 successful += 1
#             # self.assertEqual(torch.sign(strategic_model(x_prime_test)), y)
#         print(f"Adv: successful = {successful}")


# Test for the noisy label delta


# class TestLinearNoisyLabelDelta(unittest.TestCase):
#     def setUp(self) -> None:
#         pass

#     def test_adv_separable_needs_movement(self) -> None:
#         self.x, self.y = create_adv_need_movement()
#         # Create a strategic model
#         strategic_model = LinearStrategicModel(in_features=2)

#         # Create a cost function
#         cost = CostNormL2()

#         # Create a strategic delta
#         strategic_delta: LinearNoisyLabelDelta = LinearNoisyLabelDelta(
#             cost,
#             strategic_model,
#             cost_weight=1.0,
#             p_bernoulli=0.9,
#         )

#         # Train the strategic model
#         loss_fn = torch.nn.BCEWithLogitsLoss()
#         optimizer = torch.optim.Adam(strategic_model.parameters(), lr=0.01)

#         strategic_model.train()
#         for _ in range(1401):
#             optimizer.zero_grad()
#             with torch.no_grad():
#                 delta_move: torch.Tensor = strategic_delta(self.x, self.y)
#             output = strategic_model(delta_move)
#             loss = loss_fn(output, self.y)
#             loss.backward()
#             optimizer.step()
#         print("The strategic model has been trained")

#         # validate the the distance between the two points is less than 1
#         successful = 0
#         strategic_model.eval()
#         for x, y in zip(self.x, self.y):
#             x = x.unsqueeze(0)
#             y = y.unsqueeze(0)
#             x_prime_test = strategic_delta.forward(x, y)
#             print_if_verbose(
#                 f"""
#                 x = {x},
#                 delta = {x_prime_test},
#                 cost = {cost(x, x_prime_test)},
#                 y = {y},
#                 x pred {(strategic_model(x))},
#                 x_prime = {(strategic_model(x_prime_test))}
#                 """
#             )
#             if torch.sign(strategic_model(x_prime_test)) == y:
#                 successful += 1
#         print(f"Noisy Label: successful = {successful}")


if __name__ == "__main__":
    unittest.main()
