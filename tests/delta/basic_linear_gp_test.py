import torch
import unittest

from strategic_ml import (
    CostNormL2,
    LinearModel,
    LinearStrategicDelta,
    LinearAdvDelta,
    StrategicHingeLoss,
    LinearL2Regularization,
)

VERBOSE: bool = False


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


def create_one_dimensional_data():
    x = torch.Tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = torch.Tensor([[1], [1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1]])
    return x, y


class TestLinearStrategicDelta(unittest.TestCase):

    def test_demo(self) -> None:
        """This test just checks that the flow of the code is correct.
        It does not check the correctness of the code.
        """
        self.x, self.y = create_strategic_separable_data()
        # Create a strategic model
        strategic_model = LinearModel(in_features=2)

        # Create a cost function
        cost = CostNormL2(dim=1)

        # Create a strategic delta
        strategic_delta: LinearStrategicDelta = LinearStrategicDelta(
            cost,
            strategic_model,
            cost_weight=1.0,
        )

        # Train the strategic model
        optimizer = torch.optim.SGD(strategic_model.parameters(), lr=0.001)
        loss = torch.nn.BCEWithLogitsLoss()
        strategic_model.train()
        for _ in range(200):
            delta_move: torch.Tensor = strategic_delta(self.x)
            x_prime = delta_move
            optimizer.zero_grad()
            prediction = strategic_model(x_prime)
            output = loss(prediction, self.y)
            output.backward()
            optimizer.step()
        print("The strategic model has been trained")

        # validate the the distance between the two points is less than 1
        for x, y in zip(self.x, self.y):
            x = x.unsqueeze(0)
            x_prime_test = strategic_delta.forward(x)
            self.assertEqual(torch.sign(strategic_model(x_prime_test)), y)
            print_if_verbose(
                f"""
                cost = {cost(x, x_prime_test)},
                y = {y},
                x pred {(strategic_model(x))},
                x_prime = {(strategic_model(x_prime_test))}
                """
            )
            self.assertTrue(cost(x, x_prime_test) < 1)

    def test_strategic_separable_needs_movement(self) -> None:
        self.x, self.y = create_strategic_need_movement()
        # Create a strategic model
        strategic_model = LinearModel(in_features=2)

        # Create a cost function
        cost = CostNormL2(dim=1)

        # Create a strategic delta
        strategic_delta: LinearStrategicDelta = LinearStrategicDelta(
            cost,
            strategic_model,
            cost_weight=1.0,
        )

        # Train the strategic model
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(strategic_model.parameters(), lr=0.01)

        strategic_model.train()
        for _ in range(1401):
            optimizer.zero_grad()
            with torch.no_grad():
                delta_move: torch.Tensor = strategic_delta(self.x)
            output = strategic_model(delta_move)
            loss = loss_fn(output, self.y)
            loss.backward()
            optimizer.step()
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
            self.assertEqual(torch.sign(strategic_model(x_prime_test)), y)
            if torch.sign(strategic_model(x_prime_test)) == y:
                successful += 1
        print(f"Strategic: successful = {successful}")

    def test_strategic_separable_needs_movement_hinge_loss(self) -> None:
        self.x, self.y = create_strategic_need_movement()
        strategic_model = LinearModel(in_features=2)
        cost = CostNormL2(dim=1)
        strategic_delta: LinearStrategicDelta = LinearStrategicDelta(
            cost,
            strategic_model,
            cost_weight=1.0,
        )
        loss_fn = StrategicHingeLoss(
            model=strategic_model,
            delta=strategic_delta,
        )

        optimizer = torch.optim.SGD(strategic_model.parameters(), lr=4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        strategic_model.train()
        REGULARIZATION = 0.001
        for _ in range(1501):
            optimizer.zero_grad()
            loss = loss_fn(self.x, self.y)
            w, b = strategic_model.get_weight_and_bias_ref()
            loss += REGULARIZATION * (torch.norm(w, p=1) + torch.norm(b, p=1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            print_if_verbose(
                f"""loss = {loss.item()}
                                model = {strategic_model.get_weight_and_bias()}
                                movement = {strategic_delta(self.x)}
"""
            )
        print("The strategic model has been trained")
        successful = 0
        print(strategic_model.get_weight_and_bias())
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


class TestLinearAdvDelta(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_adv_separable_needs_movement(self) -> None:
        self.x, self.y = create_adv_need_movement()
        # Create a strategic model
        strategic_model = LinearModel(in_features=2)

        # Create a cost function
        cost = CostNormL2(dim=1)

        # Create a strategic delta
        strategic_delta: LinearAdvDelta = LinearAdvDelta(
            cost,
            strategic_model,
            cost_weight=1.0,
        )

        # Train the strategic model
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(strategic_model.parameters(), lr=0.1)

        strategic_model.train()
        for _ in range(1401):
            optimizer.zero_grad()
            with torch.no_grad():
                delta_move: torch.Tensor = strategic_delta(self.x, self.y)
            output = strategic_model(delta_move)
            loss = loss_fn(output, self.y)
            loss.backward()
            optimizer.step()
        print("The strategic model has been trained")
        successful = 0

        # validate the the distance between the two points is less than 1
        strategic_model.eval()
        for x, y in zip(self.x, self.y):
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            x_prime_test = strategic_delta.forward(x, y)
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
            if torch.sign(strategic_model(x_prime_test)) == y:
                successful += 1
            # self.assertEqual(torch.sign(strategic_model(x_prime_test)), y)
        print(f"Adv: successful = {successful}")

    def test_strategic_one_dimensional_data(self) -> None:
        self.x, self.y = create_one_dimensional_data()
        # Create a strategic model
        strategic_model = LinearModel(in_features=1)
        cost = CostNormL2(dim=1)
        strategic_delta: LinearStrategicDelta = LinearStrategicDelta(
            cost,
            strategic_model,
            cost_weight=1.0,
        )

        # Train the strategic model
        loss_fn = StrategicHingeLoss(model=strategic_model, delta=strategic_delta)
        regularization = LinearL2Regularization(lambda_=0.01)
        optimizer = torch.optim.Adam(strategic_model.parameters(), lr=0.01)
        strategic_model.train()
        # train without delta
        for _ in range(100):
            optimizer.zero_grad()
            output = strategic_model(self.x)
            loss = loss_fn(output, self.y) + regularization(strategic_model)
            loss.backward()
            optimizer.step()
        print(strategic_model.get_weight_and_bias())

        for _ in range(100):
            optimizer.zero_grad()
            delta_move: torch.Tensor = strategic_delta(self.x)
            output = strategic_model(delta_move) + regularization(strategic_model)
            loss = loss_fn(output, self.y)
            optimizer.step()
        print("The strategic model has been trained")
        successful = 0
        strategic_model.eval()
        for x, y in zip(self.x, self.y):
            x = x.unsqueeze(0)
            x_prime_test = strategic_delta.forward(x)
            print(
                f"""
                x = {x},
                delta = {x_prime_test},
                cost = {cost(x, x_prime_test)},
                y = {y},
                x pred {(strategic_model(x))},
                x_prime = {(strategic_model(x_prime_test))}
                """
            )
            if torch.sign(strategic_model(x_prime_test)) == y:
                successful += 1
        print(f"successful = {successful}")
        print(strategic_model.get_weight_and_bias())


if __name__ == "__main__":
    unittest.main()
