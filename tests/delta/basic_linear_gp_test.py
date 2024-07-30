import torch
import unittest
from strategic_ml import CostNormL2, LinearStrategicModel, LinearStrategicDelta, LinearAdvDelta

def create_data():
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

    # Print the dataset
    return x, y

class TestLinearStrategicDelta(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up the test")
        self.x, self.y = create_data()
    
    def test_demo(self) -> None:
        # Create a strategic model
        strategic_model = LinearStrategicModel(in_features=2)

        # Create a cost function
        cost = CostNormL2()

        # Create a strategic delta
        strategic_delta = LinearStrategicDelta(cost, strategic_model)

        strategic_model.set_delta(strategic_delta)

        # Train the strategic model
        optimizer = torch.optim.SGD(strategic_model.parameters(), lr=0.1)
        loss = torch.nn.MSELoss()
        for _ in range(100):
            optimizer.zero_grad()
            prediction = strategic_model(self.x)
            output = loss(prediction, self.y)
            output.backward()
            optimizer.step()
        
        print("The strategic model has been trained")
        

        self.assertTrue(True)




if __name__ == '__main__':
    unittest.main()