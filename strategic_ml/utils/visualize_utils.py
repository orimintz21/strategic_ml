import numpy as np
import torch
from torch import nn
import torch.optim as optim
from typing import Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import unittest
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
from random import sample

# internal imports
from strategic_ml.models import LinearStrategicModel
from strategic_ml.model_suit import ModelSuit
from strategic_ml.gsc import LinearStrategicDelta, NonLinearStrategicDelta
from strategic_ml.cost_functions import CostNormL2
from strategic_ml.regularization import SocialBurden


def visualize_classifier(
    model, dataset, delta, grid_size=100, display_percentage=1.0, prefix=""
):
    """
    Visualizes a binary classification model's decision boundary and data points.

    Args:
        model: The trained classification model.
        dataset: The dataset containing the features and labels.
        delta (optional): The delta values for strategic point manipulations.
        grid_size: The number of grid points for the decision boundary visualization.
        display_percentage: The percentage of data points to display (from 0 to 1).

    Returns:
        None
    """

    # Extract weights and bias if the model is a LinearStrategicModel
    if isinstance(model, LinearStrategicModel):
        w, b = model.get_weight_and_bias()
    else:
        raise AttributeError(
            f"The provided model is not a LinearStrategicModel, it's {type(model)}"
        )

    w = w.cpu().numpy().flatten()
    b = b.cpu().numpy().flatten()

    print(f"Weight: {w}, Bias: {b}")

    # Plot all data points
    X, Y = (
        dataset.dataset.tensors[0].cpu().numpy(),
        dataset.dataset.tensors[1].cpu().numpy(),
    )

    # Extract points based on labels
    positive_indices = np.where(Y[:, 0] == 1)[0]
    negative_indices = np.where(Y[:, 0] == -1)[0]

    # Control the percentage of points to display
    pos_display_count = int(len(positive_indices) * display_percentage)
    neg_display_count = int(len(negative_indices) * display_percentage)

    # Randomly sample the points to display
    if pos_display_count < len(positive_indices):
        positive_indices = sample(list(positive_indices), pos_display_count)
    if neg_display_count < len(negative_indices):
        negative_indices = sample(list(negative_indices), neg_display_count)

    # Combine the selected positive and negative indices for plotting
    displayed_indices = np.concatenate([positive_indices, negative_indices])

    # Extract only the displayed points for x_min, x_max, y_min, y_max
    displayed_points = X[displayed_indices]
    x_min, x_max = displayed_points[:, 0].min() - 1, displayed_points[:, 0].max() + 1
    y_min, y_max = displayed_points[:, 1].min() - 1, displayed_points[:, 1].max() + 1

    # Create a grid of points for decision boundary
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )

    # Calculate the corresponding y values for the decision boundary wx + b = 0 => y = -(w[0]/w[1]) * x - (b/w[1])
    if w[1] != 0:  # Prevent division by zero in case w[1] is zero
        y_vals = -(w[0] / w[1]) * xx - (b / w[1])
    else:
        # In case w[1] is zero, the line is vertical
        x_vals = np.full_like(xx, -b / w[0])
        y_vals = np.linspace(y_min, y_max, 100)

    plt.plot(xx[0], y_vals[0], label="Decision Boundary wx+b=0", color="red")

    # Plot the sampled points
    plt.scatter(
        X[positive_indices, 0],
        X[positive_indices, 1],
        color="blue",
        label="Positive class",
    )
    plt.scatter(
        X[negative_indices, 0],
        X[negative_indices, 1],
        color="red",
        label="Negative class",
    )

    # Plot deltas (if provided)
    if delta is not None:
        X_selected = X[displayed_indices]  # Only selected points
        x_prime = delta.forward(
            torch.from_numpy(X_selected).to(torch.float64)
        )  # Compute for selected points
        deltas = (x_prime - torch.from_numpy(X_selected)).detach().cpu().numpy()

        # Debugging: print delta values for each point
        print("Delta values:")
        for i, idx in enumerate(displayed_indices):
            print(f"Point {X[idx]} -> Manipulated {x_prime[i]} | Delta {deltas[i]}")

        # Only display deltas for the selected points
        for i, idx in enumerate(displayed_indices):
            if deltas[i, 0] == 0 and deltas[i, 1] == 0:
                # don't show the delta
                continue

            # Plot each delta as an arrow, pointing from the original point to the manipulated point
            plt.arrow(
                X[idx, 0],
                X[idx, 1],
                deltas[i, 0],
                deltas[i, 1],
                head_width=0.2,
                head_length=0.2,
                color="green",
                alpha=0.5,
            )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max")
    plt.ylim(-10, 10)
    plt.xlim(0, 20)
    plt.title(
        f"Classifier Visualization with {display_percentage * 100}% of Data Points"
    )
    plt.legend()
    plt.grid(True)
    # plt.axis('equal')
    plt.savefig(prefix + "_training_results.png")
    print("Training results saved to 'training_results.png'")