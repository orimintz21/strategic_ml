# External imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from random import sample

# internal imports
from strategic_ml.models import LinearModel
from strategic_ml.gsc import _GSC 



def visualize_linear_classifier_2D(
    model:LinearModel, data_loader, delta:_GSC, grid_size=100, display_percentage=1.0, prefix=""
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
    assert 0 <= display_percentage <= 1, "display_percentage should be between 0 and 1"

    # plt.figure(figsize=(10, 6))
    plt.clf()

    # Extract weights and bias if the model is a LinearModel
    if isinstance(model, LinearModel):
        w, b = model.get_weight_and_bias()
    else:
        raise AttributeError(
            f"The provided model is not a LinearModel, it's {type(model)}"
        )

    w = w.cpu().numpy().flatten()
    b = b.cpu().numpy().flatten()

    print(f"Weight: {w}, Bias: {b}")

    # Plot all data points
    X, Y = (
        data_loader.dataset.tensors[0].cpu().numpy(),
        data_loader.dataset.tensors[1].cpu().numpy(),
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
    deltas = None

    # Plot deltas (if provided)
    if delta is not None:
        X_selected = X[displayed_indices]  # Only selected points
        x_prime = delta.forward(
            torch.from_numpy(X_selected).to(torch.float64)
        )  # Compute for selected points
        deltas = (x_prime - torch.from_numpy(X_selected)).detach().cpu().numpy()

        # Determine the size of points based on whether they moved or not
        moved_size = 60  # Size for points that moved
        normal_size = 20  # Size for points that didn't move

        # Create arrays to store the sizes of positive and negative points
        pos_sizes = []
        neg_sizes = []

        # Check each point and set size accordingly
        for i, idx in enumerate(positive_indices):
            if np.all(deltas[i] == 0):
                pos_sizes.append(normal_size)
            else:
                pos_sizes.append(moved_size)

        for i, idx in enumerate(negative_indices):
            if np.all(deltas[len(positive_indices) + i] == 0):
                neg_sizes.append(normal_size)
            else:
                neg_sizes.append(moved_size)

    else:
        # If no delta, just use normal size for all points
        pos_sizes = [20] * len(positive_indices)
        neg_sizes = [20] * len(negative_indices)

    # Plot the sampled points with different sizes based on movement
    plt.scatter(
        X[positive_indices, 0],
        X[positive_indices, 1],
        color="blue",
        label="Positive class",
        s=pos_sizes,  # Size for positive class points
    )
    plt.scatter(
        X[negative_indices, 0],
        X[negative_indices, 1],
        color="red",
        label="Negative class",
        s=neg_sizes,  # Size for negative class points
    )

    # Plot arrows for deltas
    if delta is not None:
        for i, idx in enumerate(displayed_indices):
            if (deltas is None) or (deltas[i, 0] == 0 and deltas[i, 1] == 0):
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
    plt.ylim(-10, 10)
    plt.xlim(0, 20)
    plt.title(
        f"Classifier Visualization with {display_percentage * 100}% of Data Points"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(prefix + "_results.png")
    print("Results saved to " + prefix + "'_results.png'")
