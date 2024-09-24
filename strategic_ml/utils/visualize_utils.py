# External imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from random import sample
from typing import Optional

# internal imports
from strategic_ml.models import LinearModel
from strategic_ml.gsc import _GSC


def visualize_data_and_delta_2D(
    model: Optional[LinearModel],
    data_loader,
    delta: _GSC,
    grid_size=100,
    display_percentage=1.0,
    prefix="",
):
    """
    Visualizes a binary classification model's decision boundary and data points.

    Args:
        model: The trained classification model. Only supports LinearModel,
        if the model is not a LinearModel, set it to None.
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

    # Extract weights and bias if the model is a LinearModel
    if model is None:
        print("No model provided, only plotting data points")
    else:
        if isinstance(model, LinearModel):
            w, b = model.get_weight_and_bias()
        else:
            raise AttributeError(
                f"The provided model is not a LinearModel, it's {type(model)}"
            )

        w = w.cpu().numpy().flatten()
        b = b.cpu().numpy().flatten()
        # Calculate the corresponding y values for the decision boundary
        # wx + b = 0 => y = -(w[0]/w[1]) * x - (b/w[1])
        if w[1] != 0:  # Prevent division by zero in case w[1] is zero
            y_vals = -(w[0] / w[1]) * xx - (b / w[1])
        else:
            # In case w[1] is zero, the line is vertical
            x_vals = np.full_like(xx, -b / w[0])
            y_vals = np.linspace(y_min, y_max, 100)

        plt.plot(xx[0], y_vals[0], label="Decision Boundary wx+b=0", color="red")
        print(f"Weight: {w}, Bias: {b}")
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


def visualize_data_and_delta_1D(
    model: Optional[LinearModel],
    data_loader,
    delta: Optional[_GSC] = None,
    display_percentage=1.0,
    prefix="",
):
    """
    Visualizes a binary classification model's decision boundary and data points in 1D,
    making the arrows clearer and adjusting point styles based on delta values.

    Args:
        model: The trained classification model. Only supports LinearModel.
               If the model is not a LinearModel, set it to None.
        data_loader: The data loader containing the features and labels.
        delta (optional): The delta values for strategic point manipulations.
        display_percentage: The percentage of data points to display (from 0 to 1).
        prefix: Prefix for the saved plot filename.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from random import sample
    import torch
    from matplotlib.patches import FancyArrowPatch

    assert 0 <= display_percentage <= 1, "display_percentage should be between 0 and 1"

    plt.figure(figsize=(12, 6))
    plt.clf()

    # Extract data
    X, Y = (
        data_loader.dataset.tensors[0].cpu().numpy(),
        data_loader.dataset.tensors[1].cpu().numpy(),
    )
    Y = Y.flatten()
    X = X.flatten()

    # Extract points based on labels
    positive_indices = np.where(Y == 1)[0]
    negative_indices = np.where(Y == -1)[0]

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

    # Extract only the displayed points for x_min, x_max
    displayed_points = X[displayed_indices]
    x_min, x_max = displayed_points.min() - 1, displayed_points.max() + 1

    # Prepare y-values for plotting
    Y_displayed = Y[displayed_indices]

    # Extract weights and bias if the model is a LinearModel
    if model is None:
        print("No model provided, only plotting data points")
    else:
        if isinstance(model, LinearModel):
            w, b = model.get_weight_and_bias()
        else:
            raise AttributeError(
                f"The provided model is not a LinearModel, it's {type(model)}"
            )

        w = w.cpu().numpy().flatten()
        b = b.cpu().numpy().flatten()
        # For 1D data, w and b should be scalars
        if w[0] != 0:
            decision_boundary_x = -b[0] / w[0]
            plt.axvline(
                x=decision_boundary_x,
                color="green",
                linestyle="--",
                label=f"Decision Boundary (x = {decision_boundary_x:.2f})",
            )
            print(f"Decision boundary at x = {decision_boundary_x}")
        else:
            print("Cannot compute decision boundary, weight is zero")

    # Initialize lists for plotting
    original_positive_x = []
    original_negative_x = []
    shifted_positive_x = []
    shifted_negative_x = []
    original_positive_colors = []
    original_negative_colors = []

    # Plot deltas (if provided)
    if delta is not None:
        X_selected = X[displayed_indices]  # Only selected points
        x_prime = delta.forward(
            torch.from_numpy(X_selected).unsqueeze(1).to(torch.float64)
        )  # Compute for selected points
        deltas = (
            (x_prime - torch.from_numpy(X_selected).unsqueeze(1)).detach().cpu().numpy()
        )
        deltas = deltas.flatten()

        # Process each point
        for i, idx in enumerate(displayed_indices):
            delta_i = deltas[i]
            x_orig = X[idx]
            y_value = 1 if Y[idx] == 1 else -1

            # Determine color intensity based on delta
            if delta_i > 0:
                # Darker color for points with delta > 0
                color_intensity = 1.0  # Fully opaque
                marker_edge_color = "black"  # Darker edge
            else:
                # Lighter color for points with delta <= 0
                color_intensity = 0.2  # Semi-transparent
                marker_edge_color = "gray"  # Lighter edge

            # Add original point to the list
            if Y[idx] == 1:
                original_positive_x.append(x_orig)
                original_positive_colors.append((0, 0, 1, color_intensity))  # Blue
            else:
                original_negative_x.append(x_orig)
                original_negative_colors.append((1, 0, 0, color_intensity))  # Red

            # Plot shifted points and arrows only if delta > 0
            if delta_i > 0:
                x_shifted = x_orig + delta_i

                # Add shifted point to the appropriate list
                if Y[idx] == 1:
                    shifted_positive_x.append((x_shifted, y_value + 0.05))
                else:
                    shifted_negative_x.append((x_shifted, y_value + 0.05))

                # Draw arrow from original to shifted point
                arrow = FancyArrowPatch(
                    (x_orig, y_value),
                    (x_shifted, y_value + 0.05),
                    arrowstyle="->",
                    color="purple",
                    linewidth=1.5,
                    mutation_scale=15,
                    zorder=2,
                )
                plt.gca().add_patch(arrow)
    else:
        # If no delta, plot all original points with default settings
        original_positive_x = X[positive_indices]
        original_negative_x = X[negative_indices]
        original_positive_colors = ["blue"] * len(original_positive_x)
        original_negative_colors = ["red"] * len(original_negative_x)

    # Plot original positive points
    plt.scatter(
        original_positive_x,
        np.ones(len(original_positive_x)),
        color=original_positive_colors,
        edgecolor="black",
        label="Positive class",
        s=50,
        zorder=3,
    )

    # Plot original negative points
    plt.scatter(
        original_negative_x,
        -np.ones(len(original_negative_x)),
        color=original_negative_colors,
        edgecolor="black",
        label="Negative class",
        s=50,
        zorder=3,
    )

    # Plot shifted positive points
    if shifted_positive_x:
        shifted_xs, shifted_ys = zip(*shifted_positive_x)
        plt.scatter(
            shifted_xs,
            shifted_ys,
            color="darkblue",
            marker="x",
            s=60,
            zorder=3,
            label="Shifted Positive Point",
        )

    # Plot shifted negative points
    if shifted_negative_x:
        shifted_xs, shifted_ys = zip(*shifted_negative_x)
        plt.scatter(
            shifted_xs,
            shifted_ys,
            color="darkred",
            marker="x",
            s=60,
            zorder=3,
            label="Shifted Negative Point",
        )

    plt.xlabel("Feature")
    plt.ylabel("Class Label")
    plt.ylim(-2, 2)
    plt.xlim(x_min, x_max)
    plt.title(
        f"1D Classifier Visualization with {display_percentage * 100:.0f}% of Data Points"
    )
    plt.yticks([-1, 1], ["Negative Class", "Positive Class"])
    plt.legend()
    plt.grid(True)
    plt.savefig(prefix + "_results.png")
    plt.show()
    print("Results saved to " + prefix + "_results.png")
