import os
from typing import Tuple, Dict


def visualize_cost_weight_test(
    cost_weight_assumed_to_tested_to_loss: Dict[
        float, Dict[float, Tuple[float, float]]
    ],
    save_dir: str,
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


def visualize_loss_test(
    loss_dict: Dict[str, Tuple[float, float]],
    save_dir: str,
):
    import matplotlib.pyplot as plt
    import numpy as np

    """
    Visualizes loss values from a dictionary and saves the plot to the specified directory.

    Parameters:
    - loss_dict: A dictionary with loss function names as keys and tuples of (loss_value, zero_one_loss_value) as values.
    - save_dir: The directory where the plot image will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract data from the dictionary
    loss_names = list(loss_dict.keys())
    loss_values = [values[0] for values in loss_dict.values()]
    zero_one_loss_values = [values[1] for values in loss_dict.values()]

    x = range(len(loss_names))  # X-axis positions
    width = 0.35  # Width of each bar

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the loss values
    ax.bar(
        [pos - width / 2 for pos in x],
        loss_values,
        width,
        label="Loss Value",
        color="skyblue",
    )

    # Plot the zero-one loss values
    ax.bar(
        [pos + width / 2 for pos in x],
        zero_one_loss_values,
        width,
        label="Zero-One Loss Value",
        color="lightgreen",
    )

    # Customize the plot
    ax.set_xlabel("Loss Function", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Comparison of Loss Values", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(loss_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the plot to the specified directory
    save_path = os.path.join(save_dir, "loss_values_plot.png")
    plt.savefig(save_path)
    plt.close()


def visualize_reg_weight_test(
    cost_weight_assumed_to_tested_to_loss: Dict[
        float, Dict[float, Tuple[float, float]]
    ],
    save_dir: str,
):
    """
    This function visualize the results of the cost weight test.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    data = cost_weight_assumed_to_tested_to_loss
    cost_weights = sorted(
        set(key for subdict in data.values() for key in subdict.keys())
    )
    # create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Increase spacing between groups of bars
    group_spacing = 1.0
    bar_width = 0.2
    index = np.arange(len(cost_weights)) * (bar_width * len(data) + group_spacing)

    # Plotting the loss
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (reg_weight, loss_dict) in enumerate(sorted(data.items())):
        losses = [
            loss_dict[real_w][0] if real_w in loss_dict else 0
            for real_w in cost_weights
        ]
        ax.bar(
            index + i * bar_width,
            losses,
            bar_width,
            label=f"Cost Value: {'no movement' if reg_weight == float('inf') else reg_weight}",
        )

    ax.set_xlabel("Regularization Weight", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title(
        "Loss vs Regularization Weight for Different Cost Weights", fontsize=16
    )
    ax.set_xticks(index + bar_width * (len(data) - 1) / 2)
    ax.set_xticklabels(
        ["No Regularization" if real_w == 0 else real_w for real_w in cost_weights],
        fontsize=12,
    )
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    path = os.path.join(save_dir, "reg_weight_test_loss.png")
    plt.savefig(path)

    # Plotting the zero-one loss
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (reg_weight, loss_dict) in enumerate(sorted(data.items())):
        zero_one_losses = [
            loss_dict[real_w][1] if real_w in loss_dict else 0
            for real_w in cost_weights
        ]
        ax.bar(
            index + i * bar_width,
            zero_one_losses,
            bar_width,
            label=f"Cost Weight: {reg_weight if reg_weight != float('inf') else 'no movement'}",
        )

    ax.set_xlabel("Regularization Weight", fontsize=14)
    ax.set_ylabel("Zero-One Loss", fontsize=14)
    ax.set_title(
        "Zero-One Loss vs Regularization Weight for Different Cost Weights", fontsize=16
    )
    ax.set_xticks(index + bar_width * (len(data) - 1) / 2)
    ax.set_xticklabels(
        ["No Regularization" if real_w == 0 else real_w for real_w in cost_weights],
        fontsize=12,
    )
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    path = os.path.join(save_dir, "reg_weight_test_zero_one_loss.png")
    plt.savefig(path)


def visualize_full_connected_2_layers(
    output_dict: Dict[int, Dict[bool, Tuple[float, float]]],
    save_dir: str,
):
    """
    This function visualize the results of the cost weight test.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    data = output_dict
    layers = sorted(data.keys())

    # Initialize lists to hold the extracted values
    loss_delta = []
    loss_no_delta = []
    zero_one_loss_delta = []
    zero_one_loss_no_delta = []

    # Extract the loss and zero-one loss values for each layer
    for layer in layers:
        delta_data = data[layer][True]
        no_delta_data = data[layer][False]
        loss_delta.append(delta_data[0])
        zero_one_loss_delta.append(delta_data[1])
        loss_no_delta.append(no_delta_data[0])
        zero_one_loss_no_delta.append(no_delta_data[1])

    # Define the width of each bar and the positions
    bar_width = 0.35
    x = np.arange(len(layers))  # The label locations
    # Plot Loss vs Number of Layers
    # Plot Loss vs Number of Layers (Bar Chart)
    plt.figure(figsize=(10, 5))
    plt.bar(x - bar_width / 2, loss_delta, width=bar_width, label="Loss with Delta")
    plt.bar(
        x + bar_width / 2, loss_no_delta, width=bar_width, label="Loss without Delta"
    )
    plt.xlabel("Size of Hidden Layer")
    plt.ylabel("Loss")
    plt.title("Loss vs Size of Hidden Layer")
    plt.xticks(x, layers)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "one_hidden_layer_loss.png")
    plt.savefig(path)

    # Plot Zero-One Loss vs Number of Layers (Bar Chart)
    plt.figure(figsize=(10, 5))
    plt.bar(
        x - bar_width / 2,
        zero_one_loss_delta,
        width=bar_width,
        label="Zero-One Loss with Delta",
    )
    plt.bar(
        x + bar_width / 2,
        zero_one_loss_no_delta,
        width=bar_width,
        label="Zero-One Loss without Delta",
    )
    plt.xlabel("Number of Layers")
    plt.ylabel("Zero-One Loss")
    plt.title("Zero-One Loss vs Number of Layers")
    plt.xticks(x, layers)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "one_hidden_layer_zero_one_loss.png")
    plt.savefig(path)
