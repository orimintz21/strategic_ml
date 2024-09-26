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
