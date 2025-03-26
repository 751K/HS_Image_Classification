import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def plot_weight_distribution(length=21, sigmas=[0.5, 1.0, 2.0], save_path=None):
    """
    Visualize weights calculated by get_weights function with different sigma values
    using smooth curves with x-axis centered at 0.

    Args:
        length: Length of weight array
        sigmas: List of sigma values to visualize
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))

    center = length // 2
    positions = np.arange(length) - center

    for sigma in sigmas:
        squared_distances = (positions) ** 2 + center ** 2
        decay_factor = -0.5 / (sigma ** 2)
        weights = np.exp(decay_factor * squared_distances)
        weights = weights / weights.sum()

        positions_smooth = np.linspace(positions[0], positions[-1], 300)
        spl = make_interp_spline(positions, weights, k=3)  # k=3 for cubic spline
        weights_smooth = spl(positions_smooth)

        plt.plot(positions_smooth, weights_smooth, '-', linewidth=2, label=f'σ={sigma}')

    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return plt


if __name__ == '__main__':
    plot_weight_distribution(length=15, sigmas=[1.2, 1.6, 2.0, 3.0], save_path="weight_distribution.png")