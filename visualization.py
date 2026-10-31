from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_route(cities: np.ndarray, depot: tuple[float, float], route: list[int], title: str, output_path: str | Path, color: str = "tab:orange") -> None:
    """Show a single route on a map of the delivery locations."""
    figure, axis = plt.subplots(figsize=(8, 8))

    x_values = cities[:, 0]
    y_values = cities[:, 1]
    axis.scatter(x_values, y_values, c="tab:blue", s=30, label="Cities")
    axis.scatter([depot[0]], [depot[1]], s=120, marker="D", color="tab:red", label="Depot")

    ordered_indices = [*route]
    route_x = [depot[0]] + [cities[index, 0] for index in ordered_indices] + [depot[0]]
    route_y = [depot[1]] + [cities[index, 1] for index in ordered_indices] + [depot[1]]
    axis.plot(route_x, route_y, color=color, linewidth=2, alpha=0.9, label="Route")

    for city_index in ordered_indices:
        axis.text(cities[city_index, 0], cities[city_index, 1], str(city_index), fontsize=8, ha="right")

    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_route_comparison(cities: np.ndarray, depot: tuple[float, float], nn_route: list[int], ga_route: list[int], nn_distance: float, ga_distance: float, output_path: str | Path) -> None:
    """Plot both the nearest-neighbor and GA routes on the same instance."""
    figure, axis = plt.subplots(figsize=(9, 9))
    axis.scatter(cities[:, 0], cities[:, 1], c="tab:blue", s=35, label="Cities")
    axis.scatter([depot[0]], [depot[1]], s=120, marker="D", color="tab:red", label="Depot")

    for label, route, color in [("Nearest Neighbor", nn_route, "tab:orange"), ("Genetic Algorithm", ga_route, "tab:green")]:
        ordered = [*route]
        xs = [depot[0]] + [cities[index, 0] for index in ordered] + [depot[0]]
        ys = [depot[1]] + [cities[index, 1] for index in ordered] + [depot[1]]
        axis.plot(xs, ys, color=color, linewidth=2, alpha=0.9, label=f"{label} ({(nn_distance if label == 'Nearest Neighbor' else ga_distance):.2f})")

    axis.set_title("Nearest Neighbor vs Genetic Algorithm")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_convergence(history: list[float], output_path: str | Path) -> None:
    """Plot best route distance across generations."""
    figure, axis = plt.subplots(figsize=(9, 5))
    generations = np.arange(1, len(history) + 1)
    axis.plot(generations, history, color="tab:green", linewidth=2)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Best Route Distance")
    axis.set_title("GA Convergence")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_population_experiment(results: list[dict], output_path: str | Path) -> None:
    """Show mean route distance by population size with one standard deviation error bars."""
    sizes = [item["population_size"] for item in results]
    means = [item["mean_distance"] for item in results]
    stds = [item["std_dev"] for item in results]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(sizes, means, yerr=stds, capsize=5, color="tab:purple", alpha=0.8)
    axis.set_xlabel("Population Size")
    axis.set_ylabel("Mean Final Route Distance")
    axis.set_title("Population Size Experiment")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
