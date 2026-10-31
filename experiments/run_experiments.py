from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from genetic_algorithm import GAConfig, run_genetic_algorithm
from heuristics import nearest_neighbor_solution
from tsp import generate_instance
from visualization import plot_convergence, plot_population_experiment, plot_route_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write a CSV file containing experiment results."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_algorithm_comparison() -> dict:
    """Compare the nearest-neighbor heuristic against the GA over repeated independent runs."""
    rng = np.random.default_rng(42)
    cities = generate_instance(30, rng)
    depot = (50.0, 50.0)

    nn_route, nn_distance = nearest_neighbor_solution(cities, depot)

    ga_results: list[float] = []
    ga_routes: list[list[int]] = []
    for seed in range(20):
        config = GAConfig(
            num_cities=30,
            population_size=100,
            generations=500,
            mutation_rate=0.05,
            crossover_rate=0.8,
            tournament_size=3,
            elitism_count=2,
            random_seed=seed,
        )
        result = run_genetic_algorithm(config, cities, depot)
        ga_results.append(result["best_distance"])
        ga_routes.append(result["best_route"])

    mean_distance = float(np.mean(ga_results))
    std_dev = float(np.std(ga_results, ddof=1) if len(ga_results) > 1 else 0.0)
    best_distance = float(np.min(ga_results))
    improvement = ((nn_distance - mean_distance) / nn_distance) * 100.0

    comparison_rows = [
        {
            "method": "Nearest Neighbor",
            "mean_distance": format(nn_distance, ".6f"),
            "std_dev": "0.000000",
            "best_distance": format(nn_distance, ".6f"),
        },
        {
            "method": "Genetic Algorithm",
            "mean_distance": format(mean_distance, ".6f"),
            "std_dev": format(std_dev, ".6f"),
            "best_distance": format(best_distance, ".6f"),
        },
    ]
    write_csv(RESULTS_DIR / "algorithm_comparison.csv", ["method", "mean_distance", "std_dev", "best_distance"], comparison_rows)

    best_ga_route = ga_routes[int(np.argmin(ga_results))]
    plot_route_comparison(
        cities,
        depot,
        nn_route,
        best_ga_route,
        nn_distance,
        best_distance,
        RESULTS_DIR / "route_comparison.png",
    )

    summary = {
        "nearest_neighbor_distance": nn_distance,
        "ga_mean_distance": mean_distance,
        "ga_std_dev": std_dev,
        "ga_best_distance": best_distance,
        "improvement_percent": improvement,
    }
    return summary


def run_population_experiment() -> list[dict]:
    """Explore how different population sizes affect route quality."""
    rng = np.random.default_rng(42)
    cities = generate_instance(30, rng)
    depot = (50.0, 50.0)

    results: list[dict] = []
    for population_size in [20, 50, 100, 200]:
        trial_distances: list[float] = []
        for seed in range(20):
            config = GAConfig(
                num_cities=30,
                population_size=population_size,
                generations=500,
                mutation_rate=0.05,
                crossover_rate=0.8,
                tournament_size=3,
                elitism_count=2,
                random_seed=seed,
            )
            result = run_genetic_algorithm(config, cities, depot)
            trial_distances.append(result["best_distance"])

        mean_distance = float(np.mean(trial_distances))
        std_dev = float(np.std(trial_distances, ddof=1) if len(trial_distances) > 1 else 0.0)
        best_distance = float(np.min(trial_distances))

        results.append(
            {
                "population_size": population_size,
                "mean_distance": mean_distance,
                "std_dev": std_dev,
                "best_distance": best_distance,
            }
        )

    rows = [
        {
            "population_size": result["population_size"],
            "mean_distance": format(result["mean_distance"], ".6f"),
            "std_dev": format(result["std_dev"], ".6f"),
            "best_distance": format(result["best_distance"], ".6f"),
        }
        for result in results
    ]
    write_csv(RESULTS_DIR / "population_experiment.csv", ["population_size", "mean_distance", "std_dev", "best_distance"], rows)
    plot_population_experiment(results, RESULTS_DIR / "population_experiment.png")
    return results


def run_convergence_analysis() -> list[float]:
    """Run a representative GA optimization and save convergence traces."""
    rng = np.random.default_rng(42)
    cities = generate_instance(30, rng)
    depot = (50.0, 50.0)

    config = GAConfig(
        num_cities=30,
        population_size=100,
        generations=500,
        mutation_rate=0.05,
        crossover_rate=0.8,
        tournament_size=3,
        elitism_count=2,
        random_seed=42,
    )
    result = run_genetic_algorithm(config, cities, depot)
    plot_convergence(result["history"], RESULTS_DIR / "convergence.png")
    return result["history"]


def run_all() -> dict:
    """Generate all experiment outputs used by the project README."""
    comparison_summary = run_algorithm_comparison()
    population_summary = run_population_experiment()
    convergence_history = run_convergence_analysis()

    return {
        "comparison": comparison_summary,
        "population": population_summary,
        "convergence": convergence_history,
    }


if __name__ == "__main__":
    run_all()
    print("Experiment outputs saved in:", RESULTS_DIR)
