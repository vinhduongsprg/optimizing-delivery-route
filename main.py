from __future__ import annotations

from experiments.run_experiments import run_all


def main() -> None:
    summary = run_all()
    comparison = summary["comparison"]
    print("Nearest-neighbor distance:", round(comparison["nearest_neighbor_distance"], 3))
    print("GA mean distance:", round(comparison["ga_mean_distance"], 3))
    print("GA std dev:", round(comparison["ga_std_dev"], 3))
    print("GA best distance:", round(comparison["ga_best_distance"], 3))
    print("Improvement over nearest neighbor:", round(comparison["improvement_percent"], 3), "%")
    print("All results and plots were written to the results/ directory.")


if __name__ == "__main__":
    main()
