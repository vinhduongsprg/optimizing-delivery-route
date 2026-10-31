from __future__ import annotations

from typing import Sequence

import numpy as np


def euclidean_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    """Compute the Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))


def distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Return a matrix containing all pairwise Euclidean distances between cities."""
    differences = cities[:, None, :] - cities[None, :, :]
    return np.linalg.norm(differences, axis=2)


def route_distance(route: Sequence[int], cities: np.ndarray, depot: Sequence[float], distance_matrix_values: np.ndarray | None = None) -> float:
    """Compute the total path length for a route visiting city indices in order."""
    if len(route) == 0:
        return 0.0

    route = list(route)
    if distance_matrix_values is None:
        distance_matrix_values = distance_matrix(cities)

    total_distance = euclidean_distance(depot, cities[route[0]])
    for city_index, next_city_index in zip(route[:-1], route[1:]):
        total_distance += float(distance_matrix_values[city_index, next_city_index])
    total_distance += euclidean_distance(cities[route[-1]], depot)
    return float(total_distance)


def validate_chromosome(chromosome: Sequence[int], num_cities: int) -> bool:
    """Ensure a chromosome is a valid permutation of all cities."""
    values = list(chromosome)
    return len(values) == num_cities and sorted(values) == list(range(num_cities))


def generate_instance(num_cities: int, rng: np.random.Generator, depot: Sequence[float] | None = None) -> np.ndarray:
    """Generate deterministic city coordinates in a bounded area."""
    if depot is None:
        depot = (50.0, 50.0)

    coordinates = rng.uniform(0.0, 100.0, size=(num_cities, 2))
    return coordinates
