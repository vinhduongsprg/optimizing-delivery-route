from __future__ import annotations

from typing import Sequence

import numpy as np

from tsp import route_distance


def nearest_neighbor(cities: np.ndarray, depot: Sequence[float]) -> list[int]:
    """Construct a route by repeatedly visiting the nearest unvisited city."""
    unvisited = set(range(len(cities)))
    current_point = np.asarray(depot, dtype=float)
    route: list[int] = []

    while unvisited:
        nearest_index = min(
            unvisited,
            key=lambda city_index: float(np.linalg.norm(cities[city_index] - current_point)),
        )
        route.append(nearest_index)
        unvisited.remove(nearest_index)
        current_point = cities[nearest_index]

    return route


def nearest_neighbor_solution(cities: np.ndarray, depot: Sequence[float]) -> tuple[list[int], float]:
    """Return both the nearest-neighbor route and its total distance."""
    route = nearest_neighbor(cities, depot)
    distance = route_distance(route, cities, depot)
    return route, distance
