from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsp import distance_matrix, route_distance, validate_chromosome


@dataclass(frozen=True)
class GAConfig:
    num_cities: int = 30
    population_size: int = 100
    generations: int = 500
    mutation_rate: float = 0.05
    crossover_rate: float = 0.8
    tournament_size: int = 3
    elitism_count: int = 2
    random_seed: int = 42


def make_population(config: GAConfig, rng: np.random.Generator) -> list[list[int]]:
    """Generate a valid population of random city permutations."""
    population: list[list[int]] = []
    for _ in range(config.population_size):
        chromosome = np.arange(config.num_cities)
        rng.shuffle(chromosome)
        population.append(chromosome.tolist())
    return population


def evaluate_population(population: list[list[int]], cities: np.ndarray, depot: tuple[float, float], distance_matrix_values: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Compute route distances and identify the best chromosome."""
    if distance_matrix_values is None:
        distance_matrix_values = distance_matrix(cities)

    fitness_scores = np.asarray(
        [route_distance(chromosome, cities, depot, distance_matrix_values) for chromosome in population],
        dtype=float,
    )
    best_index = int(np.argmin(fitness_scores))
    return fitness_scores, best_index


def tournament_selection(population: list[list[int]], fitness_scores: np.ndarray, rng: np.random.Generator, tournament_size: int) -> list[int]:
    """Select a parent using tournament selection."""
    tournament_indices = rng.integers(0, len(population), size=tournament_size)
    winner_index = int(tournament_indices[np.argmin(fitness_scores[tournament_indices])])
    return population[winner_index][:]


def order_crossover(parent_one: list[int], parent_two: list[int], rng: np.random.Generator) -> list[int]:
    """Apply order crossover to two valid TSP permutations."""
    length = len(parent_one)
    start, end = sorted(rng.integers(0, length, size=2))

    child = [None] * length
    segment = parent_one[start : end + 1]
    for index in range(start, end + 1):
        child[index] = parent_one[index]

    remaining = [gene for gene in parent_two if gene not in segment]
    cursor = (end + 1) % length
    for gene in remaining:
        while child[cursor] is not None:
            cursor = (cursor + 1) % length
        child[cursor] = gene
        cursor = (cursor + 1) % length

    if any(value is None for value in child):
        raise ValueError("Order crossover generated an invalid chromosome.")

    return [int(value) for value in child]


def swap_mutation(chromosome: list[int], rng: np.random.Generator, mutation_rate: float) -> list[int]:
    """Randomly swap two genes using a configurable mutation probability."""
    mutated = chromosome[:]
    if rng.random() < mutation_rate:
        swap_positions = rng.choice(len(mutated), size=2, replace=False)
        mutated[swap_positions[0]], mutated[swap_positions[1]] = mutated[swap_positions[1]], mutated[swap_positions[0]]
    return mutated


def evolve_population(population: list[list[int]], cities: np.ndarray, depot: tuple[float, float], config: GAConfig, rng: np.random.Generator) -> list[list[int]]:
    """Create the next generation from the current population while preserving elites."""
    distance_values = distance_matrix(cities)
    fitness_scores, _ = evaluate_population(population, cities, depot, distance_values)
    ranked_indices = np.argsort(fitness_scores)

    elites = [population[index][:] for index in ranked_indices[: config.elitism_count]]
    next_population: list[list[int]] = elites[:]

    while len(next_population) < config.population_size:
        parent_one = tournament_selection(population, fitness_scores, rng, config.tournament_size)
        parent_two = tournament_selection(population, fitness_scores, rng, config.tournament_size)

        if rng.random() < config.crossover_rate:
            child_one = order_crossover(parent_one, parent_two, rng)
            child_two = order_crossover(parent_two, parent_one, rng)
        else:
            child_one = parent_one[:]
            child_two = parent_two[:]

        child_one = swap_mutation(child_one, rng, config.mutation_rate)
        child_two = swap_mutation(child_two, rng, config.mutation_rate)

        next_population.extend([child_one, child_two])

    next_population = next_population[: config.population_size]
    for chromosome in next_population:
        if not validate_chromosome(chromosome, config.num_cities):
            raise ValueError(f"Invalid chromosome produced: {chromosome}")

    return next_population


def run_genetic_algorithm(config: GAConfig, cities: np.ndarray, depot: tuple[float, float]) -> dict:
    """Run the genetic algorithm and return the best route and generation history."""
    rng = np.random.default_rng(config.random_seed)
    population = make_population(config, rng)
    distance_values = distance_matrix(cities)

    best_distance = float('inf')
    best_route: list[int] | None = None
    history: list[float] = []

    for _ in range(config.generations):
        fitness_scores, best_index = evaluate_population(population, cities, depot, distance_values)
        current_best = float(fitness_scores[best_index])
        history.append(current_best)

        if current_best < best_distance:
            best_distance = current_best
            best_route = population[best_index][:]

        population = evolve_population(population, cities, depot, config, rng)

    if best_route is None:
        raise RuntimeError("Genetic algorithm did not produce a valid route.")

    return {
        "best_route": best_route,
        "best_distance": best_distance,
        "history": history,
    }
