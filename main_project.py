import matplotlib.pyplot as plt
from tsp import TSP
import time
from tabu_search import TabuSearch
import random
import math 
import logging
import numpy as np

def plot_tours(cities, initial_tour, best_tour, initial_dist, best_dist):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_single_tour(cities, initial_tour, f"Initial Tour\nDistance: {initial_dist:.2f}")
    plt.subplot(1, 2, 2)
    plot_single_tour(cities, best_tour, f"Best Tour\nDistance: {best_dist:.2f}")
    plt.tight_layout()
    plt.show()

def plot_single_tour(cities, tour, title):
    x_coords = [cities[i][0] for i in tour]
    y_coords = [cities[i][1] for i in tour]
    x_coords.append(cities[tour[0]][0])
    y_coords.append(cities[tour[0]][1])

    plt.plot(x_coords, y_coords, 'o-')
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    for i, city_coords in enumerate(cities):
         plt.text(city_coords[0], city_coords[1], f' {tour.index(i)+1}', fontsize=12, ha='right')

def plot_convergence(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history['cost'])
    plt.title("Convergence of Tabu Search")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance Found")
    plt.grid(True)
    plt.show()
    
def time_function(func, *args, **kwargs):
    """Run func(*args, **kwargs) and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def log(msg):
    print(msg)
    logging.info(msg)


def generate_tsp_coordinates(n_cities, seed=None, max_coord=100):
    """
    Generate a TSP instance using random 2D coordinates.

    Parameters
    ----------
    n_cities : int
        Number of cities (e.g., 10, 50, 100)
    seed : int or None
        Random seed for reproducibility
    max_coord : int
        Coordinates are in range [0, max_coord]

    Returns
    -------
    coords : np.ndarray (n x 2)
        Array of (x, y) coordinates for each city
    dist_matrix : np.ndarray (n x n)
        Symmetric distance matrix (Euclidean)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate coordinates
    coords = np.random.uniform(0, max_coord, (n_cities, 2))

    # Compute symmetric Euclidean distance matrix
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

    return coords, dist_matrix

if __name__ == "__main__":
    
    # configure logging to append to run_log.txt
    logging.basicConfig(filename='run_log.txt', filemode='a',
                        level=logging.INFO)
    logging.info("=== New run ===")
    # --- Problem Definition ---
    cities_coord, cities_dist = generate_tsp_coordinates(100, seed=42)
    log("Generated TSP instance with 10 cities (coordinates):")
    for idx, (x, y) in enumerate(cities_coord):
        log(f"City {idx+1}: ({x:.2f}, {y:.2f})")

    tsp_problem = TSP(cities_coord)
    tabu_solver = TabuSearch(problem=tsp_problem, tabu_size=4, max_iterations=100)
    probability = random.random()
    initial_solution = tsp_problem.generate_random_solution()
    
     # --- Run Solver & Get Results ---
    print("Starting Tabu Search for TSP...")
    (best_solution, best_value, history), elapsed = time_function(
        tabu_solver.search, probability=probability, initial_solution=initial_solution)
    
    log(f"Elapsed time: {elapsed:.4f} seconds")
    initial_value = tsp_problem.calculate_cost(initial_solution)

    # --- Print & Visualize ---
    log(f"\n--- Results for {tabu_solver.max_iterations} iterations, tabu list of {tabu_solver.tabu_size}---")
    log(f"Initial random solution cost: {initial_value:.2f}")
    log(f"Final optimized solution cost: {best_value:.2f}")
    log(f"Improvement: {((initial_value - best_value) / initial_value) * 100:.2f}%\n") # minimization
    
    # Print best tour as position -> city (1-based city IDs)
    log("Best tour (position -> city):")
    for pos, city in enumerate(best_solution):
        log(f"Position {pos+1}: City {city+1}")
    log("")

    # Prepare coordinates for plotting:
    # if cost_of_cities are coordinates use them, otherwise generate circle layout
    first = cities_coord[0]
    is_coords = (isinstance(first, (list, tuple)) and len(first) == 2
                 and isinstance(first[0], (int, float)))

    plot_tours(
        cities=cities_coord,
        initial_tour=initial_solution,
        best_tour=best_solution,
        initial_dist=initial_value,
        best_dist=best_value
    )
    
    plot_convergence(history)
    logging.info("Convergence points: %s", history.get('cost', []))