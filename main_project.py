import matplotlib.pyplot as plt
from tsp import TSP
import time
from tabu_search import TabuSearch
import random
import math 
import logging

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

if __name__ == "__main__":
    
    # configure logging to append to run_log.txt
    logging.basicConfig(filename='run_log.txt', filemode='a',
                        level=logging.INFO)
    logging.info("=== New run ===")
    # --- Problem Definition ---
    # A set of 15 cities for the TSP
    
    cost_of_cities = [
    # 1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
    [  0, 48, 30, 63, 17, 15, 87, 63, 30, 33, 68, 26, 38, 65, 65],  # 1
    [ 48,  0, 45, 35, 77, 21, 96, 64, 24, 17, 39, 30, 32, 76, 31],  # 2
    [ 30, 45,  0, 91, 65, 31, 84, 79, 34, 30, 72, 74, 97, 59, 48],  # 3
    [ 63, 35, 91,  0, 66, 29, 53, 87, 71, 55, 99, 48, 37, 83, 30],  # 4
    [ 17, 77, 65, 66,  0, 40, 15, 27, 92, 35, 45, 47, 15, 72, 95],  # 5
    [ 15, 21, 31, 29, 40,  0, 88, 59, 26, 89, 60, 24, 63, 71, 80],  # 6
    [ 87, 96, 84, 53, 15, 88,  0, 31, 39, 45, 89, 55, 96, 41, 19],  # 7
    [ 63, 64, 79, 87, 27, 59, 31,  0, 49, 46, 94, 66, 72, 75,  0],  # 8
    [ 30, 24, 34, 71, 92, 26, 39, 49,  0, 48, 28, 42, 95, 16, 92],  # 9
    [ 33, 17, 30, 55, 35, 89, 45, 46, 48,  0, 66, 32, 31, 54, 97],  # 10
    [ 68, 39, 72, 99, 45, 60, 89, 94, 28, 66,  0, 90, 47, 23, 41],  # 11
    [ 26, 30, 74, 48, 47, 24, 55, 66, 42, 32, 90,  0, 23, 89,100],  # 12
    [ 38, 32, 97, 37, 15, 63, 96, 72, 95, 31, 47, 23,  0, 39, 30],  # 13
    [ 65, 76, 59, 83, 72, 71, 41, 75, 16, 54, 23, 89, 39,  0, 35],  # 14
    [ 65, 31, 48, 30, 95, 80, 19,  0, 92, 97, 41,100, 30, 35,  0],  # 15
    ]

    tsp_problem = TSP(cost_of_cities)
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
    first = cost_of_cities[0]
    is_coords = (isinstance(first, (list, tuple)) and len(first) == 2
                 and isinstance(first[0], (int, float)))
    if is_coords:
        coords = cost_of_cities
    else:
        n = len(cost_of_cities)
        radius = 100.0
        coords = [(radius * math.cos(2 * math.pi * i / n),
                   radius * math.sin(2 * math.pi * i / n)) for i in range(n)]

    plot_tours(
        cities=coords,
        initial_tour=initial_solution,
        best_tour=best_solution,
        initial_dist=initial_value,
        best_dist=best_value
    )
    
    plot_convergence(history)
    logging.info("Convergence points: %s", history.get('cost', []))