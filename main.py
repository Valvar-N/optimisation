import matplotlib.pyplot as plt
from tsp import TSP
from simulated_annealing import SimulatedAnnealing
import time
from knapsack import Knapsack

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
         plt.text(city_coords[0], city_coords[1], f' {tour.index(i)}', fontsize=12, ha='right')

def plot_convergence(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['cost'])
    plt.title("Convergence of Simulated Annealing")
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

if __name__ == "__main__":
    # --- Problem Definition ---
    # A set of 20 cities for the TSP
    CITIES = [
        (60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
        (100, 160), (200, 160), (140, 140), (40, 120), (100, 120),
        (180, 100), (60, 80), (120, 80), (180, 60), (20, 40),
        (100, 40), (200, 40), (20, 20), (60, 20), (160, 20)
    ]
    tsp_problem = TSP(CITIES)
    
    initial_solution = tsp_problem.generate_random_solution()

    knapsack_problem = Knapsack(capacity=15)
    knapsack_problem.items = { "1": (2, 10), "2": (3, 5), "3": (5, 15), "4": (7, 7), 
                              "5": (1, 6), "6": (4, 18), "7": (1, 3), "8": (3, 12), "9": (2, 8), "10": (6, 20) }
    
    initial_knapsack_solution = knapsack_problem.generate_random_solution()

    # --- SA Parameters ---
    INITIAL_TEMPERATURE = 100000.0
    COOLING_RATE = 0.03
    MINIMUM_TEMPERATURE = 0.01
    MAX_ITERATIONS = 100000

    sa_solver = SimulatedAnnealing(
        problem=knapsack_problem,
        initial_solution=initial_knapsack_solution,
        initial_temperature=INITIAL_TEMPERATURE,
        cooling_rate=COOLING_RATE,
        min_temperature=MINIMUM_TEMPERATURE
    )

    # --- Run Solver & Get Results ---
    print("Starting Simulated Annealing for Knapsack...")
    (best_solution, best_value, history), elapsed = time_function(
        sa_solver.solve, max_iterations=MAX_ITERATIONS
    )
    print(f"Elapsed time: {elapsed:.4f} seconds")
    initial_value = history['cost'][0]

    # --- Print & Visualize ---
    print("\n--- Results ---")
    print(f"Initial random solution weight: {initial_value:.2f}")
    print(f"Final optimized solution weight: {best_value:.2f}")
    print(f"Final solution values: {knapsack_problem.value:.2f}")
    print(f"Improvement: {((initial_value - best_value) / initial_value) * 100:.2f}%\n") # assuming minimization

    plot_convergence(history)

    # Notes:
    # - The initial temperature and cooling rate are crucial for the performance of SA.
    # - The algorithm may get stuck in local minima; multiple runs with different initial solutions can help.
    # - Consider using more advanced techniques like adaptive cooling schedules.
    # - Keeping whole solution history might consume significant memory for large problems is it really needed?
    # - Maybe store only last 10(?) solution history?
    # - Experiment with different neighbor generation strategies for potentially better results.