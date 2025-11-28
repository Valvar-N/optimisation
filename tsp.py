import math
import random
import numpy as np

class TSP:
    """
    A class to represent a Traveling Salesman Problem.
    """
    def __init__(self, cities):
        """
        Initializes the TSP problem with a list of cities or a distance matrix.
        :param cities: Either a list/np.ndarray of (x, y) coordinates or a distance matrix.
        """
        self.cities = cities
        self.num_cities = len(cities)

    def calculate_cost(self, tour):
        """
        Calculates the total distance of a given tour (closed loop).
        - Validates the tour is a sequence of city indices of correct length.
        - Detects whether self.cities is a coordinate list/array or a distance matrix.
        - Sums distances for consecutive edges and closing edge.
        :param tour: list/tuple of city indices (0-based)
        :return: float total tour distance
        """
        # Basic validation
        if not isinstance(tour, (list, tuple)):
            raise TypeError("tour must be a list or tuple of city indices")
        if len(tour) != self.num_cities:
            raise ValueError(f"tour must contain exactly {self.num_cities} city indices")

        # Detect representation: coords (list/np.ndarray of (x,y)) or distance matrix
        first = self.cities[0]
        is_coords = isinstance(first, (list, tuple, np.ndarray)) and len(first) == 2 \
                    and isinstance(first[0], (int, float, np.floating, np.integer))

        total = 0.0
        n = len(tour)
        for k in range(n):
            i = tour[k]
            j = tour[(k + 1) % n]  # next city, wrap around
            total += self.cost_of(i, j, coords=is_coords)
        return total

    def cost_of(self, city1_index, city2_index, coords=None):
        """
        Return distance between two cities (by index).
        If coords is None, auto-detect from self.cities.
        Handles both numpy arrays and Python lists for coords or distance matrix.
        """
        if coords is None:
            first = self.cities[0]
            coords = isinstance(first, (list, tuple, np.ndarray)) and len(first) == 2 \
                     and isinstance(first[0], (int, float, np.floating, np.integer))

        if coords:
            x1, y1 = self.cities[city1_index]
            x2, y2 = self.cities[city2_index]
            return math.hypot(x2 - x1, y2 - y1)
        else:
            # distance matrix access: try numpy-style first, fall back to nested lists
            try:
                # works for numpy arrays and for lists of lists if using [i][j]
                return float(self.cities[city1_index, city2_index])
            except Exception:
                try:
                    return float(self.cities[city1_index][city2_index])
                except Exception as exc:
                    raise IndexError(f"unable to index distance between {city1_index} and {city2_index}: {exc}")

    def generate_random_solution(self):
        """
        Generates a random tour by shuffling the city indices.
        :return: A list of shuffled city indices.
        """
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour

    def get_neighbor(self, tour):
        """
        Single neighbor generator used by search routines.
        Randomly choose insertion or exchange move.
        """
        if self.num_cities < 2:
            return tour[:]
        if random.random() < 0.5:
            return self.get_neighbor_exchange(tour)
        else:
            return self.get_neighbor_insertion(tour)

    def get_neighbor_insertion(self, tour):
        """
        Remove one city and insert it at a different position.
        """
        if not isinstance(tour, (list, tuple)):
            tour = self.generate_random_solution()
        if len(tour) < 2:
            return list(tour)
        neighbor = list(tour)
        i = random.randrange(len(neighbor))
        city = neighbor.pop(i)
        j = random.randrange(len(neighbor) + 1)  # allow insertion at end
        neighbor.insert(j, city)
        return neighbor

    def get_neighbor_exchange(self, tour):
        """
        Swap two cities in the tour.
        """
        if not isinstance(tour, (list, tuple)):
            tour = self.generate_random_solution()
        if len(tour) < 2:
            return list(tour)
        neighbor = list(tour)
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
