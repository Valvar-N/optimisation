import math
import random

class TSP:
    """
    A class to represent a Traveling Salesman Problem.
    """
    def __init__(self, cities):
        """
        Initializes the TSP problem with a list of cities.
        :param cities: Either a list of (x, y) coordinates or a distance matrix.
        """
        self.cities = cities
        self.num_cities = len(cities)

    def calculate_cost(self, tour):
        """
        Calculates the total distance of a given tour (closed loop).
        Supports either coordinate lists or distance matrix representations.
        :param tour: A list of city indices representing the tour.
        :return: The total distance of the tour.
        """
        if not tour:
            return 0.0

        total_distance = 0.0
        # Detect coordinate list (each city is a length-2 tuple/list of numbers)
        first = self.cities[0]
        is_coords = (isinstance(first, (list, tuple)) and len(first) == 2
                     and isinstance(first[0], (int, float)))
        n = len(tour)
        for k in range(n):
            i = tour[k]
            j = tour[(k + 1) % n]  # next city, wrap around to start
            if is_coords:
                xi, yi = self.cities[i]
                xj, yj = self.cities[j]
                total_distance += math.hypot(xi - xj, yi - yj)
            else:
                # assume a distance/cost matrix
                total_distance += self.cities[i][j]
        return total_distance

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
        Single neighbor generator used by Simulated Annealing.
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
        Creates a neighbor tour by removing a city and inserting it at a different position.
        :param tour: The current tour.
        :return: A new tour with one city moved to a different position.
        """
        if self.num_cities < 2:
            return tour[:]
        neighbor_tour = tour[:]
        # Select a city to move
        city_index = random.randint(0, self.num_cities - 1)
        city = neighbor_tour.pop(city_index)
        # Select a new position to insert the city (0..num_cities-1 inclusive)
        new_position = random.randint(0, self.num_cities - 1)
        neighbor_tour.insert(new_position, city)
        return neighbor_tour
        
    def get_neighbor_exchange(self, tour):
        """
        Creates a neighbor tour by swapping two random cities.
        This is a common "move operator" for TSP.
        :param tour: The current tour.
        :return: A new tour with two cities swapped.
        """
        if self.num_cities < 2:
            return tour[:]
        neighbor_tour = tour[:]
        # Select two random indices to swap (exchange)
        i, j = random.sample(range(self.num_cities), 2)
        # Swap the cities at these indices
        neighbor_tour[i], neighbor_tour[j] = neighbor_tour[j], neighbor_tour[i]
        return neighbor_tour

    def cost_of(self, cost_list, city1_index, city2_index):
        """Returns the cost between two cities given their indices (matrix access)."""
        return cost_list[city1_index][city2_index]
