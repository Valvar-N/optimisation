import math
import random

class TSP:
    """
    A class to represent a Traveling Salesman Problem.
    """
    def __init__(self, cities):
        """
        Initializes the TSP problem with a list of cities.
        :param cities: A list of tuples, where each tuple is an (x, y) coordinate.
        """
        self.cities = cities
        self.num_cities = len(cities)

    def calculate_tour_distance(self, tour):
        """
        Calculates the total distance of a given tour.
        :param tour: A list of city indices representing the tour.
        :return: The total Euclidean distance of the tour.
        """
        total_distance = 0
        for i in range(self.num_cities):
            from_city_index = tour[i]
            to_city_index = tour[(i + 1) % self.num_cities]

            from_city = self.cities[from_city_index]
            to_city = self.cities[to_city_index]

            distance = math.sqrt((from_city[0] - to_city[0])**2 + (from_city[1] - to_city[1])**2)
            total_distance += distance
        return total_distance

    def generate_random_tour(self):
        """
        Generates a random tour by shuffling the city indices.
        :return: A list of shuffled city indices.
        """
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour
    
    def get_neighbor_tour(self, tour):
        """
        Creates a neighbor tour by swapping two random cities.
        This is a common "move operator" for TSP.
        :param tour: The current tour.
        :return: A new tour with two cities swapped.
        """
        neighbor_tour = tour[:]
        # Select two random indices to swap
        i, j = random.sample(range(self.num_cities), 2)
        # Swap the cities at these indices
        neighbor_tour[i], neighbor_tour[j] = neighbor_tour[j], neighbor_tour[i]
        return neighbor_tour
