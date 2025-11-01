import random

# Basic structure for a Knapsack problem with binary solution representation

class Knapsack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = {} # item_id: (weight, value)
        self.value = 0
    
    def generate_random_solution(self):
        solution = {}
        for item_id in self.items:
            solution[item_id] = random.choice([0, 1]) # 0: not included, 1: included
            if self.calculate_cost(solution) > self.capacity:
                solution[item_id] = 0 # Ensure we don't exceed capacity
        return solution

    def calculate_cost(self, solution):
        total_weight = sum(self.items[item_id][0] * included for item_id, included in solution.items())
        self.value = sum(self.items[item_id][1] * included for item_id, included in solution.items())
        return total_weight
    
    def get_neighbor(self, solution):
        # Generate a neighbor solution by flipping the inclusion of a random item
        neighbor = solution.copy()
        item_id = random.choice(list(self.items.keys()))
        neighbor[item_id] = 1 - neighbor[item_id] # Flip 0 to 1 or 1 to 0
        if self.calculate_cost(neighbor) > self.capacity:
            neighbor[item_id] = 0 # Ensure we don't exceed capacity
        return neighbor