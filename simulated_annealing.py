import math
import random
import tsp

class SimulatedAnnealing:
    def __init__(self, problem, initial_solution, initial_temperature, cooling_rate, min_temperature=1e-3):
        self.problem = problem
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        
    def acceptance_probability(self, old_cost, new_cost, temperature):
        """
        Calculates the probability of accepting a worse solution.
        Follows the formula P = exp((G(Sk) - G(Sc)) / Tk) from your slides.
        """
        if new_cost < old_cost:
            return 1.0
        # The cost difference will be negative, as new_cost is higher (worse)
        return math.exp((old_cost - new_cost) / temperature)
    
    def solve(self, max_iterations=1000):
        # STEP 1: Initialize
        temperature = self.initial_temperature
        self.current_solution = self.problem.generate_random_solution()
        current_cost = self.problem.calculate_cost(self.current_solution)
        self.best_solution = self.current_solution
        best_cost = current_cost
        iteration = 0
        
        history = {'cost': [current_cost]}
        
        while temperature > self.min_temperature or iteration < max_iterations:
            # STEP 2: Select candidate solution
            candidate_solution = self.problem.get_neighbor(self.current_solution)
            candidate_cost = self.problem.calculate_cost(candidate_solution)
            
            # Decide whether to accept the candidate solution //
            
            # CAREFUL: For minimization problems, condition checks are inverted !!!!
            prob = self.acceptance_probability(current_cost, candidate_cost, temperature)
            if best_cost >= candidate_cost and candidate_cost >= current_cost:
                self.current_solution = candidate_solution
                current_cost = candidate_cost
            elif candidate_cost > best_cost:
                self.best_solution = self.current_solution = candidate_solution
                best_cost = current_cost = candidate_cost
            elif current_cost > candidate_cost:
                if random.random() <= prob:
                    self.current_solution = candidate_solution
                    current_cost = candidate_cost
                else:
                    pass  # Reject the candidate solution
            
            history['cost'].append(current_cost)

            # STEP 3: Cool down
            temperature *= (1 - self.cooling_rate)
            iteration += 1
            
        return self.best_solution, best_cost, history