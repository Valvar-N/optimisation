class TabuSearch:
    def __init__(self, problem, tabu_size, max_iterations):
        self.problem = problem
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
        self.tabu_list = []

    def search(self, initial_solution, probability, neighborhood_size=20):
        current_solution = initial_solution[:]
        best_solution = initial_solution[:]
        best_objective_value = self.problem.calculate_cost(initial_solution)
        
        #history for convergence plotting
        history = {'cost': [best_objective_value]}

        for iteration in range(self.max_iterations):
            # build a set/list of candidate neighbors (get_neighbor returns one tour)
            candidates = [self.problem.get_neighbor(current_solution) for _ in range(neighborhood_size)]

            # remove duplicates and tabu solutions
            neighborhood = []
            for sol in candidates:
                if sol not in self.tabu_list and sol not in neighborhood:
                    neighborhood.append(sol)

            if not neighborhood:
                break

            current_solution = min(neighborhood, key=self.problem.calculate_cost)
            current_objective_value = self.problem.calculate_cost(current_solution)

            if current_objective_value < best_objective_value:
                best_solution = current_solution[:]
                best_objective_value = current_objective_value

            self.tabu_list.append(current_solution[:])
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)

            history['cost'].append(current_objective_value)
            
        return best_solution, best_objective_value, history
    
    def is_tabu(self, solution):
        return solution in self.tabu_list
    
    def add_tabu(self, solution):
        self.tabu_list.append(solution)
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)
