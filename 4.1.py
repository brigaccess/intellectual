#!/usr/bin/env python
# coding: utf-8
from operator import add

import pygad


def fitness(items, limits):
    """
    Fitness function generator. Provides items and limits context to fitness function
    """
    if not limits:
        return None
    
    # Actual fitness function
    def fitness_func(solution, solution_idx):
        # Fitness is calculated using the last vector element
        total = [0 for j in range(len(items[0]))]
        
        # Iterate over solution items
        for i, sol in enumerate(solution):
            # Filter out negative values
            if sol <= 0:
                continue
            # Update vector with item values
            total = list(map(add, total, items[i]))
            # If one or more vector values is over the limit
            if True in [total[j] > limits[j] for j in range(len(limits))]:
                # Set fitness = 0 and break
                total[len(total) - 1] = 0
                break
        # Return fitness
        return total[::-1][0]
    # Return func with provided context
    return fitness_func

def main():
    with open("input.txt", "r") as f:
        # Read limits
        maxWeight, maxVol = map(int, f.readline().split())
        # Read items
        items = [tuple(map(float, line.split())) for line in f]

        # Provide context to fitness func
        ff = fitness(items, [maxWeight, maxVol])
        # Initialize genetic solver
        ga = pygad.GA(num_generations=1000, num_parents_mating=2, fitness_func=ff,
                      sol_per_pop=25, num_genes=len(items), init_range_low=-1, init_range_high=1)
        # Run genetic solver
        ga.run()
        
        # Get best solution
        solution, solution_fitness, _ = ga.best_solution()
        # Prepare item information based on the best solution
        result_items = [[i + 1, items[i][0], items[i][1], items[i][2]]
                        for i, v in enumerate(solution)
                        if v > 0]

        # Write out the result
        with open("output.txt", "w") as f:
            # Header
            line = "\t".join(["N", "W", "V", "C"]) + "\n"
            print(line)
            f.write(line)
            
            # Items
            for item in result_items:
                line = "\t".join(map(str, item)) + "\n"
                print(line, end='')
                f.write(line)

            # The fitness (== sum of C)
            line = "\t".join(["Fitness", str(solution_fitness), "", ""]) + "\n"
            print(line, end='')
            f.write(line)

if __name__ == "__main__":
    main()