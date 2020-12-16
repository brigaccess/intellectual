#!/usr/bin/env python
# coding: utf-8
import itertools
from operator import add, attrgetter
import math
import random
import secrets # True randomness here!

"""
D I S C L A I M E R

The code is kinda bloated, sorry for that.
Search for '#[0-9]#' to find genetic algorithm stages, as they are
scattered throughout the code (for the sake of logical code separation).
"""


# Fitness calculation from 4.1 is still relevant, with a slight modification
def fitness(items, limits):
    """
    Fitness function generator. Provides items and limits context to fitness function
    """
    if not limits:
        return None
    
    # Actual fitness function. Slight modification == no more solution_idx parameter
    def fitness_func(solution):
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


def greedy(items, limits):
    """
    #1# INITIAL POPULATION
    
    Initialization function generator. 
    Provides items and limits context to initializer
    """
    # Use fitness function to check if we broke stuff
    ff = fitness(items, limits)
    def greedy_func(spec):
        # Get all non-selected items
        non_selected = [i for i in range(len(items)) if spec[i] == 0]
        # Sort by cost
        non_selected = sorted(non_selected, key=lambda x: items[x][2])[::-1]
        offset = 0
        while ff(spec) > 0 and offset < len(non_selected):
            spec[offset] = 1
            offset += 1
        if ff(spec) == 0:
            spec[offset - 1] = 0
    
    return greedy_func


def main():
    # Nothing really changed here since 4.1, you can skip that.
    # If you REALLY want to find changes, I'll mark 'em for you
    with open("input.txt", "r") as f:
        # Read limits
        maxWeight, maxVol = map(int, f.readline().split())
        # Read items
        items = [tuple(map(float, line.split())) for line in f]
        # Find minimal cost item <-- CHANGED SINCE 4.1
        delta = min([i[2] for i in items])

        # Provide context to fitness func
        ff = fitness(items, [maxWeight, maxVol])
        # Provide context to greedy initializer func <-- CHANGED SINCE 4.1
        init = greedy(items, [maxWeight, maxVol])
        # Initialize genetic solver (obviously also changed, but duh)
        pool = Pool(delta, gene_count=len(items), ff=ff, init=init)
        # Run genetic solver
        solution, solution_fitness = pool.run()
        
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


class Specimen():
    """
    Byte-encoded life form. Makes things a little bit simpler
    """
    def __init__(self, generation, gene_count, genome=None):
        if genome is None or type(genome) != list:
            self.genome = [0 for i in range(gene_count)]
        else:
            self.genome = genome
        self.generation = generation
        self.fitness = 0
        
    def __getitem__(self, key):
        return self.genome[key]
    
    def __setitem__(self, key, value):
        self.genome[key] = value
        
    def __len__(self):
        return len(self.genome)
        
    def _cross_over(self, other):
        """
        #3# Crossing-over: specimen with random genes from both parents
        """
        new_genome = [0 for i in range(self.__len__())]
        for i in range(min(len(self), len(other))):
            new_genome[i] = secrets.choice((self[i], other[i]))
        return Specimen(self.generation + 1, None, new_genome)
        
    def __pow__(self, other):
        """
        #3# Crossing-over: two specimen with random genes from both parents
        """
        return self._cross_over(other), self._cross_over(other)
    
    def __unicode__(self):
        return "Specimen (gen{}, fit{}): {}".format(self.generation, self.fitness, str(self.genome))
    
    def __repr__(self):
        return self.__unicode__()
    
    def __str__(self):
        return self.__unicode__()
        

class Pool():
    """
    Gene pool, implementing genetic algorithm
    """
    
    def __init__(self, delta, gene_count=1, population_limit=200, gen_limit=500, min_gen=10, ff=None, init=None):
        # All the life in the pool
        self.specimen = []
        
        # Maximal population
        self.population_limit = population_limit
        
        # Length of a specimen
        self.gene_count = gene_count
        
        # Current generation
        self.generation = 0
        
        # Fitness function
        self.fitness_func = ff
        
        # Generation delta to stop at
        self.delta = delta
        
        # Generation to stop at (if not stopped by delta)
        self.gen_limit = gen_limit
        
        # Minimal amount of generations to evolve for before delta becomes active
        self.minimum_generations = min_gen
        
        # Previous generation fitness
        self.prev_fitness = None
        
        # Function to initialize specimen with
        self.initial_gen_func = init

        
    def run(self):
        """
        Run genetic algorithm
        """
        # Reset current generation
        self.generation = 0
        
        # Generate specimen
        self.big_bang()
        
        # Calculate initial fitness
        for spec in self.specimen:
            spec.fitness = self.fitness_func(spec)
            
        # Sort specimen so that the fittest will be first
        self.specimen = sorted(self.specimen, key=attrgetter('fitness'))[::-1]
        
        # Best specimen
        self._best_specimen = self.specimen[0]
        
        # Calculate max fitness for first iteration
        prev_fitness = self.specimen[0].fitness

        # Evolve unless generation is terminal
        while self.generation < self.gen_limit:
            # Evolve recalculates fitness and 
            self.evolve()
            print("Gen", self.generation, " N=", len(self.specimen), "FIT=", self.specimen[0].fitness)
            
            # Save corpse if this was the best result
            if self._best_specimen.fitness < self.specimen[0].fitness:
                self._best_specimen = self.specimen[0]
            
            # Get maximum fitness (first element is always the fittest)
            current_fitness = self.specimen[0].fitness
            
            # Return if delta matches
            if abs(current_fitness - prev_fitness) <= self.delta:
                print(abs(current_fitness - prev_fitness), self.delta, "DELTA!")
                break
                
        return self._best_specimen, self._best_specimen.fitness
    
    def big_bang(self):
        """
        #1# Initial population

        Returns first generation specimen
        """
        self.specimen = []
        for i in range(self.population_limit):
            # Create a new specimen
            spec = Specimen(self.generation, self.gene_count)
            
            # Choose random initial item and give it to the newborn
            spec[secrets.randbelow(self.gene_count)] = 1
            
            # Process the newborn with external function (if any)
            if self.initial_gen_func is not None:
                self.initial_gen_func(spec)
            self.specimen.append(spec)
    
    def mutate(self, specimen, power=3):
        """
        #4# Mutation: randomization of 3 (or power) genes
        """
        # Record used injection points
        used_points = []
        for i in range(power):
            new_point = None
            while new_point is None or new_point in used_points:
                # Choose new unused injection point
                new_point = secrets.randbelow(len(specimen))
            # Invert injection point value (not so random, I know, but hey)
            specimen[new_point] = 0 if specimen[new_point] == 1 else 1
        return specimen
    
    @staticmethod
    def roulette(subj, position=0, last=0):
        """
        #2# Selection
        
        Requires subject to be a prepared array
        of sector end positions. First sector starts at 0
        
        Modifies the array, so be careful.
        (note: I lost my sanity by this moment)
        """
        # ! ! ! V R A S H A Y T E B A R A B A N ! ! !
        offset = random.random() * last
        position += offset
        position %= last
        # ! ! ! S P I N T H E W H E E L N O W ! ! ! !
        
        prev_val = 0
        i = 0
        # Full revolutions counter
        rot = 0

        # Spin until the value is found
        while True:
            # Roll over
            if i >= len(subj):
                i %= len(subj)
                rot += 1
                prev_val = 0
                        
            if rot > 3:
                # Break if no valid values left after three full revolutions
                return None, position
            
            # Get chosen value
            val = subj[i]

            # Skip empty values in favor of following ones
            if val is not None:
                # Roulette position is bigger than previous sector end 
                # and less than next sector start?
                if prev_val < position and val > position:
                    # Remove the value and return current sector 
                    # alongside roulette position!
                    subj[i] = None
                    return i, position
                prev_val = val

            i += 1

    
    def select(self, n=1):
        """
        #2# Selection
        
        Selection by weighted random based on current fitness.
        """
        sum_fitness = sum([s.fitness for s in self.specimen])
        
        # Sector end coordinates
        sections = list(itertools.accumulate(
            [(spec.fitness / (sum_fitness * 1.0)) * 100 
             for spec in self.specimen]
        ))

        # Last sector coordinate
        last = sections[::-1][0]

        # Filter out zero-width sectors (these come last, as specimen are sorted by fitness)
        truncate_last = 0
        for i in range(len(sections) - 2, 0, -1):
            if sections[i] != last:
                truncate_last = i + 2
                break
        sections[:truncate_last + 2]

        chosen = []
        position = 0
        # Spin the wheel n times
        for i in range(n):
            section, position = Pool.roulette(sections, position=position, last=last)
            
            # No values on the roulette => no more items
            if section is None:
                return chosen
            
            # Append to chosen specimen
            chosen.append(self.specimen[section])

        return chosen
    
    def evolve(self):
        """
        Evolve to the new generation
        """
        # faster than lambda x: x.fitness
        fkey = attrgetter('fitness')

        # Specimen amount to be replaced in current generation
        last_n = math.floor(len(self.specimen) * 0.2)
        
        # Select reproducable specimen (two parents == two children, so n=last_n)
        worthy = self.select(n=last_n)
        
        # Crossover specimen (give birth to new specimen)
        # Make pairs of worthy specimen
        pairs = [(worthy[i * 2], worthy[i * 2 + 1]) 
                 for i in range(math.floor(len(worthy) / 2.0))]

        newborns = []
        # Make babies
        for p in pairs:
            c1, c2 = p[0] ** p[1]
            newborns.append(c1)
            newborns.append(c2)
        
        # Sort newborns by fitness (fittest first)
        for spec in newborns:
            spec.fitness = self.fitness_func(spec)
        newborns = sorted(newborns, key=fkey)[::-1]
        
        # #5# New population
        #
        # Replace the worst 20% of previous generation with the best newborns
        self.specimen = self.specimen[last_n:] + newborns[:last_n]
        
        # Mutate 5% of population
        for spec in random.choices(self.specimen, k=round(len(self.specimen) * 0.05)):
            self.mutate(spec)

        # Recalculate fitness
        for spec in self.specimen:
            spec.fitness = self.fitness_func(spec)
            
        # Sort specimen so that the fittest will be first
        self.specimen = sorted(self.specimen, key=fkey)[::-1]
        
        # Boom! New generation is here
        self.generation += 1

if __name__ == "__main__":
    main()