from common import Individual
from sortedcontainers import SortedList
import random
from abc import ABC, abstractmethod

class Population(ABC):

    @abstractmethod
    def __init__(self, 
            dna_class, # : DNA,  TODO how to type correctly here?
            population_size : int, 
            random_seed_generator
            ):
        pass

    @abstractmethod
    def reproduce(self):
        ''' generates a new individual from this population for testing. '''
        pass

    @abstractmethod
    def add_individual(self, individual : Individual):
        ''' add an individual, with fitness, to the population '''
        pass

class EliteAsexual(Population):
    def __init__(self, 
            dna_class, # : DNA,  TODO how to type correctly here?
            population_size : int, 
            random_seed_generator, # TODO generator typing
            num_elites: int, # Must be less than population size
            ):

        self.codes = SortedList([Individual(dna_class(random_seed_generator.random()), -9999) for _ in range(population_size)], key = lambda x: -x.fitness)
        self.population_size = population_size
        self.random_seed_generator = random_seed_generator
        self.num_elites = num_elites


    def reproduce(self):
        i = random.randrange(0,len(self.codes))
        parent = self.codes[i]
        child = parent.dna.mutate(self.random_seed_generator.random())
        return child

    def add_individual(self, individual : Individual):
        # first, remove a random non-elite individual. Then add the new one.
        self.codes.pop(random.randrange(self.num_elites, self.population_size))
        self.codes.add(individual)
        print(f"BEST: {self.codes[0].fitness}. NEW: {individual.fitness}", flush=True)
Population.register(EliteAsexual)
