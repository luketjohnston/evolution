from common import Individual
from sortedcontainers import SortedList
import random
from abc import ABC, abstractmethod

class Population(ABC):
    '''
    the constructor must set a self.children field containing the individuals that 
    the task manager must evaluate.
    '''

    @abstractmethod
    def __init__(self, 
            dna_class, # : DNA,  TODO how to type correctly here?
            population_size : int, 
            random_seed_generator
            ):
        pass

    @abstractmethod
    def add_grownup(self, individual : Individual):
        ''' add an individual, with fitness, to the population '''
        pass

class EliteAsexual(Population):
    def __init__(self, 
            dna_class, # : DNA,  TODO how to type correctly here?
            parent_population_size : int, 
            child_population_size : int, 
            random_seed_generator, # TODO generator typing
            num_elites: int, # Must be less than population size
            ):

        # TODO: the initial parents should probably be evaluated, instead of assuming they all suck. The first elite will be passed on for no reason
        self.parent_generation = SortedList([Individual(dna_class([random_seed_generator.random()]), (-9999,0)) for _ in range(parent_population_size)], key = lambda x: x.fitness)
        self.parent_population_size = parent_population_size
        self.child_population_size = child_population_size
        self.random_seed_generator = random_seed_generator
        self.num_elites = num_elites
        self.generation = 0
        # "children" are individuals that haven't had their fitness evaluated yet
        self.children = [self.reproduce() for c in range(self.child_population_size)]
        # "grownups" are individuals that have their fitnesses evaluated
        self.grownups = SortedList([], key=lambda x: x.fitness) # sorted in order of fitness, low to high


    def reproduce(self):
        i = random.randrange(0,len(self.parent_generation))
        parent = self.parent_generation[i]
        child = parent.dna.mutate(self.random_seed_generator.random())
        return child


    def add_grownup(self, individual : Individual):
        '''
        If this is the last child of the current generation, will reproduce and return a list 
        of children for the next generation.
        '''
        # first, remove the worst individual from the population. Then add the new one
        self.grownups.add(individual)
        print(f"BEST: {self.parent_generation[0].fitness}. NEW: {individual.fitness}", flush=True)

        if len(self.grownups) >= self.child_population_size:
          print(f"FINISHED GENERATION {self.generation}", flush=True)
          self.last_generation_all_grownups = self.grownups
          self.generation += 1
          # add elite first and then take first parent_pop_size in case num_parents == 1
          if self.num_elites == 0:
              self.parent_generation = SortedList(self.grownups[-self.parent_population_size:], key=lambda x: x.fitness)
          else:
              self.parent_generation = SortedList(self.parent_generation[-self.num_elites:] + self.grownups[-self.parent_population_size:], key=lambda x: x.fitness)[-self.parent_population_size:]

          print(f"New parent generation: {self.parent_generation}", flush=True)
          self.children = [self.reproduce() for c in range(self.child_population_size)]
          self.grownups = SortedList([], key=lambda x: x.fitness)
          return self.children
        return []

# TODO possibly modularize, reduce code duplication 

class Sexual(Population):
    def __init__(self, 
            dna_class, # : DNA,  TODO how to type correctly here?
            parent_population_size : int, 
            child_population_size : int, 
            random_seed_generator, # TODO generator typing
            ):

        # TODO: the initial parents should probably be evaluated, instead of assuming they all suck. The first elite will be passed on for no reason
        self.parent_generation = SortedList([Individual(dna_class([random_seed_generator.random()]), (-9999,0)) for _ in range(parent_population_size)], key = lambda x: x.fitness)
        self.parent_population_size = parent_population_size
        self.child_population_size = child_population_size
        self.random_seed_generator = random_seed_generator
        self.generation = 0
        # "children" are individuals that haven't had their fitness evaluated yet
        self.children = [self.reproduce() for c in range(self.child_population_size)]
        # "grownups" are individuals that have their fitnesses evaluated
        self.grownups = SortedList([], key=lambda x: x.fitness)


    def reproduce(self):
        mother,father = random.sample(self.parent_generation, 2)
        child = mother.dna.combine(father.dna) # first sexual reproduction
        child = child.mutate(self.random_seed_generator.random()) # then mutation
        return child


    def add_grownup(self, individual : Individual):
        '''
        If this is the last child of the current generation, will reproduce and return a list 
        of children for the next generation.
        '''
        # first, remove the worst individual from the population. Then add the new one
        self.grownups.add(individual)
        print(f"BEST: {self.parent_generation[0].fitness}. NEW: {individual.fitness}", flush=True)
        if len(self.grownups) >= self.child_population_size:
          print(f"FINISHED GENERATION {self.generation}", flush=True)
          self.last_generation_all_grownups = self.grownups
          self.generation += 1
          self.parent_generation = self.grownups[:self.parent_population_size]
          print(f"New parent generation: {self.parent_generation}", flush=True)
          self.children = [self.reproduce() for c in range(self.child_population_size)]
          self.grownups = SortedList([], key=lambda x: x.fitness)
          return self.children
        return []

         

          


Population.register(Sexual)
Population.register(EliteAsexual)
