#!python3
'''
This file contains base classes and common methods for various types of genetic algorithms.

See codes.py, policies.py, populations.py, and evaluations.py for inheriting classes.
'''

from abc import ABC, abstractmethod
from collections import namedtuple
from sortedcontainers import SortedList
import multiprocessing as mp
import queue
import gymnasium as gym
import random
from random import Random
import torch

MAXINT=2**31-1
class RandomSeedGenerator(Random):
    def __init__(self, seed):
        self.rng = Random()
        self.rng.seed(seed)
    def random(self):
        return self.rng.randrange(0, MAXINT)



class DNA(ABC):
    @abstractmethod
    def __init__(self, random_seed):
        pass
    @abstractmethod
    def mutate(self, random_seed):
        pass # returns a NEW DNA object 
    @abstractmethod
    def serialize(self):
        pass
    @abstractmethod
    def deserialize(serialized):
        pass

class BasicDNA:
    ''' 
    Simple genetic code, simply a list of random seeds which will be used
    to generate the policy network (the initial network and then the perturbations).
    TODO move to codes.py
    '''
    def __init__(self, random_seed):
        self.seeds = [random_seed]
    def mutate(self, random_seed):
        child = BasicDNA(0)
        child.seeds = self.seeds + [random_seed]
        return child
    def serialize(self):
        return tuple(self.seeds)
    def deserialize(serialized):
        gc = BasicDNA(serialized[0])
        gc.seeds = list(serialized)
        return gc
DNA.register(BasicDNA)

class PolicyNetwork(ABC):
    @abstractmethod
    def __init__(self, dna):
        pass
    @abstractmethod
    def act(self, state):
        pass

class LinearPolicy:
    ''' 
    Simple linear policy, one hidden state with relu activation, maps input dimension to 
    action dimension.
    TODO move to policy.py
    '''
    def __init__(self, dna, state_dim, hidden_dim, act_dim):
        self.generator = torch.Generator()

        self.l1 = torch.zeros(size=(state_dim, hidden_dim))
        self.l2 = torch.zeros(size=(hidden_dim, act_dim))

        # recompute ("birth") the new policy network from the dna 
        for s in dna.seeds:
            # TODO should std be something different? For initializing networks?
            self.generator.manual_seed(s)
            self.l1 += torch.normal(mean=0, std=1, size=self.l1.shape, generator=self.generator)
            self.l2 += torch.normal(mean=0, std=1, size=self.l2.shape, generator=self.generator)

    def act(self, state):
        state = torch.tensor(state)
        hidden = torch.nn.functional.relu(torch.matmul(state,  self.l1))
        probs = torch.nn.functional.softmax(torch.matmul(hidden, self.l2), dim=0)
        action = torch.multinomial(probs, 1)
        return action.item()
PolicyNetwork.register(LinearPolicy)


# namedtuples must be named the same as the class name "Individual"
# otherwise mp wont be able to find it
Individual = namedtuple('Individual',['dna','fitness'])

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


class EvaluationMethod(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def eval(self, dna, policy_network_class):
        pass

# given an environment observation_space.shape or action_space.shape, 
# returns the total number of parameters in the flattened space
from functools import reduce
def flattened_shape(shape):
    return reduce(lambda x,y: x*y, shape, 1)  # compute product
    

class NTimes(EvaluationMethod):
    def __init__(self, env_id, times=1, render_mode=None):
        self.env_id = env_id
        self.render_mode=render_mode
        self.times=times
    def eval(self, dna, policy_network_class):
        env = gym.make(self.env_id, render_mode=self.render_mode)
        state_shape = flattened_shape(env.observation_space.shape)
        action_shape  = env.action_space.n
        policy_network = policy_network_class(dna, state_shape, 100, action_shape)
        state, _ = env.reset()
        step = 0
        total_reward = 0
        for i in range(self.times):
            while True:
                step += 1
                action = policy_network.act(state)
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    state, _ = env.reset()
                    break
        return Individual(dna, total_reward / self.times)

EvaluationMethod.register(NTimes)

def worker(dna, policy_network_class, evaluation_method):
  worker.queue.put(evaluation_method.eval(dna, policy_network_class))

def worker_initializer(queue):
    worker.queue = queue

class DistributedMethod(ABC):
    @abstractmethod
    def __init__(self, eval_method: EvaluationMethod):
        pass
    @abstractmethod
    def add_task(self, dna, policy_network_class):
        pass
    @abstractmethod
    def get_task_result(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
class LocalSynchronous(DistributedMethod):
    def __init__(self, evaluation_method: EvaluationMethod):
        self.evaluation_method = evaluation_method
        self.queue = queue.Queue()
        worker_initializer(self.queue)

    def add_task(self, dna, policy_network_class):
        worker(dna, policy_network_class, self.evaluation_method)

    def get_task_result(self):
        return self.queue.get()


DistributedMethod.register(LocalSynchronous)
 
class LocalMultithreaded(DistributedMethod):
    ''' 
    Uses python multiprocessing to add tasks to a local pool

    NOTE: need to create like so:
    with LocalMultithreaded() as x:
        ...
    so that python garbage collector can close correctly
    '''
    def __init__(self, pool_size, evaluation_method: EvaluationMethod):
        self.queue = mp.Queue()
        self.evaluation_method = evaluation_method
        self.pool = mp.Pool(pool_size, initializer=worker_initializer, initargs=(self.queue,))

    def add_task(self, dna, policy_network_class):

        self.pool.apply_async(worker, (dna, policy_network_class, self.evaluation_method))

    def get_task_result(self):
        return self.queue.get()

    # TODO understand this better, is it correct?
    def __enter__(self):
        self.pool.__enter__()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.__exit__(exc_type, exc_val, exc_tb)
        

DistributedMethod.register(LocalMultithreaded)



if __name__ == '__main__':
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
    else:
        print ("MPS device not found.")
        assert False

    population_size = 2
    num_elites = 1
    concurrent_tasks = 1
    env_id = 'CartPole-v1'
    eval_times=10

    target_fitness = 500

    eval_method = NTimes(env_id, times=eval_times)

    with LocalMultithreaded(None, eval_method) as task_manager:
    #with LocalSynchronous(eval_method) as task_manager:

        population = EliteAsexual(
                BasicDNA, 
                population_size, 
                RandomSeedGenerator(0),
                num_elites,
                )

        for i in range(concurrent_tasks):
            child = population.reproduce()
            task_manager.add_task(child, LinearPolicy)

        while True:
             # TODO rename 'individual', 'task_result', etc
             individual = task_manager.get_task_result()
             population.add_individual(individual)
             if individual.fitness >= target_fitness:
                 break
             child = population.reproduce()
             task_manager.add_task(child, LinearPolicy)

    render_eval = NTimes('CartPole-v1', times=1, render_mode='human')
    render_eval.eval(individual.dna, LinearPolicy)



    
