#!python3
'''
This file contains base classes and common methods for various types of genetic algorithms.

See codes.py, policies.py, populations.py, and evaluations.py, distributed.py for inheriting classes.
'''

from population import EliteAsexual
from distributed import LocalSynchronous, LocalMultithreaded 
from evaluations import NTimes
from policies import LinearPolicy
from codes import BasicDNA
from common import RandomSeedGenerator

import multiprocessing as mp
import torch


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



    
