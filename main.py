#!python
'''
This file contains base classes and common methods for various types of genetic algorithms.

See codes.py, policies.py, populations.py, and evaluations.py, distributed.py for inheriting classes.
'''

import pickle
from population import EliteAsexual
from distributed import LocalSynchronous, LocalMultithreaded, DistributedRabbitMQ
from evaluations import NTimes
from policies import LinearPolicy, ConvPolicy
from codes import BasicDNA
import sys
from common import RandomSeedGenerator
import time

import multiprocessing as mp
import torch


if __name__ == '__main__':
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
    else:
        print ("MPS device not found.")
        assert False

    assert (sys.argv[1] == 'worker') or (sys.argv[1] == 'master')
    is_master = (sys.argv[1] == 'master')


    population_size = 1000
    num_elites = 1
    parent_population_size = 20 # taken from GA paper for atari
    child_population_size = 1000 # taken from GA paper for atari

    parent_population_size = 32
    child_population_size = 128 

    env_id = 'CartPole-v1'
    #env_id = 'ALE/Breakout-v5'
    #env_id = 'ALE/Frostbite-v5'
    eval_times=10
    #policy_class = ConvPolicy
    policy_class = LinearPolicy

    target_fitness = 5000
    best_fitness = 0

    total_training_frames = 0

    eval_method = NTimes(env_id, policy_network_class=policy_class, times=eval_times)
    #eval_method = NTimes(env_id, times=eval_times, render_mode='human')
    #with LocalMultithreaded(None, eval_method) as task_manager:
    #with LocalSynchronous(eval_method) as task_manager:
    with DistributedRabbitMQ(eval_method, is_master=is_master) as task_manager:
        if not is_master:
            task_manager.start_worker()
        else:
            print("Creating population")
            population = EliteAsexual(
                    BasicDNA, 
                    parent_population_size, 
                    child_population_size,
                    RandomSeedGenerator(0),
                    num_elites,
                    )

            for child in population.children:
                task_manager.add_task(child)

            start = time.time()

            for individual, metadata in task_manager.get_task_results():
                 # TODO rename 'individual', 'task_result', etc
                 total_training_frames += metadata['total_frames']
                 print(metadata)
                 print("Average total fps: ", total_training_frames / (time.time() - start))
                 # most of the time, next_generation will be an empty list.
                 next_generation = population.add_grownup(individual)
                 if individual.fitness >= target_fitness:
                     break
                 if individual.fitness > best_fitness:
                     best_fitness = individual.fitness
                     pickle.dump(individual.dna, open(f'saves/{env_id.split("/")[-1]}_{individual.fitness}.pkl', 'wb'))

                 # most of the time, next_generation will be an empty list.
                 # TODO: should probably add some logic to clear existing tasks if we don't need
                 # them anymore
                 for child in next_generation:
                     task_manager.add_task(child)

    render_eval = NTimes('CartPole-v1', times=1, render_mode='human')
    render_eval.eval(individual.dna)




    
