#!python
'''
This file contains base classes and common methods for various types of genetic algorithms.

See codes.py, policies.py, populations.py, and evaluations.py, distributed.py for inheriting classes.
'''

from tensorboard import program

import math
#import matplotlib
import random
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import pickle
from population import EliteAsexual
from distributed import LocalSynchronous, LocalMultithreaded, DistributedRabbitMQ
from policies import LinearPolicy, ConvPolicy
from codes import BasicDNA
import sys
from common import RandomSeedGenerator
from config import make_configs
import time

import multiprocessing as mp
import torch


if __name__ == '__main__':
    print("About to print sys args:", flush=True)
    print(f'sys.argv: {sys.argv}', flush=True)

    experiment_name, configs = make_configs()
    device = configs[0]['eval_args']['device']
    if device == 'cuda':
        mp.set_start_method('spawn')
    
    # launches tensorboard in daemon thread, will stop when this program stops
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', f'./tensorboards/{experiment_name}'])
    url = tb.launch()


    assert sys.argv[1] in ['worker', 'master','m', 'quicktest','q']
    print("after assert", flush=True)
    is_master = (sys.argv[1] in ['master', 'quicktest','q','m'])

    os.makedirs(f'saves/{experiment_name}',exist_ok=True)





    def main(config):
        if sys.argv[1] in ['quicktest','q']:
            config['save_prefix'] = 'quicktest'
            config['distributed_class'] = LocalSynchronous
            config['distributed_args'] = {}

        target_fitness = config['target_fitness']
        best_fitness = (-99999999999999,0) # TODO make more general


        total_training_frames = 0
        tracking_address = f'./tensorboards/{experiment_name}/{config["save_prefix"]}'
        while os.path.exists(tracking_address):
            config['save_prefix'] = config['save_prefix'] + '_' + str(random.randint(0,9999999999999))
            tracking_address = f'./tensorboards/{config["save_prefix"]}'

        os.makedirs(tracking_address, exist_ok=True)
        writer = SummaryWriter(log_dir=tracking_address)

        with config['distributed_class'](config['eval_fac'], config['eval_args'], is_master=is_master, **config['distributed_args']) as task_manager:
            if not is_master:
                print("About to call task_manager.start_worker!", flush=True)
                task_manager.start_worker()
            else:
                print("Creating population",flush=True)
                population = config['population_factory'](**config['population_kwargs'])
                config['population_kwargs'].pop('random_seed_generator') # cannot be pickled

                #population=EliteAsexual(
                #        BasicDNA, 
                #        config['parent_population_size'], 
                #        config['child_population_size'],
                #        RandomSeedGenerator(0),
                #        config['num_elites'],
                #        )

                for child in population.children:
                    task_manager.add_task(child, val=False)

                true_start_time = time.time()
                last_time = time.time()
                start = time.time()
                generation = -config['eval_args']['policy_args']['sigma_only_generations']
                start_generation = generation
                total_frames = 0
                best_val_loss = 99999999
                best_val_acc = 0
                ave_policy_make_time = 0

                for individual, metadata in task_manager.get_task_results():
                     # TODO rename 'individual', 'task_result', etc
                     total_training_frames += metadata['total_frames']
                     if 'val_loss' in metadata:
                         best_val_loss = min(metadata['val_loss'], best_val_loss)
                         best_val_acc = max(metadata['val_acc'], best_val_acc)
                         writer.add_scalar('best_val_loss', best_val_loss, generation)
                         writer.add_scalar('best_val_acc', best_val_acc, generation)
                         print("FOUND VAL LOSS")
                         continue
                     total_frames += metadata['total_frames']
                     ave_policy_make_time += metadata['policy_make_time'] 
                     #print(metadata,flush=True)
                     #print("Average total fps: ", total_training_frames / (time.time() - start), flush=True)
                     #print(datetime.datetime.now(), flush=True)
                     # most of the time, next_generation will be an empty list.
                     next_generation = population.add_grownup(individual)

                     if (individual.fitness[0] == best_fitness[0] and individual.fitness[1] > best_fitness[1]): 
                         best_fitness = individual.fitness
                         best_metadata = metadata

                     if individual.fitness[0] > best_fitness[0]: 
                         best_fitness = individual.fitness
                         best_metadata = metadata
                         best_dna = individual.dna

                     if next_generation:
                         s_per_g = (time.time() - true_start_time) / (generation - start_generation + 1)
                         s_per_i = s_per_g / config['child_population_size']
                         generation += 1
                         # plot best fitness and average fitness
                         ave_fitness = sum([x.fitness[0] for x in population.last_generation_all_grownups])
                         ave_fitness /= len(population.last_generation_all_grownups)
                         print(f"BEST {best_fitness[0]}, AVE {ave_fitness}, generation {generation} complete, {s_per_g}s/g, {s_per_i}s/i")
                         ave_intrinsic_fitness = sum([x.fitness[1] for x in population.last_generation_all_grownups])
                         ave_intrinsic_fitness /= len(population.last_generation_all_grownups)
                         ave_policy_make_time /= len(population.last_generation_all_grownups)
                         writer.add_scalar('ave_policy_make_time', ave_policy_make_time, generation)
                         ave_policy_make_time = 0
                         writer.add_scalar('ave_fitness', ave_fitness, generation)
                         writer.add_scalar('ave_intrinsic_fitness', ave_intrinsic_fitness, generation)
                         writer.add_scalar('best_fitness', best_fitness[0], generation)
                         writer.add_scalar('best_fitness_intrinsic', best_fitness[1] - math.floor(best_fitness[1]), generation)
                         writer.add_scalar('total_frames', total_frames, generation)
                         writer.add_scalar('sigma1', best_metadata['sigma1'], generation)
                         writer.add_scalar('sigma2', best_metadata['sigma2'], generation)
                         writer.add_scalar('generation_elapsed_time', time.time() - last_time, generation)
                         last_time = time.time()

                         # add an eval step 
                         if generation % config['eval_every'] == 0:
                             print("ADDING EVAL STEP")
                             task_manager.add_task(best_dna, val=True)


                         if generation == 0:
                            start = time.time()
                         elapsed_time = time.time() - start
                         if generation > 0:
                             writer.add_scalar('best_fitness_time', best_fitness[0], elapsed_time)
                             writer.add_scalar('ave_fitness_time', ave_fitness, elapsed_time)
                         if generation > config['max_generation'] or individual.fitness[0] >= target_fitness:
                             pickle.dump((individual.dna,config), open(f'saves/{experiment_name}/{config["save_prefix"]}_{individual.fitness[0]}_last.pkl', 'wb'))
                             break

                         if generation % config['checkpoint_every'] == 0:

                             #should be the best performing individual from the generation we just evaled
                             pickle.dump((population.parent_generation[0].dna,config), open(f'saves/{experiment_name}/{config["save_prefix"]}_{population.parent_generation[0].fitness}_gen{generation}.pkl', 'wb'))
                             pickle.dump((best_dna,config), open(f'saves/{experiment_name}/{config["save_prefix"]}_{best_fitness[0]}.pkl', 'wb'))

                     # most of the time, next_generation will be an empty list.
                     # TODO: should probably add some logic to clear existing tasks if we don't need
                     # them anymore
                     for child in next_generation:
                         task_manager.add_task(child)

        #render_eval.eval(individual.dna)

    for config in configs:
        main(config)

        os.system(f'aws s3 cp tensorboards/{experiment_name} s3://luke-genetics/{experiment_name} --recursive')
        os.system(f'aws s3 cp saves/{experiment_name} s3://luke-genetics/{experiment_name} --recursive')





    
