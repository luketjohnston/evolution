#!python
'''
This file contains base classes and common methods for various types of genetic algorithms.

See codes.py, policies.py, populations.py, and evaluations.py, distributed.py for inheriting classes.
'''

from tensorboard import program

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

    
    # launches tensorboard in daemon thread, will stop when this program stops
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', './tensorboards'])
    url = tb.launch()


    assert sys.argv[1] in ['worker', 'master','m', 'quicktest','q']
    print("after assert", flush=True)
    is_master = (sys.argv[1] in ['master', 'quicktest','q','m'])

    os.makedirs('saves',exist_ok=True)





    def main(config):
        if sys.argv[1] in ['quicktest','q']:
            config['save_prefix'] = 'quicktest'
            config['distributed_class'] = LocalSynchronous

        target_fitness = 0
        best_fitness = -99999999999999


        total_training_frames = 0
        tracking_address = f'./tensorboards/{config["save_prefix"]}'
        while os.path.exists(tracking_address):
            config['save_prefix'] = config['save_prefix'] + '_' + str(random.randint(0,9999999999999))
            tracking_address = f'./tensorboards/{config["save_prefix"]}'

        os.makedirs(tracking_address, exist_ok=True)
        writer = SummaryWriter(log_dir=tracking_address)

        with config['distributed_class'](config['eval_method'], is_master=is_master) as task_manager:
            if not is_master:
                print("About to call task_manager.start_worker!", flush=True)
                task_manager.start_worker()
            else:
                print("Creating population",flush=True)
                population = EliteAsexual(
                        BasicDNA, 
                        config['parent_population_size'], 
                        config['child_population_size'],
                        RandomSeedGenerator(0),
                        config['num_elites'],
                        )

                for child in population.children:
                    task_manager.add_task(child)

                start = time.time()
                generation = 0
                total_frames = 0
                best_val_loss = 99999999
                ave_policy_make_time = 0

                for individual, metadata in task_manager.get_task_results():
                     # TODO rename 'individual', 'task_result', etc
                     total_training_frames += metadata['total_frames']
                     if 'val_loss' in metadata:
                         best_val_loss = min(metadata['val_loss'], best_val_loss)
                         #print(f'best_val_loss: {best_val_loss}, val_loss: {metadata["val_loss"]}')
                     total_frames += metadata['total_frames']
                     ave_policy_make_time += metadata['policy_make_time'] 
                     #print(metadata,flush=True)
                     #print("Average total fps: ", total_training_frames / (time.time() - start), flush=True)
                     #print(datetime.datetime.now(), flush=True)
                     # most of the time, next_generation will be an empty list.
                     next_generation = population.add_grownup(individual)
                     if individual.fitness[0] >= target_fitness:
                         break
                     if individual.fitness[0] > best_fitness:
                         best_fitness = individual.fitness[0]
                         pickle.dump(individual.dna, open(f'saves/{config["save_prefix"]}_{individual.fitness}.pkl', 'wb'))
                     if next_generation:
                         generation += 1
                         # plot best fitness and average fitness
                         ave_fitness = sum([x.fitness[0] for x in population.last_generation_all_grownups])
                         ave_intrinsic_fitness = sum([x.fitness[1] for x in population.last_generation_all_grownups])
                         ave_fitness /= len(population.last_generation_all_grownups)
                         ave_intrinsic_fitness /= len(population.last_generation_all_grownups)
                         ave_policy_make_time /= len(population.last_generation_all_grownups)
                         writer.add_scalar('ave_policy_make_time', ave_policy_make_time, generation)
                         ave_policy_make_time = 0
                         writer.add_scalar('ave_fitness', ave_fitness, generation)
                         writer.add_scalar('ave_intrinsic_fitness', ave_intrinsic_fitness, generation)
                         writer.add_scalar('best_fitness', best_fitness, generation)
                         writer.add_scalar('total_frames', total_frames, generation)
                         elapsed_time = time.time() - start
                         writer.add_scalar('best_fitness_time', best_fitness, elapsed_time)
                         writer.add_scalar('ave_fitness_time', ave_fitness, elapsed_time)
                         if generation > config['max_generation']:
                             pickle.dump(individual.dna, open(f'saves/{config["save_prefix"]}_{individual.fitness}_last.pkl', 'wb'))
                             break

                     if generation % config['checkpoint_every'] == 0:
                         pickle.dump(individual.dna, open(f'saves/{config["save_prefix"]}_{individual.fitness}_gen{generation}.pkl', 'wb'))



                     # most of the time, next_generation will be an empty list.
                     # TODO: should probably add some logic to clear existing tasks if we don't need
                     # them anymore
                     for child in next_generation:
                         task_manager.add_task(child)

        #render_eval.eval(individual.dna)

    for config in make_configs():
        main(config)



    
