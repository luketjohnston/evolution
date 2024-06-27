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
from evolution.src.population import EliteAsexual
from evolution.src.distributed import LocalSynchronous, LocalMultithreaded, DistributedRabbitMQ
from evolution.src.policies import LinearPolicy, ConvPolicy
import sys
from evolution.src.common import RandomSeedGenerator, Fitness
from evolution.src.config import make_configs
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
    #if device == 'mps':
    #    mp.set_start_method('fork')
    
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
        best_fitness = Fitness(-99999999999999,0) # TODO make more general


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
                last_checkpoint = -999999999
                val_didnt_improve = 0

                for (individual, metadata),policy in task_manager.get_task_results():

                    # TODO rename 'individual', 'task_result', etc
                    total_training_frames += metadata['total_frames']
                    if 'val_loss' in metadata: # means policy is not None also

                        if generation - last_checkpoint > config['checkpoint_every']:
                            torch.save((individual,policy,config), open(f'saves/{experiment_name}/{config["save_prefix"]}_val{metadata["val_loss"]}_gen{generation}.pt', 'wb'))
                            last_checkpoint = generation


                        if metadata['val_loss'] >= best_val_loss:
                            val_didnt_improve += 1
                        else:
                            val_didnt_improve = 0

                        best_val_loss = min(metadata['val_loss'], best_val_loss)
                        best_val_acc = max(metadata['val_acc'], best_val_acc)
                        writer.add_scalar('best_val_loss', best_val_loss, generation)
                        writer.add_scalar('best_val_acc', best_val_acc, generation)



                        if val_didnt_improve > 10:
                            print("Failed to improve val loss for 10 evals, terminating...")
                            break

                        if generation > config['max_generation'] or metadata['val_fitness'] >= target_fitness:
                            print("Reached target fitness or max generation, terminating...")
                            break
                        continue
                    total_frames += metadata['total_frames']
                    ave_policy_make_time += metadata['policy_make_time'] 

                    #print(metadata,flush=True)
                    #print("Average total fps: ", total_training_frames / (time.time() - start), flush=True)
                    #print(datetime.datetime.now(), flush=True)
                    # most of the time, next_generation will be an empty list.
                    next_generation = population.add_grownup(individual)

                    if (metadata['train_fitness'] == best_fitness.base and metadata['train_intrinsic_fitness'] > best_fitness.intrinsic): 
                        best_fitness = Fitness(metadata['train_fitness'], metadata['train_intrinsic_fitness'])
                        best_metadata = metadata

                    if metadata['train_fitness'] > best_fitness.base: 
                        best_fitness = Fitness(metadata['train_fitness'], metadata['train_intrinsic_fitness'])
                        best_metadata = metadata
                        best_dna = individual.dna

                    if next_generation:
                        s_per_g = (time.time() - true_start_time) / (generation - start_generation + 1)
                        s_per_i = s_per_g / config['child_population_size']
                        generation += 1
                        # plot best fitness and average fitness
                        ave_fitness = sum([x.fitness.base for x in population.last_generation_all_grownups])
                        ave_fitness /= len(population.last_generation_all_grownups)
                        print(f"BEST {best_fitness.base}, AVE {ave_fitness}, generation {generation} complete, {s_per_g}s/g, {s_per_i}s/i")
                        ave_intrinsic_fitness = sum([x.fitness.base for x in population.last_generation_all_grownups])
                        ave_intrinsic_fitness /= len(population.last_generation_all_grownups)
                        ave_policy_make_time /= len(population.last_generation_all_grownups)
                        writer.add_scalar('ave_policy_make_time', ave_policy_make_time, generation)
                        ave_policy_make_time = 0
                        writer.add_scalar('ave_fitness', ave_fitness, generation)
                        writer.add_scalar('ave_intrinsic_fitness', ave_intrinsic_fitness, generation)
                        writer.add_scalar('best_fitness', best_fitness.base, generation)
                        writer.add_scalar('best_fitness_intrinsic', best_fitness.intrinsic, generation)
                        writer.add_scalar('total_frames', total_frames, generation)
                        writer.add_scalar('sigma1', best_metadata['sigma1'], generation)
                        writer.add_scalar('sigma2', best_metadata['sigma2'], generation)
                        writer.add_scalar('generation_elapsed_time', time.time() - last_time, generation)
                        last_time = time.time()

                        # add an eval step 
                        if generation % config['eval_every'] == 0:
                            print("ADDING EVAL STEP")
                            task_manager.add_task(best_dna, val=True, ret_policy=True) # TODO use ret_policy
                            #task_manager.add_task(best_dna, val=True) # TODO use ret_policy


                        if generation == 0:
                           start = time.time()
                        elapsed_time = time.time() - start
                        if generation > 0:
                            writer.add_scalar('best_fitness_time', best_fitness.base, elapsed_time)
                            writer.add_scalar('ave_fitness_time', ave_fitness, elapsed_time)

                        if generation > config['max_generation'] or individual.fitness.base >= target_fitness:
                            task_manager.add_task(best_dna, val=True, ret_policy=True) # TODO use ret_policy


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





    
