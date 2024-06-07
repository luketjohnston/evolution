from policies import ConvPolicy, LinearPolicy, MultiConv, MemorizationModule, MemorizationModuleWithLR, MemorizationModuleWithLRFull, MemModuleBasic
from itertools import product
import torch
from codes import BasicDNA
import random
from population import Sexual, EliteAsexual
from evaluations import MemorizationDataset, NTimes, MNIST
from common import RandomSeedGenerator
from distributed import LocalMultithreaded, LocalSynchronous, DistributedRabbitMQ

#experiment_name = 'num_train_datapoints_2'
#experiment_name = 'normal_faster_maybe'
#experiment_name = 'normal_with_starting_002'

#experiment_name = 'normal_hyperparam_search1'
#experiment_name = 'test_exp_sigma1start_1'
#experiment_name = 'print1'
experiment_name = 'test'

configs = []

pool_size=None
eval_every=200


popsizes = [(256,1)]
#popsizes = [(3,1)]

#input_dims=[64,64,3]
#input_dims=[128]
#kernels = [8,4,3]
#channels = [3,32,64,64]
#strides = [4,2,1]
#hidden_size = 512
#num_classes = 10

#factory,name = (mem_factory, 'memmodule')
#factory,name = (mem_factory_lr, 'mmemmodulelr')
#factory,name = (mem_factory_lr_full, 'memlrfull')
#factory,name = (mem_factory_basic, 'membasic_updated1')


if torch.cuda.is_available():
    device='cuda'
elif torch.backends.mps.is_available():
    device='mps'
else:
    device='cpu'

print(f"Using device {device}")


#factory,name = (ConvPolicy, 'conv')
factory,name = (LinearPolicy, 'mlp')
eval_factory,eval_name=(MNIST,'mnist')
input_dims=[28,28,1] # mnist
trials = 1
kernel_dims = [3,3]
channels = [1,32,64]
strides = [1,1,1]
hidden_size = 128
num_classes = 10
target_fitness = -1e-3 # stop when this is reached

#num_train_datapoints_l = [512,2048,'all']
num_train_datapoints_l = [500]
mutations = ['normal']
#mutations = ['one']
#sigma_mutations = [1.1] # 1 means no mutation
#sigma_mutations = [1.05] # 1 means no mutation
sigma_mutations = [1.0] # 1 means no mutation

batch_sizes=[10000]

#sigma_only_generations = 100
# NOTE that the base sigma still cannot be too high or else it will not be able to even find
# which direction it needs to mutate in. May  be worth investigating which layer this effect
# is more important for
#sigma_only_generations = 200
#max_generation=2000

#sigma_l = [0.005, 0.01, 0.02]
#sigma_l = [0.03] # normal distribution usually equalizes at higher than 0.03 in sigma-only phase

#sigma_l = [(7e-3,6e-3), (2e-3,1e-3), (1e-3,4e-5)] # Can start both exponential and normal at 0.02
# Can start both exponential and normal at 0.02
sigma_l = [(0.001,0.001)] 

sigma_only_generations_l = [-1]
#sigma_only_generations_l = [1]
max_generation=500000

#sigma_only_generations = 0
#trials=1
#max_generation=10


#factory,name = (MemorizationModule, 'memApr23remove')
#eval_factory,eval_name=(MemorizationDataset,'memds')
#trials = 20
#input_dims=[128]



num_elites=0
num_classes = 10
def make_configs():
    configs = []

    for num_train_datapoints, popsize, trial, batch_size, sigma_only_generations, sigma_mutation, sigma, mutation in product(
            num_train_datapoints_l, popsizes, range(trials), batch_sizes, sigma_only_generations_l, sigma_mutations, sigma_l, mutations):

      child_population_size, parent_population_size = popsize

      initialization_seed = random.randint(0,9999999)
      pop_init_seed = initialization_seed

      # TODO any way to scale these the same way?
      #if mutation == 'exponential':
      #  sigma = 0.2
      #elif mutation == 'normal':
      #  sigma = 0.03
      #elif mutation == 'uniform':
      #  sigma = 0.1
      #elif mutation == 'one':
      #  sigma = 0.05


      #loss_type='num_incorrect'
      loss_type='cross_entropy'
      #loss_type='num_till_death'

      #policy_args = {
      #    'sigma': sigma, 
      #    'heads': memheads,
      #    'input_dim': input_dims,
      #    'act_dim': num_classes,
      #    'initialization_seed': initialization_seed,
      #    'sigma': sigma,
      #    'add_memory_prob': add_memory_prob,
      #    'remove_memory_prob': 0.2,
      #    }

      policy_args = {
          'sigma': sigma, 
          'input_dim': input_dims[0] * input_dims[1],
          'act_dim': num_classes,
          'hidden_dim': 128,
          'initialization_seed': initialization_seed,
          'mutation': mutation,
          'device':device,
          'sigma_mutation': sigma_mutation, # 1 means no mutation
          'sigma_only_generations': sigma_only_generations,

          }

      # TODO automate other parts of save-prefix (asexual, etc)
      save_prefix = f'{eval_name}-{loss_type}-asex-{name}-p{parent_population_size}-c{child_population_size}-e{num_elites}-ds{num_train_datapoints}-t{trial}-pis{pop_init_seed}-s1{sigma[0]}-s2{sigma[1]}'
      for k,v in policy_args.items():
          if k in ['kernel_dims', 'strides','channels','act_dim','hidden_size','input_dim', 'device', 'sigma']:
              continue
          save_prefix += f'-{k}{v}'

      config = {
        'num_elites':  num_elites,
        'parent_population_size': parent_population_size,
        'child_population_size': child_population_size,
        'save_prefix': save_prefix,
        'max_generation': max_generation,
        'distributed_class': LocalMultithreaded,
        'checkpoint_every': 1000,
        'target_fitness': target_fitness,
        'eval_every': eval_every,
      }
      config['distributed_args'] = {
        'pool_size': pool_size
      }

      config['eval_args'] = {
              #'input_dims': input_dims, 
              #'num_classes': num_classes, 
              #'batch_size': 'all',
              'batch_size': batch_size,
              'num_train_datapoints': num_train_datapoints,
              'policy_factory': factory,
              'policy_args': policy_args,
              'loss_type': loss_type,
              'device':device,
              #'seed': initialization_seed,
              }

      config['eval_fac'] = eval_factory

      config['population_factory'] = EliteAsexual
      config['population_kwargs'] = {
                  'dna_class': BasicDNA, 
                  'parent_population_size': config['parent_population_size'], 
                  'child_population_size': config['child_population_size'],
                  'random_seed_generator': RandomSeedGenerator(pop_init_seed),
                  'num_elites': config['num_elites'],
                  }


              
      configs.append(config)
    return (experiment_name, configs)
            
            
            #adam gets to 1e-5 within ~100 epochs on MemorizationDataset

  #'env_id': 'ALE/Breakout-v5',
  #'env_id': 'CartPole-v1',

"""
'''Quick config for CartPole'''
config = {
  'num_elites':  1,
  'parent_population_size': 2,
  'child_population_size': 4,
  'env_id': 'CartPole-v1',
  'save_prefix': 'quicktest',
  'distributed_class': LocalMultithreaded,
}
config['eval_method'] = NTimes(config['env_id'], policy_network_class=LinearPolicy, times=1)
"""


"""
''' Below is the config for the first experiment on Frostbite that I ran '''
config = {
  'num_elites':  1,
  'parent_population_size': 32,
  'child_population_size': 128,
  'env_id': 'ALE/Frostbite-v5',
  'save_prefix': 'Frostbite-v5',
  'distributed_class': DistributedRabbitMQ
}
config['eval_method'] = NTimes(config['env_id'], policy_network_class=ConvPolicy, times=)
"""
