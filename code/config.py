from policies import ConvPolicy, LinearPolicy, MultiConv, MemorizationModule, MemorizationModuleWithLR, MemorizationModuleWithLRFull, MemModuleBasic
import torch
from codes import BasicDNA
from population import Sexual, EliteAsexual
from evaluations import MemorizationDataset, NTimes, MNIST
from common import RandomSeedGenerator
from distributed import LocalMultithreaded, LocalSynchronous, DistributedRabbitMQ

experiment_name = 'num_train_datapoints_2'

configs = []


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
trials = 5
kernel_dims = [3,3]
channels = [1,32,64]
strides = [1,1,1]
hidden_size = 128
num_classes = 10
target_fitness = -1e-3 # stop when this is reached

num_train_datapoints_l = [512,2048,'all']
#mutations = ['exponential','normal']
mutations = ['exponential']

sigma_only_generations = 100
max_generation=5000

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
    #for (child_population_size,parent_population_size) in [(16*4,1)]:
    for (child_population_size,parent_population_size) in [(256,1)]:
    #for (child_population_size,parent_population_size) in [(2,1)]:
    #for (child_population_size,parent_population_size) in [(16,1)]:
      #for parent_population_size in [32]:
      #for lr_sigma in [1e-1,5e-2,2e-2,1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4,5e-5,2e-5,1e-5]:
      #for memheads in [1024,1024,1024,1024,1024,1024,1024,1024,1024]:
      #memheads = 64

      #for lr_sigma in [0]:
      for initialization_seed in range(trials):
        #for add_memory_prob in [0.5,0.6]:
        #for sigma in [0.02]:
        #for sigma in [0.01, 0.006, 0.003]:
        #for sigma in [0.1]: 
        for sigma in [0.05]: 
          for mutation in mutations:
            #print("Making config with init seed: ", initialization_seed)
            #for sigma in [0.2,0.1,0.04,0.02,0.01,0.004]:
            #for sigma in [0.2]:
                #num_train_datapoints = 64
            for num_train_datapoints in num_train_datapoints_l:
            #for num_train_datapoints in [10]:

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
                    'sigma': sigma,
                    'mutation': mutation,
                    'device':device,
                    'sigma_mutation': 1.1, # 1 means no mutation
                    'sigma_only_generations': sigma_only_generations,
                    #'sigma_only_generations': -1,

                    }

                #policy_args = {
                #    'sigma': sigma, 
                #    'input_dim': input_dims,
                #    'kernel_dims': kernel_dims,
                #    'channels': channels,
                #    'strides': strides,
                #    'act_dim': num_classes,
                #    'hidden_size': hidden_size,
                #    'initialization_seed': initialization_seed,
                #    'sigma': sigma,
                #    'update_type': 'exponential',
                #    }

                # TODO automate other parts of save-prefix (asexual, etc)
                save_prefix = f'{eval_name}-{loss_type}-asexual-{name}-parent{parent_population_size}-child{child_population_size}-sigma{sigma}-elites{num_elites}-ds{num_train_datapoints}'
                for k,v in policy_args.items():
                    if k in ['kernel_dims', 'strides','channels','act_dim','hidden_size','input_dim', 'device']:
                        continue
                    save_prefix += f'-{k}{v}'

                config = {
                  'num_elites':  num_elites,
                  'parent_population_size': parent_population_size,
                  'child_population_size': child_population_size,
                  'save_prefix': save_prefix,
                  'max_generation': max_generation,
                  #'max_generation': 10,
                  'distributed_class': LocalMultithreaded,
                  'checkpoint_every': 1000,
                  'target_fitness': target_fitness,
                }

                config['eval_args'] = {
                        #'input_dims': input_dims, 
                        #'num_classes': num_classes, 
                        #'batch_size': 'all',
                        'batch_size': 10000,
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
                            'random_seed_generator': RandomSeedGenerator(0),
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
