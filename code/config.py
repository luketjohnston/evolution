from policies import ConvPolicy, LinearPolicy, MultiConv, MemorizationModule, MemorizationModuleWithLR, MemorizationModuleWithLRFull, MemModuleBasic
import torch
from codes import BasicDNA
from population import Sexual, EliteAsexual
from evaluations import MemorizationDataset, NTimes
from common import RandomSeedGenerator
from distributed import LocalMultithreaded, LocalSynchronous, DistributedRabbitMQ

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
factory,name = (MemorizationModule, 'memApr23remove')
factory,name = (LinearPolicy, 'memLinear')


trials = 4
num_elites=0
input_dims=[128]
num_classes = 10
def make_configs():
    configs = []
    for (child_population_size,parent_population_size) in [(64,16)]:
      #for parent_population_size in [32]:
      #for lr_sigma in [1e-1,5e-2,2e-2,1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4,5e-5,2e-5,1e-5]:
      #for memheads in [1024,1024,1024,1024,1024,1024,1024,1024,1024]:
      for memheads in [64]:
        #for lr_sigma in [0]:
        for initialization_seed in range(trials):
            print("Making config with init seed: ", initialization_seed)
            for sigma in [1.0]:
                num_train_datapoints = 64

                #loss_type='num_incorrect'
                loss_type='num_till_death'

                #policy_args = {
                #    'sigma': sigma, 
                #    'heads': memheads,
                #    'input_dim': input_dims,
                #    'act_dim': num_classes,
                #    'initialization_seed': initialization_seed,
                #    'sigma': sigma,
                #    'add_memory_prob': 0.4,
                #    'remove_memory_prob': 0.3,
                policy_args = {
                    'sigma': sigma, 
                    'input_dim': 128,
                    'act_dim': num_classes,
                    'hidden_dim': 128,
                    'initialization_seed': initialization_seed,
                    'sigma': sigma,
                    }

                # TODO automate other parts of save-prefix (asexual, etc)
                save_prefix = f'{loss_type}-asexual-{name}-parent{parent_population_size}-child{child_population_size}-sigma{sigma}-elites{num_elites}-ds{num_train_datapoints}'
                for k,v in policy_args.items():
                    save_prefix += f'-{k}{v}'

                config = {
                  'num_elites':  num_elites,
                  'parent_population_size': parent_population_size,
                  'child_population_size': child_population_size,
                  'save_prefix': save_prefix,
                  #'save_prefix': f'quicktest',
                  #'save_prefix': f'quick_new_imp2_initseed1',
                  #'distributed_class': LocalSynchronous,
                  'max_generation': 20000,
                  'distributed_class': LocalMultithreaded,
                  'checkpoint_every': 100,
                }

                config['eval_method'] = MemorizationDataset(
                        input_dims=input_dims, 
                        num_classes=num_classes, 
                        batch_size=min(32, num_train_datapoints // 2),
                        #num_train_datapoints=512,
                        num_train_datapoints=num_train_datapoints,
                        num_val_datapoints=0,
                        policy_factory=factory,
                        # somehow make it so I don't have to do this garbage
                        #policy_args={'sigma': sigma, 'lr_sigma': lr_sigma},
                        #policy_args={'memheads': memheads},
                        policy_args=policy_args,
                        #policy_args={'sigma': sigma},
                        loss_type=loss_type,
                        seed=initialization_seed,
                        #loss_type='cross_entropy',
                        )
                config['population'] = EliteAsexual(
                #config['population'] = Sexual(
                            BasicDNA, 
                            config['parent_population_size'], 
                            config['child_population_size'],
                            RandomSeedGenerator(0),
                            config['num_elites'],
                            )
                        
                configs.append(config)
    return configs
            
            
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
