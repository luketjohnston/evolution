from policies import ConvPolicy, LinearPolicy
import torch
from evaluations import MemorizationDataset, NTimes
from distributed import LocalMultithreaded, LocalSynchronous, DistributedRabbitMQ



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

configs = []



''' Below is the config for an experiment to just memorize a randn dataset. '''
config = {
  'num_elites':  1,
  'parent_population_size': 32,
  #'parent_population_size': 2,
  'child_population_size': 128,
  #'child_population_size': 4,
  'save_prefix': 'mem-e1-p32-c128-td64-s0p01',
  #'save_prefix': 'quicktest2',
  #'save_prefix': 'quicktest',
  'distributed_class': LocalSynchronous,
  'max_generation': 200,
  #'distributed_class': LocalMultithreaded,
}
config['eval_method'] = MemorizationDataset(
        input_dims=[64,64,3], 
        num_classes=10, 
        batch_size=32,
        #num_train_datapoints=512,
        num_train_datapoints=64,
        num_val_datapoints=32,
        sigma=0.01)


#config['eval_method'] = NTimes('ALE/Frostbite-v5', policy_network_class=ConvPolicy, times=1)


def make_configs():
    configs = []
    for child_population_size in [64]:
        for parent_population_size in [32]:
            for sigma in [0.02]:
                config = {
                  'num_elites':  1,
                  'parent_population_size': parent_population_size,
                  'child_population_size': child_population_size,
                  'save_prefix': f'memfast-parent{parent_population_size}-child{child_population_size}-sigma{sigma}',
                  #'save_prefix': f'quicktest2',
                  #'distributed_class': LocalSynchronous,
                  'max_generation': 500,
                  'distributed_class': LocalMultithreaded,
                }
                config['eval_method'] = MemorizationDataset(
                        input_dims=[64,64,3], 
                        num_classes=10, 
                        batch_size=32,
                        #num_train_datapoints=512,
                        num_train_datapoints=64,
                        num_val_datapoints=32,
                        sigma=sigma,
                        loss_type='cross_entropy')
                        
                configs.append(config)
    return configs
            
            
            #adam gets to 1e-5 within ~100 epochs on MemorizationDataset
