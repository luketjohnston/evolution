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




''' Below is the config for an experiment to just memorize a randn dataset. '''
config = {
  'num_elites':  1,
  'parent_population_size': 32,
  #'parent_population_size': 2,
  'child_population_size': 128,
  #'child_population_size': 4,
  'save_prefix': 'mem-e1-p32-c128-d1028',
  #'distributed_class': LocalSynchronous,
  'distributed_class': LocalMultithreaded,
}
config['eval_method'] = MemorizationDataset(
        input_dims=[64,64,3], 
        num_classes=10, 
        batch_size=32,
        num_datapoints=1028,
        val_frac=0.1,
        sigma=0.002)
#config['eval_method'] = NTimes('ALE/Frostbite-v5', policy_network_class=ConvPolicy, times=1)


