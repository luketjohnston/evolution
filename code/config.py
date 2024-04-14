from policies import ConvPolicy, LinearPolicy, MultiConv, MemorizationModule, MemorizationModuleWithLR, MemorizationModuleWithLRFull, MemModuleBasic
import torch
from codes import BasicDNA
from population import Sexual, EliteAsexual
from evaluations import MemorizationDataset, NTimes
from common import RandomSeedGenerator
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
#config['eval_method'] = NTimes('ALE/Frostbite-v5', policy_network_class=ConvPolicy, times=1)


input_dims=[64,64,3]
kernels = [8,4,3]
channels = [3,32,64,64]
strides = [4,2,1]
hidden_size = 512
num_classes = 10

def conv_factory(dna, sigma):
    return ConvPolicy(dna, input_dims, kernels, channels, strides, num_classes, hidden_size, initialization_seed, sigma)

multi=4
memheads=64
proj_dim=128
initialization_seed=9

def multi_conv_factory(dna, sigma):
    return MultiConv(dna, input_dims, kernels, channels, strides, num_classes, hidden_size, initialization_seed=initialization_seed, sigma=sigma, multi=multi)

def mem_factory(dna, sigma):
    return MemorizationModule(dna, input_dims, num_classes, memheads, initialization_seed=initialization_seed, sigma=1,proj_dim=proj_dim)

def mem_factory_lr(dna, sigma, lr_sigma):
    return MemorizationModuleWithLR(dna, input_dims, num_classes, memheads, initialization_seed=initialization_seed, sigma=1,proj_dim=proj_dim, lr_sigma=lr_sigma)

def mem_factory_lr_full(dna, sigma, lr_sigma):
    return MemorizationModuleWithLRFull(dna, input_dims, num_classes, memheads, initialization_seed=initialization_seed, sigma=1,proj_dim=proj_dim, lr_sigma=lr_sigma)

def mem_factory_basic(dna):
    return MemModuleBasic(dna, input_dims, num_classes, memheads, initialization_seed=initialization_seed, proj_dim=proj_dim)

#factory,name = (mem_factory, 'memmodule')
#factory,name = (mem_factory_lr, 'mmemmodulelr')
#factory,name = (mem_factory_lr_full, 'memlrfull')
factory,name = (mem_factory_basic, 'membasic_plus3')


num_elites=0
def make_configs():
    configs = []
    for (child_population_size,parent_population_size) in [(64,16)]:
        #for parent_population_size in [32]:
        #for lr_sigma in [1e-1,5e-2,2e-2,1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4,5e-5,2e-5,1e-5]:
        #for lr_sigma in [0]:
            #for sigma in [1.0]:
                num_train_datapoints = 64

                loss_type='num_incorrect'
                #loss_type='num_till_death'

                config = {
                  'num_elites':  num_elites,
                  'parent_population_size': parent_population_size,
                  'child_population_size': child_population_size,
                  #'save_prefix': f'memfast-sexualv1-argmax-parent{parent_population_size}-child{child_population_size}-sigma{sigma}-elites{num_elites}-ds{num_train_datapoints}',
                  #'save_prefix': f'{loss_type}-asexual-{name}-heads{memheads}-lrsigma{lr_sigma}-proj{proj_dim}-parent{parent_population_size}-child{child_population_size}-sigma{sigma}-elites{num_elites}-ds{num_train_datapoints}-initseeed{initialization_seed}',
                  'save_prefix': f'{loss_type}-asexual-{name}-heads{memheads}-proj{proj_dim}-parent{parent_population_size}-child{child_population_size}-elites{num_elites}-ds{num_train_datapoints}-initseeed{initialization_seed}',
                  #'save_prefix': f'quicktest',
                  #'save_prefix': f'quick_new_imp2_initseed1',
                  #'distributed_class': LocalSynchronous,
                  'max_generation': 100000,
                  'distributed_class': LocalMultithreaded,
                }
                config['eval_method'] = MemorizationDataset(
                        input_dims=input_dims, 
                        num_classes=num_classes, 
                        batch_size=min(32, num_train_datapoints // 2),
                        #num_train_datapoints=512,
                        num_train_datapoints=num_train_datapoints,
                        num_val_datapoints=32,
                        policy_factory=factory,
                        #policy_args={'sigma': sigma, 'lr_sigma': lr_sigma},
                        policy_args={},
                        #policy_args={'sigma': sigma},
                        loss_type=loss_type,
                        #loss_type='cross_entropy',
                        )
                #config['population'] = EliteAsexual(
                config['population'] = Sexual(
                            BasicDNA, 
                            config['parent_population_size'], 
                            config['child_population_size'],
                            RandomSeedGenerator(0),
                #            config['num_elites'],
                            )
                        
                configs.append(config)
    return configs
            
            
            #adam gets to 1e-5 within ~100 epochs on MemorizationDataset
