from common import Individual
import random
import torch
from policies import LinearPolicy, ConvPolicy
from abc import ABC, abstractmethod
import time
import gymnasium as gym

class EvaluationMethod(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def eval(self, dna):
        pass

# given an environment observation_space.shape or action_space.shape, 
# returns the total number of parameters in the flattened space
from functools import reduce
def flattened_shape(shape):
    return reduce(lambda x,y: x*y, shape, 1)  # compute product


class NTimes(EvaluationMethod):
    def __init__(self, env_id, policy_network_class, times=1, render_mode=None):
        self.env_id = env_id
        self.render_mode=render_mode
        self.times=times
        self.policy_network_class = policy_network_class
    def eval(self, dna):
        t1 = time.time()
        env = gym.make(self.env_id, render_mode=self.render_mode)
        t2 = time.time()
        action_shape  = env.action_space.n
        # TODO make this less hacky
        if self.policy_network_class is LinearPolicy:
            state_shape = flattened_shape(env.observation_space.shape)
            policy_network = self.policy_network_class(dna, state_shape, 100, action_shape)
        elif self.policy_network_class is ConvPolicy:
            state_shape = env.observation_space.shape
            #kernels = [8,4,3]
            #channels = [3,16,32,32]
            #strides = [4,2,1]
            #hidden_size = 256

            kernels = [8,4,3]
            channels = [3,32,64,64]
            strides = [4,2,1]
            hidden_size = 512

            #kernels = [8,4]
            #channels = [3,8,16]
            #strides = [4,2]
            #hidden_size = 128
            policy_network = self.policy_network_class(dna, state_shape, kernels, channels, strides, action_shape, hidden_size)
        else:
            assert False, "unsupported policy class type {policy_network}"
        t3 = time.time()
        state, _ = env.reset()
        frames = 0
        total_reward = 0
        for i in range(self.times):
            step = 0
            while True:
                step += 1
                action = policy_network.act(state)
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                #print(f'step: {step}, terminated: {terminated}, truncated: {truncated}, info: {info}')
                if terminated or truncated: # or step > 500:
                    state, _ = env.reset()
                    break
            frames += step
        t4 = time.time()
        metadata = {
          'env_make_time': t2 - t1,
          'policy_make_time': t3-t2,
          'episode_time_average': (t4-t3) / self.times,
          'frame_time_average': (t4-t3) / frames,
          'total_frames':  frames,
          }
          

        
        return Individual(dna, total_reward / self.times), metadata

EvaluationMethod.register(NTimes)

class MemorizationDataset(EvaluationMethod):
    def __init__(self, input_dims, num_classes, batch_size, num_train_datapoints, num_val_datapoints, sigma, loss_type='num_incorrect'):

        self.sigma = sigma
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.batch_size =  batch_size

        self.num_train_batches = num_train_datapoints // batch_size
        self.num_val_batches = num_val_datapoints // batch_size

        self.train_dataloader  = RandomDataloader(input_dims, num_classes, num_train_datapoints, batch_size)
        self.val_dataloader  = RandomDataloader(input_dims, num_classes, num_val_datapoints, batch_size)
        self.loss_type = loss_type
        assert loss_type in ['num_incorrect', 'cross_entropy']

    def eval(self, dna, cached_policy=None):

        kernels = [8,4,3]
        channels = [3,32,64,64]
        strides = [4,2,1]
        hidden_size = 512

        if cached_policy:
            policy_network = cached_policy.update_dna(dna)
        else:
            policy_network = ConvPolicy(dna, self.input_dims, kernels, channels, strides, self.num_classes, hidden_size, sigma=self.sigma)

        train_loss = 0
        val_loss = 0

        for i,(x,y) in enumerate(self.train_dataloader):
            #print(f'i: {i}, x: {x}, y: {y}')
            r = policy_network(x)
            if self.loss_type == 'cross_entropy':
                loss = torch.nn.functional.cross_entropy(r, y)
            elif self.loss_type == 'num_incorrect':
                #print('r:', r)
                #print('argmax:', torch.argmax(r, dim=1))
                #print('y:', y)
                loss = torch.sum(torch.ne(torch.argmax(r, dim=1), y))
            train_loss += loss / self.num_train_batches

        for i,(x,y) in enumerate(self.val_dataloader):
            r = policy_network(x)
            if self.loss_type == 'cross_entropy':
                loss = torch.nn.functional.cross_entropy(r, y)
            elif self.loss_type == 'num_incorrect':
                loss = torch.sum(torch.ne(torch.argmax(r, dim=1), y))
            val_loss += loss / self.num_val_batches

        metadata = {
            'train_loss': train_loss.item(),
            'val_loss': val_loss.item(),
            'total_frames': self.num_train_batches * self.batch_size,
            'policy_make_time': policy_network.metadata['policy_make_time'],
            }       
        return (Individual(dna, -1*train_loss.item()), metadata), policy_network

        


class RandomDataloader():
    def __init__(self, input_dims, num_classes, num_datapoints, batch_size, seed=0, shuffle=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.num_batches = num_datapoints // batch_size
        self.total_datapoints = num_datapoints
        self.shuffle = shuffle
        self.seed = seed


    def __iter__(self):
        # Can't make generator in constructor since torch.generator cannot be pickled 
        #print("Making dataloader")
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if not self.shuffle:
            for i in range(self.num_batches):
                yield [torch.randn([self.batch_size, *self.input_dims], generator=generator), torch.randint(high=self.num_classes, size=[self.batch_size], generator=generator)]
        else:
            xl = []
            yl = []
            for i in range(self.num_batches):
                xl.append(torch.randn([self.batch_size, *self.input_dims], generator=generator))
                yl.append(torch.randint(high=self.num_classes, size=[self.batch_size], generator=generator))
            xl = torch.cat(xl, dim=0)
            yl = torch.cat(yl, dim=0)
            indices = torch.randperm(xl.shape[0])
            #print("Done")
            for i in range(self.num_batches):
             
                x = xl[indices[i*self.batch_size:i*self.batch_size + self.batch_size]]
                y = yl[indices[i*self.batch_size:i*self.batch_size + self.batch_size]]
                yield [x,y]
            


class DumbDataloader():
    def __init__(self, input_dims, num_classes, num_batches, batch_size):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.num_batches = num_batches
        self.total_datapoints = batch_size * num_batches


    def __iter__(self):
        # Can't make generator in constructor since torch.generator cannot be pickled 
        for i in range(self.num_batches):
            labels = torch.arange(self.batch_size) % self.num_classes
            datapoints = labels[:,None,None,None] * torch.ones([1, *self.input_dims])
            yield [datapoints, labels]
