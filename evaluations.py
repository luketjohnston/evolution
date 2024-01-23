from common import Individual
from abc import ABC, abstractmethod
import gymnasium as gym

class EvaluationMethod(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def eval(self, dna, policy_network_class):
        pass

# given an environment observation_space.shape or action_space.shape, 
# returns the total number of parameters in the flattened space
from functools import reduce
def flattened_shape(shape):
    return reduce(lambda x,y: x*y, shape, 1)  # compute product
    

class NTimes(EvaluationMethod):
    def __init__(self, env_id, times=1, render_mode=None):
        self.env_id = env_id
        self.render_mode=render_mode
        self.times=times
    def eval(self, dna, policy_network_class):
        env = gym.make(self.env_id, render_mode=self.render_mode)
        state_shape = flattened_shape(env.observation_space.shape)
        action_shape  = env.action_space.n
        policy_network = policy_network_class(dna, state_shape, 100, action_shape)
        state, _ = env.reset()
        step = 0
        total_reward = 0
        for i in range(self.times):
            while True:
                step += 1
                action = policy_network.act(state)
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    state, _ = env.reset()
                    break
        return Individual(dna, total_reward / self.times)

EvaluationMethod.register(NTimes)
