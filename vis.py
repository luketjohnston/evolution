#!python3
import gymnasium as gym
import pickle
from common import Individual
from evaluations import NTimes
from policies import ConvPolicy


#env = gym.make('CartPole-v1', render_mode='human')
env_id = 'ALE/Breakout-v5'
env = gym.make(env_id, render_mode='human')


render_eval = NTimes(env_id, times=1, render_mode='human')

env.reset()
dna = pickle.load(open('saves/Breakout-v5_8.4.pkl', 'rb'))
#dna = pickle.load(open('saves/Breakout-v5_3.3.pkl', 'rb'))
#dna = pickle.load(open('saves/Breakout-v5_2.6.pkl', 'rb'))
#dna = pickle.load(open('saves/Breakout-v5_2.1.pkl', 'rb'))
#dna = pickle.load(open('saves/Breakout-v5_0.3.pkl', 'rb'))

render_eval.eval(dna, ConvPolicy)


#while True:
#  i += 1
#  state, reward, terminated, truncated, info = env.step(env.action_space.sample())
#  env.render()
#  print(i)
#  if terminated or truncated: break
    
