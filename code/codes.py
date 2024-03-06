from abc import ABC, abstractmethod
import pickle

class DNA(ABC):
    @abstractmethod
    def __init__(self, random_seed):
        pass
    @abstractmethod
    def mutate(self, random_seed):
        pass # returns a NEW DNA object 
    @abstractmethod
    def serialize(self):
        pass
    @abstractmethod
    def deserialize(serialized):
        pass

class BasicDNA:
    ''' 
    Simple genetic code, simply a list of random seeds which will be used
    to generate the policy network (the initial network and then the perturbations).
    '''
    def __init__(self, seeds):
        self.seeds = seeds
    def mutate(self, random_seed):
        child = BasicDNA([])
        child.seeds = self.seeds + [random_seed]
        return child
    def serialize(self):
        return pickle.dumps(tuple(self.seeds))
    def deserialize(serialized):
        seeds = pickle.loads(serialized)
        gc = BasicDNA(seeds)
        return gc
DNA.register(BasicDNA)


