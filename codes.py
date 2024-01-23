from abc import ABC, abstractmethod

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
    TODO move to codes.py
    '''
    def __init__(self, random_seed):
        self.seeds = [random_seed]
    def mutate(self, random_seed):
        child = BasicDNA(0)
        child.seeds = self.seeds + [random_seed]
        return child
    def serialize(self):
        return tuple(self.seeds)
    def deserialize(serialized):
        gc = BasicDNA(serialized[0])
        gc.seeds = list(serialized)
        return gc
DNA.register(BasicDNA)


