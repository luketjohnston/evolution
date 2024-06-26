from abc import ABC, abstractmethod
import random
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
        gc = type(self)(seeds)
        return gc
    def copy(self):
        c = type(self)(self.seeds.copy())
        return c

    # simple sexual reproduction, keep all genes in same order, append partners missing genes.
    def combine(self, partner_dna):
        child = BasicDNA([])
        child.seeds = self.seeds.copy()
        p1 = set(self.seeds)
        for s in partner_dna.seeds:
            if not s in p1:
                child.seeds.append(s)
        return child

class OrderedDNA(BasicDNA):
    def combine(self, partner_dna):
        child = OrderedDNA([])
        child.seeds = []
        for i,(s1,s2) in enumerate(zip(self.seeds, partner_dna.seeds)):
            child.seeds.append(random.choice((s1,s2)))
        return child

class OrderedMultiDNA(BasicDNA):
    def mutate(self, random_seed):
        child = type(self)([])
        child.seeds = self.seeds + [(random_seed,)]
        return child

    # when combining, we always replace the last gene with 
    # a list of the all other successful genes from this population.
    # So, dnas should be a list of all such dnas (i.e. the 64 parents, including self)
    def combine(self, dnas):
        child = OrderedMultiDNA([])
        child.seeds = []

        child.seeds = self.seeds.copy()
        
        if child.seeds:
            child.seeds[-1] = tuple(x.seeds[-1][0] for x in dnas)
        # Otherwise, dnas will all be empty

        return child

class AsexualSexualDummyDNA(BasicDNA):
    def combine(self, partner_dna):
        child = OrderedDNA([])
        child.seeds = self.seeds.copy()
        return child
           


DNA.register(BasicDNA)
DNA.register(OrderedDNA)
DNA.register(AsexualSexualDummyDNA)
DNA.register(OrderedMultiDNA)


