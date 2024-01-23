
from random import Random
from collections import namedtuple

MAXINT=2**31-1
class RandomSeedGenerator(Random):
    def __init__(self, seed):
        self.rng = Random()
        self.rng.seed(seed)
    def random(self):
        return self.rng.randrange(0, MAXINT)


# namedtuples must be named the same as the class name "Individual"
# otherwise mp wont be able to find it
Individual = namedtuple('Individual',['dna','fitness'])

