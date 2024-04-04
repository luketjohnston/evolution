import torch
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


''' x entries must be either 0 or 1 '''
def first_nonzero_index(x):
  idx = torch.arange(x.shape[0] + 1, 1, -1) 
  tmp = idx * x
  return torch.tensor(-1 * torch.max(tmp, 0)[0] + x.shape[0] + 1)


if __name__ == '__main__':
  x = torch.tensor([0,0,0,0,1,0,1,])
  assert first_nonzero_index(x) == 4, first_nonzero_index(x)
  x = torch.tensor([0,0,0,0,0,0,0,])
  assert first_nonzero_index(x) == 8, first_nonzero_index(x)
  x = torch.tensor([1,0,0,0,0,0,0,])
  assert first_nonzero_index(x) == 0, first_nonzero_index(x)
  
 
