from torch import nn
import torch
import torch.nn.functional as F


class EvoLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, improvement=False, mate_multiplier=8) -> None:
        """ 
        "improvement" arg means that we are going to be keeping track of last generation's
        weights, and whenever we forward throug the layer we also forward through the previous
        weights, so that we can then eventually compute the difference in fitness between 
        the current weights and the previous weights.
        """
       
        super().__init__()
        self.weight: torch.Tensor
        # TODO should we initialize differently here?
        self.register_buffer('weight', torch.zeros(out_features, in_features))
        self.mate_multiplier = mate_multiplier
        

    def next_generation(self, population_size: int, lr: float):
        out_features, in_features = self.weight.size()
        mean = self.weight.expand(population_size, out_features, in_features) 
        self.offspring = torch.normal(mean, std=lr) 

    def mate(self, parents, fitness_weights=None):
        if fitness_weights is None:
            adjustment = self.offspring[parents, :, :].mean(0, keepdim=False)  - self.weight
            self.weight = self.weight + self.mate_multiplier * adjustment
            #print(f'adj norm for {self.weight.shape}:', torch.norm(adjustment))
        else:
            # This seems mostly unnecessary, rarely ever happens after first generation.
            fitness_weights = (fitness_weights > 0) * fitness_weights 
            # TODO update this hyperparam, 10 seems best so far (3 diverges)
            fitness_weights = torch.nn.functional.normalize(fitness_weights, dim=None)  / 10
            adjustment  = ((self.offspring[parents, ...] - self.weight[None,...])*fitness_weights[:,None,None]).sum(0, keepdim=False)

            #print(f'adj norm for {self.weight.shape}:', torch.norm(adjustment))
            self.weight = self.weight + self.mate_multiplier * adjustment

    def reset(self):
        self.offspring = None

    def forward(self, x):
        #print('x:', type(x))
        #print(tuple(_.shape for _ in x))
            
        original_results =  F.linear(x[0], self.weight)
        if self.offspring is not None and len(x) > 1:
            mutation_results =  torch.einsum('pbi,poi->pbo', x[1], self.offspring)
            return [original_results, mutation_results]
        return [original_results]



class EvoModel(nn.Module):
    def __init__(self, hidden_size, mate_multiplier):
        super().__init__()
        self.fc1 = EvoLinear(28 * 28, hidden_size, mate_multiplier=mate_multiplier)
        self.fc2 = EvoLinear(hidden_size, 10, mate_multiplier=mate_multiplier)

    def forward(self, x):
        x = self.fc1.forward([_.flatten(2) for _ in x])
        return self.fc2.forward([F.relu(_) for _ in x])  

    def next_generation(self, population_size: int, lr: float):
        for m in self.modules():
            if isinstance(m, EvoLinear):
                m.next_generation(population_size, lr)

    def mate(self, parents, fitness_weights=None):
        for m in self.modules():
            if isinstance(m, EvoLinear):
                m.mate(parents, fitness_weights)

    def reset(self):
        for m in self.modules():
            if isinstance(m, EvoLinear):
                m.reset()
