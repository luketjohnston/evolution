from abc import ABC, abstractmethod
import torch

class PolicyNetwork(ABC):
    @abstractmethod
    def __init__(self, dna):
        pass
    @abstractmethod
    def act(self, state):
        pass

class LinearPolicy:
    ''' 
    Simple linear policy, one hidden state with relu activation, maps input dimension to 
    action dimension.
    TODO move to policy.py
    '''
    def __init__(self, dna, state_dim, hidden_dim, act_dim):
        self.generator = torch.Generator()

        self.l1 = torch.zeros(size=(state_dim, hidden_dim))
        self.l2 = torch.zeros(size=(hidden_dim, act_dim))

        # recompute ("birth") the new policy network from the dna 
        for s in dna.seeds:
            # TODO should std be something different? For initializing networks?
            self.generator.manual_seed(s)
            self.l1 += torch.normal(mean=0, std=1, size=self.l1.shape, generator=self.generator)
            self.l2 += torch.normal(mean=0, std=1, size=self.l2.shape, generator=self.generator)

    def act(self, state):
        state = torch.tensor(state)
        hidden = torch.nn.functional.relu(torch.matmul(state,  self.l1))
        probs = torch.nn.functional.softmax(torch.matmul(hidden, self.l2), dim=0)
        action = torch.multinomial(probs, 1)
        return action.item()
PolicyNetwork.register(LinearPolicy)
