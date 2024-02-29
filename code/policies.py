from abc import ABC, abstractmethod
import math
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
            # TODO fix this, see ConvPolicy
            self.generator.manual_seed(s)
            self.l1 += torch.normal(mean=0, std=1, size=self.l1.shape, generator=self.generator)
            self.l2 += torch.normal(mean=0, std=1, size=self.l2.shape, generator=self.generator)

    def act(self, state):
        state = torch.tensor(state).float()
        hidden = torch.nn.functional.relu(torch.matmul(state,  self.l1))
        probs = torch.nn.functional.softmax(torch.matmul(hidden, self.l2), dim=0)
        action = torch.multinomial(probs, 1)
        return action.item()
PolicyNetwork.register(LinearPolicy)


def calc_std(in_fan, out_fan, gain):
   ''' 'gain' should be root(2) for ReLU '''

class ConvPolicy:
    def __init__(self, dna, input_dim, kernel_dims, channels, strides, act_dim, hidden_size, initialization_seed=0, sigma=0.002):
        ''' sigma is the variance of mutations (torch.normal(mean=0, std=he_init_std * sigma))
        '''
        self.generator = torch.Generator()
        self.generator.manual_seed(initialization_seed) 
        input_dim = list(input_dim)

        self.kernels = []
        self.strides = strides

        for i,k in enumerate(kernel_dims):
            std = 2 / math.sqrt(kernel_dims[i] * kernel_dims[i] * channels[i]) # He initialization
            self.kernels.append(torch.normal(mean=0, std=std, size=(channels[i+1], channels[i], k, k), generator=self.generator))
            input_dim[0] = math.floor((input_dim[0] - kernel_dims[i]) / self.strides[i] + 1) # TODO verify correct?
            input_dim[1] = math.floor((input_dim[1] - kernel_dims[i]) / self.strides[i] + 1) # TODO verify correct?

        std = 2 / math.sqrt(input_dim[0] * input_dim[1] * channels[-1])
        self.l1 = torch.normal(mean=0, std=std, size=(input_dim[0] * input_dim[1] * channels[-1], hidden_size), generator=self.generator)
        std = 2 / math.sqrt(hidden_size)
        self.l2 = torch.normal(mean=0, std=std, size=(hidden_size, act_dim), generator=self.generator)

        # recompute ("birth") the new policy network from the dna 
        for s in dna.seeds:
            # TODO should std be something different? For initializing networks?
            self.generator.manual_seed(s)
            for i in range(len(self.kernels)):
                # TODO should we actually scale by He init std here?
                std = 2 / math.sqrt(kernel_dims[i] * kernel_dims[i] * channels[i]) # He initialization
                self.kernels[i] += torch.normal(mean=0, std=std*sigma, size=self.kernels[i].shape, generator=self.generator)

            
            std = 2 / math.sqrt(self.l1.shape[0])
            self.l1 += torch.normal(mean=0, std=std * sigma, size=self.l1.shape, generator=self.generator)
            std = 2 / math.sqrt(self.l2.shape[0])
            self.l2 += torch.normal(mean=0, std=std * sigma, size=self.l2.shape, generator=self.generator)

    def __call__(self, state):
        state = torch.tensor(state).float()
        batch_size = state.shape[0]
        if len(state.shape) < 4: # add batch dimension if necessary
            state = torch.unsqueeze(state, 0)
            batch_size = state.shape[0]
           
        state = torch.permute(state, [0,3,1,2])
        for i,k in enumerate(self.kernels):
            #print('state:', state, flush=True)
            #print('k:', k, flush=True)
            #print('strides[i]:', self.strides[i], flush=True)
            state = torch.nn.functional.conv2d(state, k, stride=self.strides[i]) # TODO do I need bias for the kernels?
            state = torch.nn.functional.relu(state)

  
        #print('state.shape:', state.shape, flush=True)
        #print('batch_size:', batch_size, flush=True)
        state = state.reshape([batch_size, -1]) # flatten
        state = torch.nn.functional.relu(torch.matmul(state, self.l1))
        logits = torch.matmul(state, self.l2)
        return logits

    def act(self, state):
        logits = self(state)
        probs = torch.nn.functional.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1)
        if action.shape[0] == 1: # TODO get rid of this item() logic
            return action.item()
        else:
            return action
        
PolicyNetwork.register(ConvPolicy)
        
