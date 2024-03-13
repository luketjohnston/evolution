from abc import ABC, abstractmethod
import math
import torch
import time
from codes import BasicDNA # TODO shouldnt have to do this

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


class ConvPolicy(torch.nn.Module):
    def __init__(self, dna, input_dim, kernel_dims, channels, strides, act_dim, hidden_size, initialization_seed=0, sigma=0.002):
        ''' sigma is the variance of mutations (torch.normal(mean=0, std=he_init_std * sigma))
        '''
        super().__init__()
        t1 = time.time()
        self.kernel_dims = kernel_dims
        self.generator = torch.Generator()
        self.generator.manual_seed(initialization_seed) 
        input_dim = list(input_dim)

        self.kernels = []
        self.kernel_biases = []
        self.strides = strides
        self.dna = BasicDNA([]) # don't set to dna arg yet, will do that with call to self.update_dna(dna) below
        self.sigma = sigma
        self.channels = channels

        self.initialization_seed = initialization_seed

        # initialization
        for i,k in enumerate(kernel_dims):
            std = 2 / math.sqrt(kernel_dims[i] * kernel_dims[i] * channels[i]) # He initialization
            self.kernels.append(torch.normal(mean=0, std=std, size=(channels[i+1], channels[i], k, k), generator=self.generator))
            self.kernel_biases.append(torch.zeros(channels[i+1]))
            input_dim[0] = math.floor((input_dim[0] - kernel_dims[i]) / self.strides[i] + 1) # TODO verify correct?
            input_dim[1] = math.floor((input_dim[1] - kernel_dims[i]) / self.strides[i] + 1) # TODO verify correct?
       
        std = 2 / math.sqrt(input_dim[0] * input_dim[1] * channels[-1])
        self.l1 = torch.normal(mean=0, std=std, size=(input_dim[0] * input_dim[1] * channels[-1], hidden_size), generator=self.generator)
        self.l1_bias = torch.zeros(hidden_size)
        std = 2 / math.sqrt(hidden_size)
        self.l2 = torch.normal(mean=0, std=std, size=(hidden_size, act_dim), generator=self.generator)
        self.l2_bias = torch.zeros(act_dim)

        self.update_dna(dna)


        t2 = time.time()
        self.metadata = {
            'policy_make_time': time.time() - t1
        }

        # register some parameters so we can try gradient descent instead of GA, for SCIENCE!
        # TODO are we still using this?
        #self.l1 = torch.nn.Parameter(self.l1)
        #self.l1_bias = torch.nn.Parameter(self.l1_bias)
        #self.l2 = torch.nn.Parameter(self.l2)
        #self.l2_bias = torch.nn.Parameter(self.l2_bias)
        #self.kernels = [torch.nn.Parameter(k) for k in self.kernels]
        #self.kernel_biases = [torch.nn.Parameter(k) for k in self.kernel_biases]


    def update_dna(self, new_dna):
        ''' used to update the policy with a new dna. 
        Mutates intelligently in that we only perform the minimum modifications to convert
        the old network into the new one (to speed up policy creation time).
        '''
        # TODO: right now the network initialization is not encoded in the DNA, counterintuitive...
        current_seeds = set(self.dna.seeds)
        new_seeds = set(new_dna.seeds)

        for s in set(new_dna.seeds + self.dna.seeds):
            if s in current_seeds and s in new_seeds:
                continue # do nothing, this mutation is cached
            elif s in current_seeds and not s in new_seeds:
                sign = -1
            elif s not in current_seeds and s in new_seeds:
                sign = 1

            self.generator.manual_seed(s)
            for i in range(len(self.kernels)):
                # TODO should we actually scale by He init std here?
                std = 2 / math.sqrt(self.kernel_dims[i] * self.kernel_dims[i] * self.channels[i]) # He initialization
                self.kernels[i] += sign * torch.normal(mean=0, std=std*self.sigma, size=self.kernels[i].shape, generator=self.generator)
                # TODO IMPORTANT if we keep biases in here, we have to actually use them...

            
            std = 2 / math.sqrt(self.l1.shape[0])
            self.l1 += sign * torch.normal(mean=0, std=std * self.sigma, size=self.l1.shape, generator=self.generator)
            std = 2 / math.sqrt(self.l2.shape[0])
            self.l2 += sign * torch.normal(mean=0, std=std * self.sigma, size=self.l2.shape, generator=self.generator)
        self.dna = new_dna
        return self

    def __call__(self, state):
        state = torch.tensor(state).float()
        batch_size = state.shape[0]
        if len(state.shape) < 4: # add batch dimension if necessary
            state = torch.unsqueeze(state, 0)
            batch_size = state.shape[0]
           
        state = torch.permute(state, [0,3,1,2])
        for i,k in enumerate(self.kernels):
            state = torch.nn.functional.conv2d(state, k, stride=self.strides[i]) # TODO do I need bias for the kernels?
            state = state + self.kernel_biases[i][None, :, None, None] # TODO does this broadcast correctly?
            state = torch.nn.functional.relu(state)

  
        state = state.reshape([batch_size, -1]) # flatten
        state = torch.nn.functional.relu(torch.matmul(state, self.l1))
        state = state + self.l1_bias
        logits = torch.matmul(state, self.l2)
        logits = logits + self.l2_bias
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

class ConvModule(torch.nn.Module):
    def __init__(self, dna, input_dim, kernel_dims, channels, strides, act_dim, hidden_size, initialization_seed=0, sigma=0.002):
        ''' sigma is the variance of mutations (torch.normal(mean=0, std=he_init_std * sigma))
        '''
        super().__init__()
        input_dim = list(input_dim)

        self.kernels = []
        self.strides = strides

        for i,k in enumerate(kernel_dims):
            self.kernels.append(torch.nn.Conv2d(channels[i], channels[i+1], k, stride=strides[i]))
            input_dim[0] = math.floor((input_dim[0] - kernel_dims[i]) / self.strides[i] + 1) # TODO verify correct?
            input_dim[1] = math.floor((input_dim[1] - kernel_dims[i]) / self.strides[i] + 1) # TODO verify correct?
       
        self.l1 = torch.nn.Linear(input_dim[0] * input_dim[1] * channels[-1], hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, act_dim)


    def __call__(self, state):
        state = torch.tensor(state).float()
        batch_size = state.shape[0]
        if len(state.shape) < 4: # add batch dimension if necessary
            state = torch.unsqueeze(state, 0)
            batch_size = state.shape[0]
           
        state = torch.permute(state, [0,3,1,2])
        for i,k in enumerate(self.kernels):
            state = k(state) # TODO do I need bias for the kernels?
            state = torch.nn.functional.relu(state)

  
        state = state.reshape([batch_size, -1]) # flatten
        state = torch.nn.functional.relu(self.l1(state))
        logits = self.l2(state)
        return logits

class MultiConv(torch.nn.Module):
    ''' MultiConv contains multiple ConvPolicy in parallel. The logit for each class is the max
    of corresponding class logit for each ConvPolicy. Mutations only apply to one ConvPolicy at a time.
    The motivation for this is that it will make it less likely for new mutations to override
    old knowledge.  '''
    def __init__(self, dna, input_dim, kernel_dims, channels, strides, act_dim, hidden_size, initialization_seed=0, sigma=0.002, multi=4):
        ''' sigma is the variance of mutations (torch.normal(mean=0, std=he_init_std * sigma))
        '''
        super().__init__()
        t1 = time.time()
        input_dim = list(input_dim)

        self.strides = strides
        self.sigma = sigma
        self.channels = channels
        self.multi = multi

        self.initialization_seed = initialization_seed

        self.convs = []

        for c in range(multi):
            c_init_seed = initialization_seed * multi + c
            print('c init seed:', c_init_seed)
            start_dna = BasicDNA([])
            self.convs.append(ConvPolicy(start_dna, input_dim, kernel_dims, channels, strides, act_dim, hidden_size, initialization_seed=c_init_seed, sigma=self.sigma))
            
        self.update_dna(dna)

        t2 = time.time()
        self.metadata = {
            'policy_make_time': time.time() - t1
        }


    def update_dna(self, new_dna):
        ''' used to update the policy with a new dna. 
        Mutates intelligently in that we only perform the minimum modifications to convert
        the old network into the new one (to speed up policy creation time).
        '''
        # TODO: right now the network initialization is not encoded in the DNA, counterintuitive...

        dnas = [[] for _ in range(self.multi)]

        # split up dna so 1/4 of the genes apply to each ConvPolicy
        for s in new_dna.seeds:
            dnas[s % self.multi].append(s // self.multi)

        for c,new_dna in zip(self.convs, dnas):
            c_dna = BasicDNA(new_dna)
            c.update_dna(c_dna)

        self.dna = new_dna
        return self

    def __call__(self, state):
        logitl = []
        for c in self.convs:
            logitl.append(c(state))

        logits = torch.stack(logitl)
        logits = torch.max(logits, dim=0)[0]
            
        return logits

    def act(self, state):
        logits = self(state)
        probs = torch.nn.functional.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1)
        if action.shape[0] == 1: # TODO get rid of this item() logic
            return action.item()
        else:
            return action
        
PolicyNetwork.register(MultiConv)
