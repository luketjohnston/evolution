from abc import ABC, abstractmethod
import numpy as np
import functools
import math
import torch
import time
from evolution.src.codes import BasicDNA # TODO shouldnt have to do this

                ## randomly pick two axes, generate 2d rotation matrix
                #a1,a2 = torch.randint(self.heads,size=(2,), generator=self.generator)
                #rot = torch.eye(n=self.memories.shape[1])
                #theta = self.sigma * torch.rand(size=(1,), generator=self.generator)
                #rot[a1,a1] = torch.cos(theta)
                #rot[a2,a2] = torch.cos(theta)
                #rot[a1,a2] = torch.sin(theta)
                #rot[a2,a1] = -torch.sin(theta)
                ##print(rot[a1,a1],rot[a1,a2],rot[a2,a1])
                ##print(f'before rot ({a1},{a2}) {theta}:', self.memories[memory_i])
                ## rotate the memory
                #self.memories[memory_i] = rot @ self.memories[memory_i]
                ##print(f'after rot ({a1},{a2}) {theta}:', self.memories[memory_i])


class PolicyNetwork(ABC):
    @abstractmethod
    def __init__(self, dna):
        pass
    @abstractmethod
    def act(self, state):
        pass


class MemorizationModule(torch.nn.Module):
    def __init__(self, dna, input_dim, act_dim, heads, initialization_seed=0, sigma=0.002,proj_dim=None,add_memory_prob=0.4,remove_memory_prob=0.3):
        # TODO update to flatten all memories and inputs etc, see other module at bottom
        self.metadata = {}
        self.add_memory_prob = add_memory_prob
        self.remove_memory_prob = remove_memory_prob
        #self.perturb_memory_prob = 0.3
        self.act_dim = act_dim
        self.heads = heads
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.sigma = sigma
        self.generator = torch.Generator()
        self.initialization_seed = initialization_seed
        self.initialize_helper(initialization_seed)
        self.update_dna(dna)


    def initialize_helper(self, initialization_seed):
        self.generator.manual_seed(initialization_seed)

        if self.proj_dim:
            flattened_input_dim = functools.reduce(lambda a,b: a * b,self.input_dim,1)
            self.random_projection = torch.normal(mean=0,std=1,size=[flattened_input_dim,self.proj_dim],generator=self.generator)
            #self.memories = torch.nn.functional.normalize(torch.normal(mean=0,std=1,size=[self.heads,self.proj_dim], generator=self.generator), dim=1)
            self.logits = torch.zeros([heads, act_dim])
            self.memories = torch.zeros([self.heads,self.proj_dim])
        else:
            #self.memories = torch.nn.functional.normalize(torch.normal(mean=0,std=1,size=[self.heads] + list(self.input_dim), generator=self.generator), dim=1)
            self.logits = torch.zeros([self.heads, self.act_dim])
            self.memories = torch.zeros([self.heads] + list(self.input_dim))
        self.used_memories = []
        self.unused_memories = list(range(self.heads))
        self.dna = BasicDNA([]) # don't set to dna arg yet, will do that with call to self.update_dna(dna) below

    def update_dna(self, new_dna):
        ''' used to update the policy with a new dna. 
        Mutates intelligently in that we only perform the minimum modifications to convert
        the old network into the new one (to speed up policy creation time).
        '''
        t1 = time.time()
        # TODO: right now the network initialization is not encoded in the DNA, counterintuitive...
        current_seeds = set(self.dna.seeds)
        new_seeds = set(new_dna.seeds)
        self.initialize_helper(self.initialization_seed)

        for s in new_dna.seeds:
            self.generator.manual_seed(s)
            mutation_type = torch.rand(size=(1,), generator=self.generator).item()

            if mutation_type < self.add_memory_prob or len(self.used_memories) == 0:
                if self.unused_memories:
                    memory_i_i = torch.randint(len(self.unused_memories), size=(1,), generator=self.generator).item()
                    memory_i = self.unused_memories[memory_i_i]
                    self.unused_memories.pop(memory_i_i)
                    self.used_memories.append(memory_i)
                else:
                    memory_i_i = torch.randint(len(self.used_memories), size=(1,), generator=self.generator).item()
                    memory_i = self.used_memories[memory_i_i]
                    

                # set the std such that the expectation of the norm of the tensor is 1 TODO not important?
                self.memories[memory_i] = self.sigma * torch.normal(mean=0, std=1.0 / (self.memories.shape[1])**0.5, size=self.memories.shape[1:], generator=self.generator)
                self.memories[memory_i] = torch.nn.functional.normalize(self.memories[memory_i],dim=0)
                self.logits[memory_i] = torch.normal(mean=0,std=1,size=self.logits.shape[1:],generator=self.generator)

            elif mutation_type < self.add_memory_prob + self.remove_memory_prob:
                # delete a memory
                memory_i_i = torch.randint(len(self.used_memories), size=(1,), generator=self.generator).item()
                memory_i = self.used_memories[memory_i_i]
                self.memories[memory_i] = torch.zeros(self.memories.shape[1])
                self.used_memories.pop(memory_i_i)
                self.unused_memories.append(memory_i)
            else:
                # perturb a memory
                memory_i_i = torch.randint(len(self.used_memories), size=(1,), generator=self.generator).item()
                memory_i = self.used_memories[memory_i_i]

                self.memories[memory_i] += self.sigma * torch.normal(mean=0, std=1.0 / (self.memories.shape[1])**0.5, size=self.memories.shape[1:], generator=self.generator)
                self.memories[memory_i] = torch.nn.functional.normalize(self.memories[memory_i],dim=0)



        #for s in set(new_dna.seeds + self.dna.seeds):
        #    if s in current_seeds and s in new_seeds:
        #        continue # do nothing, this mutation is cached
        #    elif s in current_seeds and not s in new_seeds:
        #        sign = -1
        #    elif s not in current_seeds and s in new_seeds:
        #        sign = 1
        #    memory_i = s % self.memories.shape[0]
        #    memory_seed = s // self.memories.shape[0]
        #    self.generator.manual_seed(memory_seed)
        #    self.memories[memory_i] += sign * self.sigma * torch.normal(mean=0, std=1, size=self.memories.shape[1:], generator=self.generator)
        #    if self.normalize:
        #        self.memories[memory_i] = torch.nn.functional.normalize(self.memories[memory_i],dim=0)
        #    self.logits[memory_i] += sign * self.sigma * torch.normal(mean=0,std=1,size=self.logits.shape[1:],generator=self.generator)

        self.dna = new_dna
        self.metadata['policy_make_time'] = time.time() - t1
        return self
        
    def __call__(self, state):
        # state [batch, h,w,c]
        # memories [heads,h,w,c]
        if self.proj_dim:
            state = state.view([state.shape[0], -1]) # flatten
            state = state @ self.random_projection
            similarities = torch.sum(state[:,None] * self.memories[None,:],dim=-1)
        else:
            reduce_dimensions = list(range(2,len(self.input_dim) + 2))
            similarities = torch.sum(state[:,None] * self.memories[None,:], dim=reduce_dimensions)
        max_similarities, closest_memories = torch.max(similarities, dim=1)
        intrinsic_fitness = torch.sum(max_similarities)
        intrinsic_fitness += len(self.unused_memories) * 100 
        logits = self.logits[closest_memories]
        return logits, intrinsic_fitness.item()

    def act(self, state):
        logits = self(state)
        probs = torch.nn.functional.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1)
        if action.shape[0] == 1: # TODO get rid of this item() logic
            return action.item()
        else:
            return action

class LinearPolicy(torch.nn.Module):
    ''' 
    Simple linear policy, one hidden state with relu activation, maps input dimension to 
    action dimension.
    '''
    def __init__(self, dna, input_dim, hidden_dim, act_dim, initialization_seed, sigma, trainable=False, mutation='one', device='cpu', sigma_mutation=1.3, sigma_only_generations=-1):
        super().__init__()
        self.start_sigma = sigma
        self.sigma_mutation = sigma_mutation
        self.device = device
        self.generator = torch.Generator(device=device)
        self.initialization_seed = initialization_seed
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.sigma_only_generations = sigma_only_generations
        self.metadata={'policy_make_time' : 0}
        assert mutation in ['one','exponential','normal', 'uniform']
        self.mutation=mutation
        self.trainable=trainable
        self.initialization_helper()
        self.update_dna(dna)



    def initialization_helper(self):
        self.generator.manual_seed(self.initialization_seed)
        self.sigma1 = self.start_sigma[0]
        self.sigma2 = self.start_sigma[1]
        std = 2 / math.sqrt(self.hidden_dim)
        self.l1 = torch.nn.Parameter(torch.normal(mean=0, std=std, size=(self.input_dim, self.hidden_dim), generator=self.generator, device=self.device), requires_grad=self.trainable)
        std = 2 / math.sqrt(self.act_dim)
        self.l2 = torch.nn.Parameter(torch.normal(mean=0, std=std, size=(self.hidden_dim, self.act_dim), generator=self.generator, device=self.device), requires_grad=self.trainable)
        self.dna = BasicDNA([]) # don't set to dna arg yet, will do that with call to self.update_dna(dna) below

    def mutate(self, seed, sign, sigma_only=False):
        self.generator.manual_seed(seed)
        sigma1_update = (torch.randint(3, size=(1,), generator=self.generator, device=self.device) - 1).item()
        sigma2_update = (torch.randint(3, size=(1,), generator=self.generator, device=self.device) - 1).item()

        if sign == 1: 
            self.sigma1 *= self.sigma_mutation**(sign * sigma1_update)
            self.sigma2 *= self.sigma_mutation**(sign * sigma2_update)

            
        if self.mutation == 'normal' and not sigma_only:
            std = 2 / math.sqrt(self.hidden_dim)
            self.l1 += sign * self.sigma1 * torch.normal(mean=0, std=1, size=self.l1.shape, generator=self.generator,device=self.device)
            std = 2 / math.sqrt(self.act_dim)
            self.l2 += sign * self.sigma2 * torch.normal(mean=0, std=1, size=self.l2.shape, generator=self.generator, device=self.device)
        elif self.mutation == 'exponential' and not sigma_only:
                lambda1 = 1
                randsign = torch.randint(2, size=self.l1.shape, generator=self.generator, device=self.device) * 2 - 1
                self.l1 += self.sigma1 * sign * randsign * torch.zeros(self.l1.shape, device=self.device).exponential_(lambda1, generator=self.generator)
                lambda2 = 1
                randsign = torch.randint(2, size=self.l2.shape, generator=self.generator, device=self.device) * 2 - 1
                self.l2 += self.sigma2 * sign * randsign * torch.zeros(self.l2.shape, device=self.device).exponential_(lambda2, generator=self.generator)
        elif self.mutation == 'one' and not sigma_only: 
            randsign = (torch.randint(2, size=(1,), generator=self.generator, device=self.device) * 2 - 1)
            if randsign > 0:
                randsign = (torch.randint(2, size=(1,), generator=self.generator, device=self.device) * 2 - 1)
                self.l1[torch.randint(self.l1.shape[0], size=(1,), generator=self.generator, device=self.device)] += self.sigma1 * sign  * randsign
            else:
                randsign = (torch.randint(2, size=(1,), generator=self.generator, device=self.device) * 2 - 1)
                self.l2[torch.randint(self.l2.shape[0], size=(1,), generator=self.generator, device=self.device)] += self.sigma2 * sign * randsign
        elif self.mutation == 'uniform' and not sigma_only:
                self.l1 += self.sigma1 * sign * (torch.rand(self.l1.shape, device=self.device, generator=self.generator) * 2 - 1)
                self.l2 += self.sigma2 * sign * (torch.rand(self.l2.shape, device=self.device, generator=self.generator) * 2 - 1)

        if sign == -1:
            self.sigma1 *= self.sigma_mutation**(sign * sigma1_update)
            self.sigma2 *= self.sigma_mutation**(sign * sigma2_update)

    # Invariant:
    # Full mutation is applied for all the seed indexed i==self.sigma_only_generations 
    # Full mutation is always applied for the last seed
    def update_dna(self, new_dna):
        t1 = time.time()

        i = 0
        while (i < min(len(new_dna.seeds), len(self.dna.seeds))) and  new_dna.seeds[i] == self.dna.seeds[i]:
            i += 1

        # If we are still mutating only the sigmas, we need to make sure to rollback the last self.dna.seed
        if (i == len(self.dna.seeds)) and i <= self.sigma_only_generations and i > 0: 
            i -= 1

        # TODO need to verify this all actually works once I start running with more parents again 
        # rollback unused mutations
        current_i = len(self.dna.seeds)
        for s in reversed(self.dna.seeds[i:]):
            current_i -= 1
            sigma_only = current_i < len(self.dna.seeds)-1 and current_i < self.sigma_only_generations
            #print(f"ROLLING BACK self.dna: {self.dna.seeds}, new_dna: {new_dna.seeds}, seed: {s}, sigma_only: {sigma_only}", flush=True)
            self.mutate(s, -1, sigma_only=sigma_only)

        # add new mutations
        current_i = i
        for s in new_dna.seeds[i:]:
            sigma_only = current_i < len(new_dna.seeds)-1 and current_i < self.sigma_only_generations
            current_i += 1
            #print(f"MUTATING self.dna: {self.dna.seeds}, new_dna: {new_dna.seeds}, seed: {s}, sigma_only: {sigma_only}", flush=True)
            self.mutate(s, 1, sigma_only=sigma_only)
        #print(f"Positive mutations: {positive_mutations}, negative_mutations: {negative_mutations}")



        #for s in set(new_dna.seeds + self.dna.seeds):
        #    if s in current_seeds and s in new_seeds:
        #        continue # do nothing, this mutation is cached
        #    elif s in current_seeds and not s in new_seeds:
        #        sign = -1
        #    elif s not in current_seeds and s in new_seeds:
        #        sign = 1
        #        self.sigma *= 2**(sigma_update)



        self.dna = new_dna
        self.metadata['policy_make_time'] = time.time() - t1
        self.metadata['sigma1'] = self.sigma1
        self.metadata['sigma2'] = self.sigma2
        return self

    def act(self, state):
        state = torch.tensor(state).float()
        state = state.view([state.shape[0], -1]) # flatten 
        hidden = torch.nn.functional.relu(torch.matmul(state,  self.l1))
        probs = torch.nn.functional.softmax(torch.matmul(hidden, self.l2), dim=0)
        action = torch.multinomial(probs, 1)
        return action.item()
    def __call__(self, state):
        state = torch.tensor(state).float()
        state = state.view([state.shape[0], -1]) # flatten 
        hidden = torch.nn.functional.relu(torch.matmul(state,  self.l1))
        logits = torch.matmul(hidden, self.l2)
        if not self.trainable:
            return logits,0
        else:
            return logits

PolicyNetwork.register(LinearPolicy)


class ConvPolicy(torch.nn.Module):
    def __init__(self, dna, input_dim, kernel_dims, channels, strides, act_dim, hidden_size, initialization_seed=0, sigma=0.002, update_type='exponential'):
        ''' sigma is the variance of mutations (torch.normal(mean=0, std=he_init_std * sigma))

        '''
        assert update_type in ['exponential','normal']
        self.update_type = update_type
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
            input_dim[0] = math.floor((input_dim[0] - kernel_dims[i]) / self.strides[i] + 1) // 2 # TODO update, make max pool an arg / setting
            input_dim[1] = math.floor((input_dim[1] - kernel_dims[i]) / self.strides[i] + 1) // 2 # TODO update, make max pool an arg / setting
       
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
            lambdas = [182, 466, 976, 136]

            for i in range(len(self.kernels)):
                # TODO should we actually scale by He init std here?
                if self.update_type == 'normal':
                    std = 2 / math.sqrt(self.kernel_dims[i] * self.kernel_dims[i] * self.channels[i]) # He initialization
                    adjustment = torch.normal(mean=0, std=std, size=self.kernels[i].shape, generator=self.generator)
                elif self.update_type == 'exponential':
                   # Use the exponential distribution we found from examining the gradient updates:
                   adjustment = torch.zeros(size=self.kernels[i].shape).exponential_(lambdas[i], generator=self.generator)
                   # Flip half of distribution below 0 (giving us symmetric exponential around 0)
                   adjustment *= (torch.randint(2, size=self.kernels[i].shape, generator=self.generator) * 2 - 1)
                else: 
                    assert False

                self.kernels[i] += sign * self.sigma * adjustment
                # TODO if we keep biases in here, we have to actually use them...

            
            if self.update_type == 'normal':
                std = 2 / math.sqrt(self.l1.shape[0])
                self.l1 += sign * self.sigma * torch.normal(mean=0, std=std, size=self.l1.shape, generator=self.generator)
                std = 2 / math.sqrt(self.l2.shape[0])
                self.l2 += sign * self.sigma * torch.normal(mean=0, std=std, size=self.l2.shape, generator=self.generator)

            elif self.update_type == 'exponential':
                adjustment = torch.zeros(size=self.l1.shape).exponential_(lambdas[2], generator=self.generator)
                adjustment *= (torch.randint(2, size=self.l1.shape, generator=self.generator) * 2 - 1)
                self.l1 += sign * self.sigma * adjustment
                adjustment = torch.zeros(size=self.l2.shape).exponential_(lambdas[3], generator=self.generator)
                adjustment *= (torch.randint(2, size=self.l2.shape, generator=self.generator) * 2 - 1)
                self.l2 += sign * self.sigma * adjustment
            else:
                assert False
        self.dna = new_dna
        return self

    def __call__(self, state):
        state = torch.tensor(state).float()
        batch_size = state.shape[0]
        if len(state.shape) < 4: # add batch dimension if necessary
            state = torch.unsqueeze(state, 0)
            batch_size = state.shape[0]
           
        #if state.shape[-1] <= 3:
        #state = torch.permute(state, [0,3,1,2])

        for i,k in enumerate(self.kernels):
            state = torch.nn.functional.conv2d(state, k, stride=self.strides[i]) # TODO do I need bias for the kernels?
            state = state + self.kernel_biases[i][None, :, None, None] # TODO does this broadcast correctly?
            state = torch.nn.functional.max_pool2d(state, 2) # TODO make maxpool a param
            state = torch.nn.functional.relu(state)

  
        state = state.reshape([batch_size, -1]) # flatten
        state = torch.nn.functional.relu(torch.matmul(state, self.l1))
        #state = state + self.l1_bias
        logits = torch.matmul(state, self.l2)
        #logits = logits + self.l2_bias
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

        self.kernels = torch.nn.ParameterList([])
        self.strides = strides

        for i,k in enumerate(kernel_dims):
            self.kernels.append(torch.nn.Conv2d(channels[i], channels[i+1], k, stride=strides[i], bias=False))
            input_dim[0] = math.floor((input_dim[0] - kernel_dims[i]) / self.strides[i] + 1) // 2 # TODO verify correct?
            input_dim[1] = math.floor((input_dim[1] - kernel_dims[i]) / self.strides[i] + 1) // 2 # TODO verify correct?
       
        self.l1 = torch.nn.Linear(input_dim[0] * input_dim[1] * channels[-1], hidden_size, bias=False)
        self.l2 = torch.nn.Linear(hidden_size, act_dim, bias=False)


    def __call__(self, state):
        state = torch.tensor(state).float()
        batch_size = state.shape[0]
        if len(state.shape) < 4: # add batch dimension if necessary
            state = torch.unsqueeze(state, 0)
            batch_size = state.shape[0]
           
        if state.shape[-1] <= 3:
            state = torch.permute(state, [0,3,1,2])
        for i,k in enumerate(self.kernels):
            state = k(state) # TODO do I need bias for the kernels?
            state = torch.nn.functional.max_pool2d(state, 2) # TODO make maxpool a param
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





# TODO modularize the learned learning rate etc.
class MemorizationModuleWithLR(MemorizationModule):
    def __init__(self, dna, input_dim, act_dim, heads, initialization_seed=0, sigma=0.002,proj_dim=None,lr_sigma=0.002):
       self.lr_sigma = lr_sigma
       self.ln_lrs = torch.zeros(heads) # each head has a unique lr, stored in log space
       super().__init__(dna,input_dim,act_dim,heads,initialization_seed,sigma,proj_dim) 
    def update_dna(self, new_dna):
        ''' used to update the policy with a new dna. 
        Mutates intelligently in that we only perform the minimum modifications to convert
        the old network into the new one (to speed up policy creation time).
        '''
        # TODO: right now the network initialization is not encoded in the DNA, counterintuitive...

        for j,s in enumerate(new_dna.seeds):
          if j >= len(self.dna.seeds) or s != self.dna.seeds[j]:
            break

        # revert seeds that are no longer included
        for i in range(len(self.dna.seeds)-1, j-1, -1):
            s = self.dna.seeds[i]
            memory_i = s % self.memories.shape[0]
            memory_seed = s // self.memories.shape[0]
            self.generator.manual_seed(memory_seed)

            noise1 = torch.normal(mean=0, std=1, size=self.memories.shape[1:], generator=self.generator)
            noise2 = torch.normal(mean=0, std=1, size=self.logits.shape[1:], generator=self.generator)
            noise3 = torch.normal(mean=0, std=1, size=self.ln_lrs[memory_i].shape, generator=self.generator)

            self.ln_lrs[memory_i] -= self.lr_sigma * noise3 # when reversing we need to do lr first
            self.memories[memory_i] -= self.sigma * noise1 * torch.exp(self.ln_lrs[memory_i])
            self.logits[memory_i] -= self.sigma * noise2 * torch.exp(self.ln_lrs[memory_i])
          
        ## add new seeds
        #for s in new_dna.seeds:
        for s in new_dna.seeds[j:]:
            memory_i = s % self.memories.shape[0]
            memory_seed = s // self.memories.shape[0]
            self.generator.manual_seed(memory_seed)

            noise1 = torch.normal(mean=0, std=1, size=self.memories.shape[1:], generator=self.generator)
            noise2 = torch.normal(mean=0, std=1, size=self.logits.shape[1:], generator=self.generator)
            noise3 = torch.normal(mean=0, std=1, size=self.ln_lrs[memory_i].shape, generator=self.generator)

            self.memories[memory_i] += self.sigma * noise1 * torch.exp(self.ln_lrs[memory_i])
            self.logits[memory_i] += self.sigma * noise2 * torch.exp(self.ln_lrs[memory_i])
            self.ln_lrs[memory_i] += self.lr_sigma * noise3

        self.dna = new_dna
        return self

class MemorizationModuleWithLRFull(MemorizationModule):
    def __init__(self, dna, input_dim, act_dim, heads, initialization_seed=0, sigma=0.002,proj_dim=None,lr_sigma=0.002):
       self.memories_ln_lrs = torch.zeros((heads, proj_dim)) # a unique lr for each parameter 
       self.logits_ln_lrs = torch.zeros((heads, act_dim))  # a unique lr for each logit parameter
       self.lr_sigma = lr_sigma
       super().__init__(dna,input_dim,act_dim,heads,initialization_seed,sigma,proj_dim) 
    def update_dna(self, new_dna):
        ''' used to update the policy with a new dna. 
        Mutates intelligently in that we only perform the minimum modifications to convert
        the old network into the new one (to speed up policy creation time).
        '''
        t1 = time.time()
        # TODO: right now the network initialization is not encoded in the DNA, counterintuitive...

        #for j,s in enumerate(new_dna.seeds):
        #  if j >= len(self.dna.seeds) or s != self.dna.seeds[j]:
        #    break

        ## revert seeds that are no longer included
        #for i in range(len(self.dna.seeds)-1, j-1, -1):
        #    s = self.dna.seeds[i]
        #    self.generator.manual_seed(s)

        #    noise1 = torch.normal(mean=0, std=1, size=self.memories.shape, generator=self.generator)
        #    noise2 = torch.normal(mean=0, std=1, size=self.logits.shape, generator=self.generator)
        #    noise3 = torch.normal(mean=0, std=1, size=self.memories_ln_lrs.shape, generator=self.generator)
        #    noise4 = torch.normal(mean=0, std=1, size=self.logits_ln_lrs.shape, generator=self.generator)

        #    self.memories_ln_lrs -= self.lr_sigma * noise3 # when reversing we need to do lr first
        #    self.logits_ln_lrs -= self.lr_sigma * noise4 
        #    self.memories -= self.sigma * noise1 * torch.exp(self.memories_ln_lrs)
        #    self.logits -= self.sigma * noise2 * torch.exp(self.logits_ln_lrs)

        self.initialize_helper(self.initialization_seed)
        ## add new seeds
        #for s in new_dna.seeds[j:]:
        for s in new_dna.seeds:
            self.generator.manual_seed(s)

            noise1 = torch.normal(mean=0, std=1, size=self.memories.shape, generator=self.generator)
            noise2 = torch.normal(mean=0, std=1, size=self.logits.shape, generator=self.generator)
            noise3 = torch.normal(mean=0, std=1, size=self.memories_ln_lrs.shape, generator=self.generator)
            noise4 = torch.normal(mean=0, std=1, size=self.logits_ln_lrs.shape, generator=self.generator)

            self.memories += self.sigma * noise1 * torch.exp(self.memories_ln_lrs)
            self.logits += self.sigma * noise2 * torch.exp(self.logits_ln_lrs)
            self.memories_ln_lrs += self.lr_sigma * noise3
            self.logits_ln_lrs += self.lr_sigma * noise4

            self.memories /= torch.norm(self.memories, dim=1, keepdim=True) 
            self.logits /= torch.norm(self.logits, dim=1, keepdim=True)  # TODO do we want to normalize logits?

        self.dna = new_dna
        self.metadata['policy_make_time'] = time.time() - t1
        return self

class MemModulePlastic(torch.nn.Module):
    def __init__(self, dna, input_dim, act_dim, heads, initialization_seed=0,proj_dim=None):
        t1 = time.time()
        self.metadata = {}
        self.act_dim = act_dim
        self.heads = heads
        self.logits = torch.zeros([heads, act_dim])
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.generator = torch.Generator()
        self.initialization_seed = initialization_seed
        self.initialize_helper(initialization_seed)
        self.update_dna(dna)

        t2 = time.time()
        self.metadata = {
            'policy_make_time': time.time() - t1
        }

    def initialize_helper(self, initialization_seed):
        self.generator.manual_seed(initialization_seed)  # TODO not used
        if self.proj_dim:
            flattened_input_dim = functools.reduce(lambda a,b: a * b,self.input_dim,1)
            self.random_projection = torch.normal(mean=0,std=1,size=[flattened_input_dim,self.proj_dim],generator=self.generator)
            self.memories = torch.zeros([self.heads,self.proj_dim])
        else:
            self.memories = torch.zeros([self.heads] + list(self.input_dim)) 
        self.memory_plasticity = torch.zeros([self.heads])
        self.dna = BasicDNA([]) # don't set to dna arg yet, will do that with call to self.update_dna(dna) below


    def update_dna(self, new_dna):
        ''' used to update the policy with a new dna. 
        Mutates intelligently in that we only perform the minimum modifications to convert
        the old network into the new one (to speed up policy creation time).
        '''
        t1 = time.time()
        self.initialize_helper(self.initialization_seed)

        # TODO speed this up
        for s in new_dna.seeds:
            self.generator.manual_seed(s)
            memory_i = torch.multinomial(torch.nn.functional.softmax(self.memory_plasticity), 1, generator=self.generator)
            self.memories[memory_i] = torch.nn.functional.normalize(torch.normal(mean=0, std=1, size=self.memories.shape[1:], generator=self.generator), dim=0)
            self.logits[memory_i] = torch.nn.functional.normalize(torch.normal(mean=0,std=1,size=self.logits.shape[1:],generator=self.generator), dim=0) # TODO simplify this, no need for logits?
            #self.memory_plasticity[memory_i] -= 100
            #self.memory_plasticity += torch.normal(mean=0,std=1,size=self.memory_plasticity.shape,generator=self.generator)

        self.dna = new_dna
        self.metadata['policy_make_time'] = time.time() - t1
        return self
        
    def __call__(self, state):
        # state [batch, h,w,c]
        # memories [heads,h,w,c]
        if self.proj_dim:
            state = state.view([state.shape[0], -1]) # flatten
            state = state @ self.random_projection
            similarities = torch.sum(state[:,None] * self.memories[None,:],dim=-1)
        else:
            reduce_dimensions = list(range(2,len(self.input_dim) + 2))
            similarities = torch.sum(state[:,None] * self.memories[None,:], dim=reduce_dimensions)
        closest_memories = torch.argmax(similarities, dim=1)
        logits = self.logits[closest_memories]
        return logits
    def act(self, state):
        logits = self(state)
        probs = torch.nn.functional.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1)
        if action.shape[0] == 1: # TODO get rid of this item() logic
            return action.item()
        else:
            return action
            

class MemModuleBasic(torch.nn.Module):
    def __init__(self, dna, input_dim, act_dim, heads, initialization_seed=0,proj_dim=None):
        t1 = time.time()
        self.metadata = {}
        self.act_dim = act_dim
        self.heads = heads
        self.logits = torch.zeros([heads, act_dim])
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        if self.proj_dim:
            self.mem_dim = [proj_dim]
        else:
            self.mem_dim = np.prod(input_dim)
        self.generator = torch.Generator()
        self.initialization_seed = initialization_seed
        self.initialize_helper(initialization_seed)
        self.update_dna(dna)
        self.proj_dim = proj_dim

        t2 = time.time()
        self.metadata = {
            'policy_make_time': time.time() - t1
        }

    def initialize_helper(self, initialization_seed):
        ''' some of the initialization logic goes here, since it will be called by update_dna also'''
        self.generator.manual_seed(initialization_seed)
        if self.proj_dim:
            flattened_input_dim = functools.reduce(lambda a,b: a * b,self.input_dim,1)
            self.random_projection = torch.normal(mean=0,std=1,size=[flattened_input_dim,self.proj_dim],generator=self.generator)

        self.memories = torch.zeros([self.heads,self.mem_dim])
        self.dna = BasicDNA([]) # don't set to dna arg yet, will do that with call to self.update_dna(dna) below


    def update_dna(self, new_dna):
        ''' used to update the policy with a new dna. 
        Mutates intelligently in that we only perform the minimum modifications to convert
        the old network into the new one (to speed up policy creation time).
        '''
        t1 = time.time()
        self.initialize_helper(self.initialization_seed)

        memories_set = set()

        # TODO should we have a way to communicate back to population which genes are obselete?    
        for s in reversed(new_dna.seeds):
            self.generator.manual_seed(s)
            memory_i = s % self.heads
            if memory_i in memories_set: continue
            memories_set.add(memory_i)
            self.memories[memory_i] = torch.nn.functional.normalize(torch.normal(mean=0, std=1, size=self.memories.shape[1:], generator=self.generator), dim=0)
            self.logits[memory_i] = torch.nn.functional.normalize(torch.normal(mean=0,std=1,size=self.logits.shape[1:],generator=self.generator), dim=0) # TODO simplify this, no need for logits?
            if len(memories_set) == self.heads: break

        self.dna = new_dna
        self.metadata['policy_make_time'] = time.time() - t1
        return self
        
    def __call__(self, state, get_closest_mem=False):
        # state [batch, h,w,c]
        # memories [heads,h,w,c]

        state = state.view([state.shape[0], -1]) # flatten
        if self.proj_dim:
            state = state @ self.random_projection

        similarities = torch.sum(state[:,None] * self.memories[None,:],dim=-1)

        closest_memories = torch.argmax(similarities, dim=1)
        logits = self.logits[closest_memories]
        if get_closest_mem: return logits, closest_memories, torch.max(similarities, dim=1)
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
PolicyNetwork.register(MemorizationModule)
PolicyNetwork.register(MemorizationModuleWithLR)
PolicyNetwork.register(MemModuleBasic)
