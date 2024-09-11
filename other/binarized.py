import numpy as np
import torch
import torch.nn as nn
from optimized_binary import EvoBinarizedLayerOptimized # TODO remove
from my_perfect_dataloader import binarize_helper1
from test import print_binarized


'''
1. input image has 3x8xHxW bits
--- Probably should duplicate the input image to include all inverted bits also
--- when doing this we would want to enforce weights don't take from both A and ~A, right? TODO 
2. convolution kernel has 3x8xKxKxC
3. Convolution just does computes "and" and then sums up
   --- gives us functional completeness, "and" and "not"
4. Do we need to threshold the sum or can we just use it directly?
   --- pretty sure we are going to need to threshold somehow, allows accumulation
   of information
      
'''

torchtype=torch.float

class EvoBinarizedLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, elite=True, activation='none'):
        super().__init__()
        self.w: torch.Tensor
        self.elite = elite
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        # TODO add seed?

        # the first dimension is for whether this w acts on x, or on ~x
        # the next two dimensions are the population dimension and the batch dimension
        w = (torch.rand([2,1,1,in_features,out_features]) > 0.5).to(torchtype)
        self.offspring=None

        #self.running_means = torch.ones([out_features], dtype=torch.half) * in_features / 2 
        self.running_means=None
        self.gamma = 0.9
        
        self.indices = None 

        self.register_buffer('w', w)

    # I used this for some debugging; it's not used for anything now.
    def to_optimized(self):
        opt = EvoBinarizedLayerOptimized(self.in_features, self.out_features, self.elite, self.activation)
        opt.thresh = self.in_features // 2 +1

        rounded_in = (self.in_features + 63) // 64
        w = torch.zeros([2,self.offspring.shape[1],rounded_in, self.out_features], dtype=torch.int64, device=self.offspring.device)

        int_offspring = self.offspring.to(torch.int64)
        padded_offspring = torch.nn.functional.pad(int_offspring, (0,0,0,rounded_in*64-self.in_features,0,0,0,0,0,0))
        for i in range(64):
          w[:,:,:,:] = torch.bitwise_or(w[:,:,:,:], padded_offspring[:,:,0,i::64,:]<<(63-i))

        opt.w = w
        return opt




    def next_generation(self, population_size: int, lr: float):

        _, _, _, in_features, out_features = self.w.size()
    
        if lr < 0: # Just flip one bit
            #print("flipping one bit")
            # TODO
            i0 = torch.randint(2, size=(population_size,))
            i1 = torch.arange(population_size)
            i2 = torch.randint(in_features, size=(population_size,))
            i3 = torch.randint(out_features, size=(population_size,))

            og_weights = self.w[i0,0,0,i2,i3]
            #print('og_weights:', og_weights)
            #print("self.w shape:", self.w.shape)
            #torch.cuda.empty_cache()
            self.offspring = self.w.repeat([1, population_size,1,1,1])
            #print('offspring shape:', self.offspring.shape)
            #print('offspring indexed shape:', self.offspring[i0,i1,0,i2,i3])

            self.offspring[i0,i1,0,i2,i3] = (1.0 - og_weights)
            self.indices = (i0, i1, i2, i3)

        else:

            flip_mask = torch.rand((2, population_size, 1, in_features, out_features), device=self.w.device) < lr
            self.offspring = torch.logical_xor(flip_mask, self.w).to(torchtype)

        if self.elite:
            self.offspring[:,0:1,:,:,:] = self.w.clone() # TODO is clone necessary? I doubt it



    def mate(self, parents, num_parents_for_mating, fitness_weights=None):
        # TODO add back in fitness weights
        # TODO add back in some kind of averaging? 
        # can't do the below, for small learning rates it will never change
        #self.w = self.offspring[parents, :, :, :].sum(axis=0,keepdim=True) > len(parents) / 2

        if torch.is_tensor(parents) and (parents.size() == 0): return
        if type(parents) == list and len(parents) == 0: return

        if num_parents_for_mating == 'all':
            #  these are the indices of the bits that have been flipped
            i0,i1,i2,i3 = self.indices

            # take only the bits that have been flipped for the set of parents
            i0 = i0[parents]
            #i1 = i1[parents]
            i2 = i2[parents]
            i3 = i3[parents]

            og_weights = self.w[i0,0,0,i2,i3].clone()
            self.w[i0,0,0,i2,i3] = (1.0 - og_weights)
        else:
            # must clone here, otherwise w retains offspring memory
            self.w = self.offspring[:, parents[0]:parents[0]+1, :, :, :].clone()


    def reset(self):
        del self.offspring # probably not important
        self.offspring = None

    def forward(self, x):

        # duplicate input  and invert it
        x = x.to(torchtype)

        notx = torch.logical_not(x).to(torchtype)

        if self.offspring is None:
            #x = torch.logical_and(x, self.w)

            #print(' xshape:', x.shape)
            #print(' self.w.shape:', self.w.shape)
            x = torch.einsum('pbi,pbio->pbo',x,self.w[0])
            x += torch.einsum('pbi,pbio->pbo',notx,self.w[1])

            # shape should now be [1, batch, input_size, output_size]
        else:
            #x = torch.logical_and(x, self.offspring)

            # convert input into format that works for optimized layer
            assert len(x.shape) == 3

            ## Debugging stuff, not needed anymore
            ## "flatten" the batch and pop dimensions
            #opt_x = x.view((-1, x.shape[2])).cpu().to(torch.bool)
            ## binarize input
            #opt_x = binarize_helper1(opt_x)
            #opt_x = torch.tensor(opt_x).to(self.w.device)
            ## unflatten the batch and pop dimensions
            #opt_x = opt_x.view((x.shape[0], x.shape[1], -1))
            ## run optimized layer

            ## First, we need to check that both are getting the same input.
            #i = 0
            #b = 0
            #o = 0
            #p = 1
            #t = 0
            #print("Input: ", x[p,b,i*64:i*64+64])
            #print("Optimized input: ")
            #print_binarized(opt_x[p,b,i:i+1])

            #print('offpsring shape:', self.offspring.shape)
            #torch.set_printoptions(precision=1, threshold=999999999999)
            #print("Weight: ", self.offspring[t,p,0,i*64:i*64+64,o])
            #print("Optimized weight: ")
            #print_binarized(opt.w[t,p,i,o])

            #opt_r = opt(opt_x)

            x = torch.einsum('pbi,pbio->pbo',x,self.offspring[0])
            x += torch.einsum('pbi,pbio->pbo',notx,self.offspring[1])
            # shape should now be [popsize, batch, input_size, output_size]

        if self.activation == 'none':
            #print("Non-optimized result: ")
            #print(x)
            #print("Optimized result:")
            #print(opt_r)
            return x
        elif self.activation == 'const':
            r =  torch.gt(x, self.w.shape[3] / 2).to(torchtype)
            #print("Non-optimized result: ")
            #print(r)
            #print("Optimized result:")
            ##opt_r = np.unpackbits(opt_r.cpu().numpy().view(np.uint8), bitorder="big").reshape((x.shape[0], x.shape[1], -1))
            #print(opt_r.shape)
            #print_binarized(opt_r)
            return r

        elif self.activation == 'batch_norm':
            if x.shape[1] > 1:
                batch_mean = torch.mean(x, dim=1, keepdim=True)
                pop_batch_mean = torch.mean(batch_mean, dim=0, keepdim=False)
                
                if self.running_means is None:
                    self.running_means = pop_batch_mean
                else:
                    self.running_means = self.gamma * self.running_means + (1 - self.gamma) * pop_batch_mean
                return torch.gt(x, batch_mean)
            else:
                return torch.gt(x, self.running_means)
        else: 
            assert False

        #x = torch.sum(x, dim=2, keepdim=False, dtype=torch.int32)

        # shape is now [batch, popsize, output_size] and has type int32
        #return x


class EvoBinarizedMnistModel(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size = 4096, output_size=10, temperature = 0.8, elite=True, layers=1, activation='const'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        
        self.layers = nn.ModuleList([])
        self.layers.append(EvoBinarizedLayer(in_features=input_size, out_features=hidden_size, elite=elite, activation=activation))
        for _ in range(layers-1):
            self.layers.append(EvoBinarizedLayer(in_features=hidden_size, out_features=hidden_size, elite=elite, activation=activation))
        self.layers.append(EvoBinarizedLayer(in_features=hidden_size, out_features=output_size, elite=elite, activation='none'))

        self.temperature = temperature

    def forward(self, image):

        if type(image) == list: # TODO should we add improvement back in?
            x = image[0]
        else:
            x = image

        #print('image x shape:', x.shape)

        x = x > 0.5 # convert input to binary

        x = x.view((*x.shape[:-2], -1)) # flatten
        #print("x shape after view:", x.shape)

        for i,layer in enumerate(self.layers):
            input_size = x.shape[2]
            x = layer.forward(x)
            

        # TODO would it help to scale x here? maybe convert float(x)^p where p is a parameter or something like that?
        return x ** self.temperature

    def next_generation(self, population_size: int, lr: float):
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayer):
                m.next_generation(population_size, lr)

    def mate(self, parents, **kwargs):
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayer):
                m.mate(parents, **kwargs)

    def reset(self):
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayer):
                m.reset()


    

