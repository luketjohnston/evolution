import numpy as np
import torch
import torch.nn as nn
from test import print_binarized


torch.ops.load_library('other/build/lib.linux-x86_64-3.8/binary_forward.cpython-38-x86_64-linux-gnu.so')


# TODO everything

torchtype=torch.float

class EvoBinarizedLayerOptimized(nn.Module):
    def __init__(self, in_features: int, out_features: int, elite=True, activation='none'):
        super().__init__()
        self.w: torch.Tensor
        self.elite = elite
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        # TODO add seed?

        in_ints = (in_features + 63) // 64
        self.rounded_in_features = in_ints * 64

        # the first dimension is for whether this w acts on x, or on ~x
        w = torch.randint(low=-2**63, high=2**63-1, size=[2,1,in_ints,out_features], dtype=torch.int64)
        #print('created w :', w)

        self.running_means=None
        
        self.register_buffer('w', w)
        #print('After init w :', w)
        #print('After init w.cuda() :', w.cuda())

    def next_generation(self, population_size: int, lr=-1, verbose=False):
        device = self.w.device
        #print('in next generation,  self w at beginning:', self.w)
        assert lr==-1

        _, _, in_features, out_features = self.w.size()
    
        i0 = torch.randint(2, size=(population_size,), device=device)
        i1 = torch.arange(population_size, device=device)
        i2 = torch.randint(in_features, size=(population_size,), device=device)
        i3 = torch.randint(out_features, size=(population_size,), device=device)

        # I used to put all these into one tuple called "indices" and it just would
        # NOT work, idk why, maybe nn.Module uses indices under the hood for something?
        self.i0 = i0.clone()
        self.i1 = i1.clone()
        self.i2 = i2.clone()
        self.i3 = i3.clone()

        self.w = self.w[:,0:1,:,:].repeat([1, population_size,1,1])

        #print('in next generation,  self w.shape before indexing:', self.w.shape)
        #print(self.w) #works
        #print('w[0,0,0,0]:', self.w[0,0,0,0]) # works
        #print('w[i0[1],i1[1],i2[1],i3[1]]:', self.w[i0[1],i1[1],i2[1],i3[1]]) #
        #print('in next generation,  self w.shape after indexing:', self.w.shape)
        #asdfjk
        rand_exponents = torch.randint(low=0,high=63,size=self.w[i0,i1,i2,i3].size(), device=self.w.device);
        #print('in next generation,  self w after rand_exponents:', self.w)
        bits_to_flip = torch.pow(2, rand_exponents)
        self.bits_to_flip = bits_to_flip


        if self.elite: 
            #print('i0 before elite:', i0)
            i0,i1,i2,i3 = i0[1:], i1[1:], i2[1:], i3[1:]
            #print('i0 after elite:', i0)
            bits_to_flip = bits_to_flip[1:]

        #print('first self w :', self.w)


        if verbose: print("W before mutation:")
        if verbose: print_binarized(self.w)

        if verbose: print("i0,i1,i2,i3:", i0,i1,i2,i3)
        if verbose: print("w[i0,i1,i2,i3]:")
        if verbose: print_binarized(self.w[i0,i1,i2,i3])
        if verbose: print("Bits to flip:")
        if verbose: print_binarized(bits_to_flip)

        #print('bits to flip shape:', bits_to_flip.shape)
        #print('self w shape:', self.w.shape)
        #print('self w device:', self.w.device)
        #print('self w :', self.w)
        #print('self w[[0,0]] :', self.w[[0,0]])
        #print('self w[i0,i1] :', self.w[i0,i1])
        #print('self w[i0,i1,i2] :', self.w[i0,i1,i2])
        #print('self w[i0,i1,i2,i3] shape:', self.w[i0,i1,i2,i3].shape)
        self.w[i0,i1,i2,i3] = torch.bitwise_xor(self.w[i0,i1,i2,i3], bits_to_flip)

        if verbose: print("W after mutation:")
        if verbose: print(self.w.shape)
        if verbose: print_binarized(self.w)
        #print("After next generation self.i0", self.i0)
        #print("After next generation self.i1", self.i1)
        #print("After next generation self.i2", self.i2)
        #print("After next generation self.i3", self.i3)



    # 'mate' produces a single individual in self.w[0]
    # (which is then mutated with next_generation() to extend self.w[population_size])
    def mate(self, parents, num_parents_for_mating, fitness_weights=None):
        # TODO add back in fitness weights
        # TODO add back in some kind of averaging? 
        # can't do the below, for small learning rates it will never change
        #self.w = self.offspring[parents, :, :, :].sum(axis=0,keepdim=True) > len(parents) / 2

        if torch.is_tensor(parents) and (parents.size() == 0): return
        if type(parents) == list and len(parents) == 0: return

        if num_parents_for_mating == 'all':
            #  these are the indices of the bits that have been flipped
            i0,i1,i2,i3 = self.i0, self.i1, self.i2, self.i3

            #print('parents:', parents)

            #print("i0 before:", i0.shape, i0)
            #print("i2 before:", i2.shape, i2)
            #print("i3 before:", i3.shape, i3)

            # take only the bits that have been flipped for the set of parents
            i0 = i0[parents]
            i2 = i2[parents]
            i3 = i3[parents]
            #print("bits_to_flip before indexing:", self.bits_to_flip.shape, self.bits_to_flip)
            bits_to_flip = self.bits_to_flip[parents]

            #print("i0 after:", i0.shape, i0)
            #print("i2 after:", i2.shape, i2)
            #print("i3 after:", i3.shape, i3)
            #print("bits_to_flip:", bits_to_flip.shape, bits_to_flip)
            #print("self.w.shape before indexing: ", self.w.shape)

            self.w[i0,0,i2,i3] = torch.bitwise_xor(self.w[i0,0,i2,i3], bits_to_flip)
        else:
            self.w = self.offspring[:, parents[0]:parents[0]+1, :, :]


    def reset(self):
        self.w = self.w[:,0:1,:,:]

    def forward(self, x):
        #print("Forwarding x with shape ", x.shape) # these look fine
        verbose = False
        if self.activation == 'none':
          # when called with threshold == 0, returns the integer activations
          if verbose: print("Weights:")
          if verbose: print_binarized(self.w)
          if verbose: print("forwarding:")
          if verbose: print_binarized(x)
          r = torch.ops.binary_forward.binary_forward_cuda(x, self.w, 0, False)
          if verbose: print("Result:")
          if verbose: print_binarized(r)
          return r
        elif self.activation == 'const':
          return torch.ops.binary_forward.binary_forward_cuda(x, self.w, self.rounded_in_features // 2, False)
        else:
          assert False


class EvoBinarizedOptimized(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size = 4096, output_size=10, temperature = 0.8, elite=True, layers=1, activation='const'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        
        self.layers = nn.ModuleList([])
        self.layers.append(EvoBinarizedLayerOptimized(in_features=input_size, out_features=hidden_size, elite=elite, activation=activation))
        for _ in range(layers-1):
            self.layers.append(EvoBinarizedLayerOptimized(in_features=hidden_size, out_features=hidden_size, elite=elite, activation=activation))
        self.layers.append(EvoBinarizedLayerOptimized(in_features=hidden_size, out_features=output_size, elite=elite, activation='none'))

        self.temperature = temperature

    def forward(self, x):
        if isinstance(x, list):
            assert len(x) == 1 or len(x) == 2
            x = x[0]

        for i,layer in enumerate(self.layers):
            x = layer.forward(x)
            

        # TODO Is this the best way to scale? Does it depend on hidden layer size etc?
        return x ** self.temperature

    def next_generation(self, population_size: int, lr: float):
        verbose = False
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayerOptimized):
                m.next_generation(population_size, lr, verbose)
                #verbose = True # TODO remove 

    def mate(self, parents, **kwargs):
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayerOptimized):
                m.mate(parents, **kwargs)

    def reset(self):
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayerOptimized):
                m.reset()




