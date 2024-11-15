import numpy as np
import random
import torch
import torch.nn as nn
from test import print_binarized
from torch.nn import BatchNorm1d


try:
  torch.ops.load_library('other/build/lib.linux-x86_64-3.8/binary_forward.cpython-38-x86_64-linux-gnu.so')
except:
  torch.ops.load_library('other/build/lib.linux-x86_64-cpython-311/binary_forward.cpython-311-x86_64-linux-gnu.so')



class EvoBinarizedLayerOptimized(nn.Module):
    def __init__(self, in_features: int, out_features: int, elite=True, activation='none', use_xor=False):
        super().__init__()
        self.w: torch.Tensor
        self.elite = elite
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.use_xor = use_xor
        # TODO add seed?

        in_ints = (in_features + 63) // 64
        self.rounded_in_features = in_ints * 64

        if use_xor:
          self.wdim2 = 1
        else: # use 'and'
          # the size 2 (second to last) dimension is for whether this w acts on x, or on ~x
          self.wdim2 = 2

        w = torch.randint(low=-2**63, high=2**63-1, size=[1,in_ints,self.wdim2,out_features], dtype=torch.int64)
        
        self.register_buffer('w', w)
        if activation == 'const':
          self.thresh = self.rounded_in_features // 2 + 1 # kernel uses >= for thresholding
          print("Making layer with threshold: ", self.thresh)

    def next_generation(self, population_size: int, lr=-1, mutate=True):
        self.all_r = []
        self.total_forwarded = 0
        device = self.w.device
        #print('in next generation,  self w at beginning:', self.w)
        assert lr==-1

        _, in_features, _, out_features = self.w.size()

        self.w = self.w[0:1,:,:,:].repeat([population_size,1,1,1])

        if mutate:
    
            i0 = torch.arange(population_size, device=device)
            i1 = torch.randint(in_features, size=(population_size,), device=device)
            i2 = torch.randint(self.wdim2, size=(population_size,), device=device)
            i3 = torch.randint(out_features, size=(population_size,), device=device)

            # I used to put all these into one tuple called "indices" and it just would
            # NOT work, idk why, maybe nn.Module uses indices under the hood for something?
            self.i0 = i0.clone()
            self.i1 = i1.clone()
            self.i2 = i2.clone()
            self.i3 = i3.clone()


            rand_exponents = torch.randint(low=0,high=63,size=self.w[i0,i1,i2,i3].size(), device=self.w.device, dtype=torch.int64);
            bits_to_flip = torch.pow(2, rand_exponents) # TODO replace this with left shift?
            self.bits_to_flip = bits_to_flip


            if self.elite: 
                i0,i1,i2,i3 = i0[1:], i1[1:], i2[1:], i3[1:]
                bits_to_flip = bits_to_flip[1:]

            self.w[i0,i1,i2,i3] = torch.bitwise_xor(self.w[i0,i1,i2,i3], bits_to_flip)



    # 'mate' produces a single individual in self.w[0]
    # (which is then mutated with next_generation() to extend self.w[population_size])
    def mate(self, parents, num_parents_for_mating, fitness_weights=None):
        # TODO add back in some kind of averaging? 

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
            i1 = i1[parents]
            i2 = i2[parents]
            i3 = i3[parents]
            #print("bits_to_flip before indexing:", self.bits_to_flip.shape, self.bits_to_flip)
            bits_to_flip = self.bits_to_flip[parents]

            #print("i0 after:", i0.shape, i0)
            #print("i2 after:", i2.shape, i2)
            #print("i3 after:", i3.shape, i3)
            #print("bits_to_flip:", bits_to_flip.shape, bits_to_flip)
            #print("self.w.shape before indexing: ", self.w.shape)

            self.w[0,i1,i2,i3] = torch.bitwise_xor(self.w[0,i1,i2,i3], bits_to_flip)
        elif num_parents_for_mating == 1:
            self.w[0,:,:,:] = self.w[parents[0], :,:, :].clone()
            #self.w = self.w[:, parents[0]:parents[0]+1, :, :].clone()
        else:
            assert False

        #print('w shape after mate:', self.w.shape)



    def reset(self):
        self.w = self.w[0:1,:,:,:]

    def forward(self, x, show_dead=False):
        #print("Forwarding x with shape ", x.shape) # these look fine

        if self.activation == 'none':
          # when called with threshold == 0, returns the integer activations
          r = torch.ops.binary_forward.binary_forward_cuda(x, self.w, 0, False, self.use_xor)
              
          return r
        elif self.activation == 'const':
          r = torch.ops.binary_forward.binary_forward_cuda(x, self.w, self.thresh, False, self.use_xor)

          if show_dead:
               #print("Reduction with or: 0s are always 0:")
               #print_binarized(r[0,:,0])

               try:
                 self.all_r.append(r.cpu().numpy())
               except:
                 return r

               self.total_forwarded  += r.shape[1]

               if self.total_forwarded >= 60000:
                   myr = np.concatenate(self.all_r, axis=1)
                   print('myr shape:', myr.shape)
                   self.all_r = []
                   self.total_forwarded = 0
                 

                   a1 = np.bitwise_or.reduce(myr, axis=1)
                   #print('a1 shape:', a1.shape)
                   #print('unpacked bits shape:', np.unpackbits(a1.view(np.uint8)).shape)
                   total_always_zero = np.sum( (1 - np.unpackbits(a1.view(np.uint8)).reshape((r.shape[0], -1))), axis=1)
                   #print("Total that are always 0: ",  total_always_zero)

                   #print("Reduction with and: 1s are always 1:")
                   a2 = np.bitwise_and.reduce(myr, axis=1)
                   #print('a2 shape:', a2.shape)
                   #print('unpacked bits shape:', np.unpackbits(a2.view(np.uint8)).shape)
                   total_always_one = np.sum( (np.unpackbits(a2.view(np.uint8)).reshape((r.shape[0], -1))), axis=1)

                   #print("Total that are always 1: ",  total_always_one)
                   print(f"Fraction dead: {total_always_zero} + {total_always_one} / {a2.shape[1] * 64} = {(total_always_zero + total_always_one) / (a2.shape[1] * 64)}")
          return r
        else:
          assert False


class EvoBinarizedOptimized(nn.Module):
    #def __init__(self, input_size = 28*28*8, hidden_size = 4096, output_size=10, temperature = 0.8, elite=True, layers=1, activation='const'):
    def __init__(self, input_size = 28*28, hidden_size = 4096, output_size=10, temperature = 0.8, elite=True, layers=1, activation='const', use_xor=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.temperature = temperature
        self.elite = elite
        self.num_layers = layers
        self.use_xor = use_xor
        self.init_layers()

    def init_layers(self):
        self.layers = nn.ModuleList([])
        self.layers.append(EvoBinarizedLayerOptimized(in_features=self.input_size, out_features=self.hidden_size, elite=self.elite, activation=self.activation, use_xor=self.use_xor))
        for _ in range(self.num_layers-1):
            self.layers.append(EvoBinarizedLayerOptimized(in_features=self.hidden_size, out_features=self.hidden_size, elite=self.elite, activation=self.activation, use_xor=self.use_xor))
        self.layers.append(EvoBinarizedLayerOptimized(in_features=self.hidden_size, out_features=self.output_size, elite=self.elite, activation='none', use_xor=self.use_xor))

    def forward(self, x):
        if isinstance(x, list):
            assert len(x) == 1 or len(x) == 2
            x = x[0]

        for i,layer in enumerate(self.layers):
            x = layer.forward(x)
            

        # TODO Is this the best way to scale? Does it depend on hidden layer size etc?
        return x ** self.temperature

    def next_generation(self, population_size: int, lr: float):

        layer_to_mutate = random.randrange(self.num_layers + 1)
        i = 0

        for j,m in enumerate(self.layers):
            if isinstance(m, EvoBinarizedLayerOptimized):
                m.next_generation(population_size, lr, mutate=(i==layer_to_mutate))
                i+=1

    def mate(self, parents, **kwargs):
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayerOptimized):
                m.mate(parents, **kwargs)

    def reset(self):
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayerOptimized):
                m.reset()

def mean_of_long(x, dim):
    s = torch.sum(x, dim=dim, keepdims=True)
    r =  torch.floor_divide(s, x.shape[dim])
    return r

class BatchCenterZero(nn.Module):
    def __init__(self, epsilon = 0.95):
        super().__init__()
        self.running_means = None  # TODO can we just set to 0 and let broadcasting do the work
        self.epsilon = epsilon
    def forward(self, x):
        #print("Before batch center 0:", x[0,0,0:64])
        if self.training:
            # TODO taking two means like this is not precisely accurate, should compute all at once
            means_per_pop = mean_of_long(x, dim=1)
            if self.running_means is None:
                self.running_means = means_per_pop[0:1]
            else:
                self.running_means = self.running_means * self.epsilon + (1 - self.epsilon) * means_per_pop[0:1]
            r = x - means_per_pop
        else:
            #print(self.running_means)
            #print("In eval x before: ", x)
            r =  x - (self.running_means.to(torch.int64))
            #print("In eval r after: ", r)
        #print("After batch center 0:", r[0,0,0:64])
        return r

class LongBatchNorm(nn.Module):
    def __init__(self, epsilon = 0.95):
        super().__init__()
        self.running_means = None  # TODO can we just set to 0 and let broadcasting do the work
        self.epsilon = epsilon
    def forward(self, x):
        #print("Before batch center 0:", x[0,0,0:64])
        if self.training:
            # TODO taking two means like this is not precisely accurate, should compute all at once
            means_per_pop = mean_of_long(x, dim=1)
            if self.running_means is None:
                self.running_means = means_per_pop[0:1]
            else:
                self.running_means = self.running_means * self.epsilon + (1 - self.epsilon) * means_per_pop[0:1]
            r = x - means_per_pop
        else:
            #print(self.running_means)
            #print("In eval x before: ", x)
            r =  x - (self.running_means.to(torch.int64))
            #print("In eval r after: ", r)
        #print("After batch center 0:", r[0,0,0:64])
        return r

class SubtractLayer(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    def forward(self, x):
        #print('After sub layer, total above 0:',  torch.sum((x - self.threshold) >= 0) / torch.numel(x))
        return x - self.threshold

class MyBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = BatchNorm1d(num_features).cuda()

    def forward(self, x):
        (p,b,n) = x.shape
        x = x.float()
        x = self.bn(x.view([-1,n])).long()
        return x.view([p,b,n])

class ScaleLayer(nn.Module):
    def __init__(self, in_features, scale_to):
        super().__init__()
        self.in_features = in_features
        self.scale_to = scale_to

    def forward(self, x):
        #print("\nbefore mean scale to: ", torch.mean(x.float()))
        #print("before var scale to: ", torch.var(x.float()))
        #print("in features:", self.in_features)
        #print("scale_to:", self.scale_to)
        r =  torch.floor_divide((x * self.scale_to), (self.in_features // 4)**0.5).long()
        #print("after mean scale to: ", torch.mean(r.float()))
        #print("after var scale to: ", torch.var(r.float()))
        return r

class ConsolidateBits(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        #print("Before consolidate bits: ", x[-1,1111,-128:-64])
        r =  torch.ops.binary_forward.consolidate_bits_cuda(x)
        #print("After consolidate bits: ")
        #print_binarized(r[-1,1111,-2])
        return r


class EvoBinarizedOptimizedImproved1(EvoBinarizedOptimized):
    def __init__(self, input_size = 28*28, hidden_size = 4096, output_size=10, temperature = 0.8, elite=True, layers=1, activation='const', use_xor=False, residual=True, normalization='batch_norm'):
        self.residual = residual
        self.normalization = normalization
        if normalization == 'batch_center':
            self.regularization_layer = BatchCenterZero
        elif normalization == 'scale':
            self.regularization_layer = ScaleLayer
        elif normalization == 'batch_norm':
            self.regularization_layer = MyBatchNorm
        else:
            assert False

        super().__init__(input_size, hidden_size, output_size, temperature, elite, layers, activation, use_xor)


    def init_layers(self):

        self.layers = nn.ModuleList([])

        rounded_in_features = ((self.input_size + 63) // 64) * 64
        layer_input_features = rounded_in_features
        for _ in range(self.num_layers):
            self.layers.append(EvoBinarizedLayerOptimized(in_features=layer_input_features, out_features=self.hidden_size, elite=self.elite, activation='none', use_xor=self.use_xor))

            if self.normalization == 'batch_center':
              self.layers.append(BatchCenterZero())
            elif self.normalization == 'scale':
              self.layers.append(SubtractLayer(layer_input_features // 2 + 1))
              self.layers.append(ScaleLayer(layer_input_features, 100))
              # need to scale activations to all be the same magnitude
            elif self.normalization == 'batch_norm':
              self.layers.append(MyBatchNorm(self.hidden_size))

            layer_input_features = self.hidden_size

            self.layers.append(ConsolidateBits())

        self.layers.append(EvoBinarizedLayerOptimized(in_features=self.hidden_size, out_features=self.output_size, elite=self.elite, activation='none', use_xor=self.use_xor))

    def forward(self, x):
        if isinstance(x, list):
            assert len(x) == 1 or len(x) == 2
            x = x[0]

        last_input = torch.zeros([x.shape[0], x.shape[1], self.hidden_size], dtype=torch.long, device=x.device)
        for i,layer in enumerate(self.layers):

            x = layer.forward(x)

            # for residual connection
            if self.residual and isinstance(layer, self.regularization_layer) and last_input.shape == x.shape:
                #print("Adding residual", last_input)
                x += last_input
                last_input = x
            

        # TODO Is this the best way to scale? Does it depend on hidden layer size etc?
        return x ** self.temperature

        
        
        



