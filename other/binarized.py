import numpy as np
import torch
import torch.nn as nn


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
    def __init__(self, in_features: int, out_features: int, elite=True):
        super().__init__()
        self.w: torch.Tensor
        self.elite = elite
        # TODO add seed?

        # the first dimension is for whether this w acts on x, or on ~x
        # the next two dimensions are the population dimension and the batch dimension
        w = (torch.rand([2,1,1,in_features,out_features]) > 0.5).to(torchtype)
        self.offspring=None

        self.register_buffer('w', w)

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
            self.offspring = self.w.repeat([1, population_size,1,1,1])
            #print('offspring shape:', self.offspring.shape)
            #print('offspring indexed shape:', self.offspring[i0,i1,0,i2,i3])

            self.offspring[i0,i1,0,i2,i3] = (1.0 - og_weights)
            

        else:

            flip_mask = torch.rand((2, population_size, 1, in_features, out_features), device=self.w.device) < lr
            self.offspring = torch.logical_xor(flip_mask, self.w).to(torchtype)

        if self.elite:
            self.offspring[:,0:1,:,:,:] = self.w.clone() # TODO is clone necessary? I doubt it



    def mate(self, parents: list[int], fitness_weights=None):
        # TODO add back in fitness weights
        # TODO add back in some kind of averaging? 
        # can't do the below, for small learning rates it will never change
        #self.w = self.offspring[parents, :, :, :].sum(axis=0,keepdim=True) > len(parents) / 2
        self.w = self.offspring[:, parents[0]:parents[0]+1, :, :, :]

    def reset(self):
        self.offspring = None

    def forward(self, x):

        # duplicate input  and invert it
        x = x.to(torchtype)

        notx = torch.logical_not(x).to(torchtype)

        if self.offspring is None:
            #x = torch.logical_and(x, self.w)

            x = torch.einsum('pbi,pbio->pbo',x,self.w[0])
            x += torch.einsum('pbi,pbio->pbo',notx,self.w[1])

            # shape should now be [1, batch, input_size, output_size]
        else:
            #x = torch.logical_and(x, self.offspring)

            x = torch.einsum('pbi,pbio->pbo',x,self.offspring[0])
            x += torch.einsum('pbi,pbio->pbo',notx,self.offspring[1])
            # shape should now be [popsize, batch, input_size, output_size]

        #x = torch.sum(x, dim=2, keepdim=False, dtype=torch.int32)

        # shape is now [batch, popsize, output_size] and has type int32
        return x


class EvoBinarizedMnistModel(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size = 4096, output_size=10, temperature = 0.8, elite=True, layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.layers = nn.ParameterList([])
        self.layers.append(EvoBinarizedLayer(in_features=input_size, out_features=hidden_size, elite=elite))
        for _ in range(layers-1):
            self.layers.append(EvoBinarizedLayer(in_features=hidden_size, out_features=hidden_size, elite=elite))
        self.layers.append(EvoBinarizedLayer(in_features=hidden_size, out_features=output_size, elite=elite))

        self.temperature = temperature

    def forward(self, image):

        if type(image) == list: # TODO should we add improvement back in?
            x = image[0]
        else:
            x = image

        x = x > 0.5 # convert input to binary
        x = x.view((*x.shape[:-2], -1)) # flatten

        for i,layer in enumerate(self.layers):

            input_size = x.shape[2]
            x = layer.forward(x)

            # assume that half of x.shape[2] are True.
            # assume that half of w are True.

            # now we just need to threshold X somehow. Note that since the input has x and ~x,
            # if all weights are 1, then the output will be input_size. Note also that this
            # is the maximum output. If we set the threshold to input_size / 2, we have only 
            # two concerns:
            # 1. it is difficult for the network to learn 'small' features
            #    --- this is solved, the network can easily 'pad' outputs with (xn + ~xn)
            #        using features xn that are unimportant to the small feature
            # 2. it is difficult for the network to learn 'large' features (close to input_size)
            #    --- this is unlikely to be necessary, and furthermore the network can learn
            #        these large features easily using more than one layer 
            #        (first_half_of_large_feature + second_half_of_large_feature)
 
            if not i == len(self.layers) - 1:
                #print(f'mean: {torch.mean(x.float())}, thresh: {input_size / 2}')
                x = torch.gt(x, input_size / 2).to(torchtype)
                #print(f"x.shape {x.shape} with activations {torch.sum(x)}")

                #print('activation percent: ', torch.sum(x) / (x.shape[0] * x.shape[1] * x.shape[2]))

        # TODO would it help to scale x here? maybe convert float(x)^p where p is a parameter or something like that?
        return x ** self.temperature

    def next_generation(self, population_size: int, lr: float):
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayer):
                m.next_generation(population_size, lr)

    def mate(self, parents: list[int], fitness_weights=None):
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayer):
                m.mate(parents, fitness_weights)

    def reset(self):
        for m in self.modules():
            if isinstance(m, EvoBinarizedLayer):
                m.reset()


    

