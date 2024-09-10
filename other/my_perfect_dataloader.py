
import numpy as np
import torch
import torchvision
import multiprocessing as mp
from torchvision import datasets, transforms


def worker(queue, population_size, batch_size, device, same_batch, x, y):


  # TODO why doesn't this work with multiprocessing and mps??
  #x = ds.data.to(device) / 255.0
  #y = ds.targets.to(device)

  # let's start very simply, just random sample indices with replacement 
  # TODO ensure indices don't repeat in any single minibatch
  while True:
    if not same_batch:
        indices = torch.randint(low=0, high=x.shape[0], size=(population_size * batch_size,), device=None)
        #print(indices.shape)
        xi = x[indices]
        yi = y[indices]
        #print('xi:', x.shape)
        #print('yi:', y.shape)
        if device == 'cuda':
            # this hasn't been speeding up anything on T4, TODO double check
            #xi = xi.pin_memory(device)
            #yi = yi.pin_memory(device)
            pass
    else:
        if batch_size != 'all':
            indices = torch.randint(low=0, high=x.shape[0], size=(batch_size,), device=None)
            #print(indices.shape)
            xi = x[indices]
            yi = y[indices]
        else:
            xi = x
            yi = y
        if device == 'cuda':
            # this hasn't been speeding up anything, TODO double check
            #xi = xi.pin_memory(device)
            #yi = yi.pin_memory(device)
            pass
        
    queue.put((xi,yi))


class MPD():
    def __init__(self, population_size, batch_size, num_workers, prefetch_num, device, same_batch=False):
    
        ds = torchvision.datasets.MNIST('./data/mnist', train=True, download=False)
        self.x = ds.data / 255.0
        self.y = ds.targets


        print("PREFETCH NUM:", prefetch_num, type(prefetch_num))
        self.queue = mp.Queue(prefetch_num)

        self.processes = [mp.Process(target=worker, args=(self.queue, population_size, batch_size, device, same_batch, self.x, self.y), daemon=True) for _ in range(num_workers)]

        for i,p in enumerate(self.processes):
            p.start()

    def close_pools(self):
        for p in self.processes:
            p.join()

    def __iter__(self):
        while True:
            yield self.queue.get()


# This is slightly slower than the above
class SyncMPD():
    def __init__(self, population_size, batch_size, device, same_batch=False):
        ds = torchvision.datasets.MNIST('./data/mnist', train=True, download=False)
        self.x = ds.data.to(device) / 255.0
        self.y = ds.targets.to(device)
        self.population_size = population_size
        self.batch_size = batch_size
        self.same_batch = same_batch
        self.device = device



    def __iter__(self):
        while True:
          if not self.same_batch:
              indices = torch.randint(low=0, high=self.x.shape[0], size=(self.population_size * self.batch_size,), device=self.device)
              xi = self.x[indices]
              yi = self.y[indices]
          else:
              indices = torch.randint(low=0, high=len(ds), size=(self.batch_size,), device=self.device)
              indices = indices.unsqueeze(0).expand((self.population_size, self.batch_size)).flatten()
              xi = self.x[indices]
              yi = self.y[indices]
              
          yield xi,yi

if __name__ == '__main__':
    mpd = MPD(1000, 1, 1, 'mps')
    for x,y in mpd:
      print(x.shape)
      print(y.shape)


class BinarizedMnistDataloader():
    def __init__(self, device, train):
    
        x, y = get_all_binarized_mnist(train=train)

        self.x = torch.tensor(x).to(device)
        self.y = torch.tensor(y).to(device)

    def __iter__(self):
        while True:
            yield self.x, self.y

    
# Deprecated, use the binarize full dataset method below
# def binarize_mnist(image):
#   x = image / 255.0
#   x = x > 0.5 # convert input to binary
#   x = x.view((*x.shape[:-2], -1)) # flatten
#   if not (x.shape[0] % 64 == 0):
#       x = np.pad(x, (0,64 - (x.shape[0] % 64)), 'constant')
# 
#   dt = np.dtype(np.int64)
#   dt = dt.newbyteorder('big')
# 
#   x = np.frombuffer(np.packbits(x,bitorder='big').data, dtype=dt)
#   return x
    

def get_all_binarized_mnist(train=False):
  transform=transforms.Compose([
      transforms.ToTensor(),
  ])

  if train:
    ds = datasets.MNIST('./data/mnist', train=True, download=False, transform=transform)
  else:
    ds = datasets.MNIST('./data/mnist', train=False, download=False, transform=transform)
  y = ds.targets
  x = ds.data

  x = x > 0.5 # convert input to binary
  x = x.view((*x.shape[:-2], -1)) # flatten
  
  return binarize_helper1(x),y

def binarize_helper1(x):
  batch_size = x.shape[0]
  # assumes x is a boolean numpy array and is flattened already.
  print("X shape before binarize helper:", x.shape)
  if not (x.shape[-1] % 64 == 0):
      x = np.pad(x, ((0,0),(0,64 - (x.shape[-1] % 64))), 'constant')

  dt = np.dtype(np.int64)
  dt = dt.newbyteorder('big')

  x = np.frombuffer(np.packbits(x,bitorder='big').data, dtype=dt)
  x = np.reshape(x, (batch_size, -1))
  x = x.astype(np.int64)
  print("X shape after binarize helper:", x.shape)
  return x
    
    
    
    
