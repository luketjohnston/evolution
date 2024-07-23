
import torch
import torchvision
import multiprocessing as mp


def worker(queue, population_size, batch_size, device, same_batch, x, y):


  # TODO why doesn't this work with multiprocessing and mps??
  #x = ds.data.to(device) / 255.0
  #y = ds.targets.to(device)

  # let's start very simply, just random sample indices with replacement 
  # TODO ensure indices don't repeat in any single minibatch
  while True:
    if not same_batch:
        indices = torch.randint(low=0, high=len(ds), size=(population_size * batch_size,), device=None)
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
            indices = torch.randint(low=0, high=len(ds), size=(batch_size,), device=None)
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



    
    
    
    
