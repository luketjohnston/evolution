
import multiprocessing as mp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import time
from my_perfect_dataloader import MPD

# TODO if I continue using this, make sure to acknowledge the reddit thread I got it from

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps'

random_seed = 1338

same_batch=True

eval_every_epoch = 10
eval_every_generation = 100

lr = 2.7E-3

batch_size = 500
val_batch_size = 32
population_size = 256
num_parents_for_mating = 64

do_sgd = False

class DummyDataloader():
  def __init__(self, batch_size, device):
      self.mult = 25
      self.batch_size = batch_size
      self.device=device
      self.x = torch.normal(mean=0, std=1.0, size=[batch_size*self.mult,28,28], device=device)
      print('xbytes:', self.x.element_size() * self.x.nelement())
      self.y = torch.randint(low=0,high=10, size=[batch_size*self.mult], device=device)
      print('ybytes:', self.y.element_size() * self.y.nelement())
  def __iter__(self):
      while True:
          r = random.randint(0,self.mult-1)
          yield self.x[r*self.batch_size :(r+1)*self.batch_size], self.y[r*self.batch_size:(r+1)*self.batch_size]


@torch.inference_mode()
def evaluate(model: nn.Module, val_loader):
    model.eval()
    total = 0
    loss = 0
    correct = 0
    for input, target in val_loader:
        input, target = input.to(device), target.to(device)
        print('input.shape:', input.shape)
        #print('target.shape:', target.shape)
        output = model.forward(input).squeeze()
        #print('output.shape:', output.shape)
        loss += F.cross_entropy(output, target, reduction='sum').item() 
        pred = output.argmax(dim=-1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item() 
        total += input.size(0)

    return loss / total, correct / total



class EvoLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight: torch.Tensor
        self.register_buffer('weight', torch.zeros(out_features, in_features))

    def next_generation(self, population_size: int, lr: float):
        out_features, in_features = self.weight.size()
        mean = self.weight.expand(population_size, out_features, in_features) 
        self.offspring = torch.normal(mean, std=lr) 

    def mate(self, parents: list[int]):
        self.weight = self.offspring[parents, :, :].mean(0, keepdim=False) 

    def reset(self):
        self.offspring = None

    def forward(self, x: torch.Tensor):

        if self.offspring is not None:
            return torch.einsum('pbi,poi->pbo', x, self.offspring)
        return F.linear(x, self.weight)
        #return torch.einsum('ab,cb->ac',x,self.weight)

class EvoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = EvoLinear(28 * 28, 128)
        self.fc2 = EvoLinear(128, 10)

    def forward(self, x: torch.Tensor):
        x = self.fc1.forward(x.flatten(2))
        return self.fc2.forward(F.relu(x))  

    def next_generation(self, population_size: int, lr: float):
        for m in self.modules():
            if isinstance(m, EvoLinear):
                m.next_generation(population_size, lr)

    def mate(self, parents: list[int]):
        for m in self.modules():
            if isinstance(m, EvoLinear):
                m.mate(parents)

    def reset(self):
        for m in self.modules():
            if isinstance(m, EvoLinear):
                m.reset()





@torch.inference_mode()
def main():

    if device == 'cuda':
        mp.set_start_method('spawn')

    
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    torch.mps.manual_seed(random_seed)
    
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train = datasets.MNIST('./data/mnist', train=True, download=False, transform=transform)
    val = datasets.MNIST('./data/mnist', train=False, transform=transform)
    
    # This complicated loader might not be necessary? 
    # works fine with only one worker and two workers doesn't seem to speed anything up
    train_loader = MPD(population_size, batch_size, 1, 2, device, same_batch) 
    print("initialization data")
    t1 = time.time()
    #train_loader = DummyDataloader(batch_size * population_size, 'cpu') 
    #train_loader = DummyDataloader(batch_size * population_size, device) 
    print(f"done in {time.time() - t1}")
    val_loader = torch.utils.data.DataLoader(val, val_batch_size, shuffle=False, pin_memory=False)

    model = EvoModel()
    model = model.to(device)

    generation_count = 0
    epoch = 0
    model.reset()
    loss, accuracy = evaluate(model, val_loader)
    print(f'epoch {epoch} | loss: {loss:.4f} | accuracy: {accuracy:.2%}')
    
    while True:
        epoch += 1
    
        model.eval()
        t0 = time.time()
        t1 = time.time()
    
        for input, target in train_loader:
            print(time.time() - t1)
            t1 = time.time()
            #print('yay')
            input, target = input.to(device), target.to(device)
    
            input = input.view((population_size, batch_size, *input.shape[1:]))
            target = target.view((population_size, batch_size, *target.shape[1:]))
    
            generation_count += 1
    
            model.next_generation(population_size, lr)
            output = model.forward(input)

            #print('output:', output.shape)
            #print('target:', target.shape)
            loss = F.cross_entropy(output.flatten(0, 1), target.flatten(0,1), reduction='none') 
            #print('loss.shape:', loss.shape)
            #print("loss unflattened shape:", loss.unflatten(0, (population_size, batch_size)).shape)
            loss = loss.unflatten(0, (population_size, batch_size)).mean(dim=-1) 
            #print(loss.sum())
            #print('loss:', loss)
            parents = torch.topk(loss, k=num_parents_for_mating, largest=False).indices.tolist() 
            model.mate(parents)

            if generation_count % eval_every_generation == 0:
                dt = time.time() -t0
                model.reset()
                if device == "mps":
                    print('synchronizing')
                    torch.mps.synchronize()
                    print('done')
                loss, accuracy = evaluate(model, val_loader)
                print(f'gen {generation_count} | epoch {epoch} | loss: {loss:.4f} | accuracy: {accuracy:.2%} | seconds per generation: {dt/eval_every_generation:.3f}')
                t0 = time.time()

    
        epoch += 1
    
        print(f'Generations: {generation_count}, epochs: {epoch}')
    
        if device == "mps":
            print('synchronizing')
            torch.mps.synchronize()
            print('done')
        elif device == 'cuda':
            print('synchronizing')
            torch.cuda.synchronize()
            print('done')
            
        dt = time.time() - t0
        model.reset()

        if epoch % eval_every_epoch == 0:
            loss, accuracy = evaluate(model)
            print(f'gen {generation_count} | epoch {epoch} | loss: {loss:.4f} | accuracy: {accuracy:.2%} | seconds per epoch: {dt:.3f}')

if __name__ == '__main__':
    main()
