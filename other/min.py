
import os
import multiprocessing as mp
import random
from tensorboard import program
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import time
from my_perfect_dataloader import MPD, SyncMPD
from itertools import product

from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

# TODO if I continue using this, make sure to acknowledge the reddit thread I got it from

configs = []
    
#same_batchs=[True, False, True, False, True, False, True, False]
same_batchs=[True]
#lrs = [2.7E-2, 2.7e-3]
lrs = [2.7e-2]
mate_multipliers=[8]
population_sizes = [256]
num_parents_for_matings = [64]
batch_sizes = [500] 
hidden_sizes=[128]
max_generation = 1000000
#max_generation = 200
target_acc = 0.97
prefix = 'test_july21_1'
fitness_weights = True
fitness_types = ['sampled_acc'] # 'cross_entropy','accuracy','sampled_acc'
#load_from = 'saves/min_sampled_july17_1_hs128_sbTrue_lr0.027_mm8_bs500_ps256_mn64_rs3592498957.pt'
load_from = ''
#fitness_types = ['cross_entropy'] # 'cross_entropy','accuracy','sampled_acc'

#prefix = 'colab1'
#same_batchs=[True]
#mate_multipliers=[4]
#batch_sizes = [32]
#population_sizes = [64]
#num_parents_for_matings = [4]


val_batch_size = 500
device = 'cuda' if torch.cuda.is_available() else 'mps'
#device = 'mps'

eval_every_generation = 100
upload_every = 10000

for sb, lr, mm, ps, parents, bs, hs, ft  in product(same_batchs, lrs, mate_multipliers, population_sizes, num_parents_for_matings, batch_sizes, hidden_sizes, fitness_types):
    random_seed = random.randint(0,9999999999)
    configs.append({
        'same_batch': sb,
        'lr': lr,
        'mate_multiplier': mm,
        'population_size': ps,
        'num_parents_for_mating': parents,
        'batch_size': bs,
        'hidden_size': hs,
        'device':'device',
        'val_batch_size': val_batch_size,
        'random_seed': random_seed,
        'max_generation': max_generation,
        'target_acc': target_acc,
        'prefix': prefix,
        'fitness_weights': fitness_weights,
        'fitness_type': ft,
        'load_from': load_from,
        })

class DummyDataloader():
  def __init__(self, batch_size, device):
      self.mult = 25
      self.batch_size = batch_size
      self.device=device
      self.x = torch.normal(mean=0, std=1.0, size=[batch_size*self.mult,28,28], device=device)
      #print('xbytes:', self.x.element_size() * self.x.nelement())
      self.y = torch.randint(low=0,high=10, size=[batch_size*self.mult], device=device)
      #print('ybytes:', self.y.element_size() * self.y.nelement())
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
        #print('input.shape:', input.shape)
        #print('target.shape:', target.shape)
        output = model.forward([input])[0].squeeze()
        #print('output.shape:', output.shape)
        loss += F.cross_entropy(output, target, reduction='sum').item() 
        pred = output.argmax(dim=-1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item() 
        total += input.size(0)

    return loss / total, correct / total



class EvoLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, improvement=False, mate_multiplier=8) -> None:
        """ 
        "improvement" arg means that we are going to be keeping track of last generation's
        weights, and whenever we forward throug the layer we also forward through the previous
        weights, so that we can then eventually compute the difference in fitness between 
        the current weights and the previous weights.
        """
       
        super().__init__()
        self.weight: torch.Tensor
        # TODO should we initialize differently here?
        self.register_buffer('weight', torch.zeros(out_features, in_features))
        self.mate_multiplier = mate_multiplier
        

    def next_generation(self, population_size: int, lr: float):
        out_features, in_features = self.weight.size()
        mean = self.weight.expand(population_size, out_features, in_features) 
        self.offspring = torch.normal(mean, std=lr) 

    def mate(self, parents: list[int], fitness_weights=None):
        if fitness_weights is None:
            adjustment = self.offspring[parents, :, :].mean(0, keepdim=False)  - self.weight
            self.weight = self.weight + self.mate_multiplier * adjustment
            #print(f'adj norm for {self.weight.shape}:', torch.norm(adjustment))
        else:
            # This seems mostly unnecessary, rarely ever happens after first generation.
            fitness_weights = (fitness_weights > 0) * fitness_weights 
            # TODO update this hyperparam, 10 seems best so far (3 diverges)
            fitness_weights = torch.nn.functional.normalize(fitness_weights, dim=None)  / 10
            adjustment  = ((self.offspring[parents, ...] - self.weight[None,...])*fitness_weights[:,None,None]).sum(0, keepdim=False)

            #print(f'adj norm for {self.weight.shape}:', torch.norm(adjustment))
            self.weight = self.weight + self.mate_multiplier * adjustment

    def reset(self):
        self.offspring = None

    def forward(self, x):
        #print('x:', type(x))
        #print(tuple(_.shape for _ in x))
            
        original_results =  F.linear(x[0], self.weight)
        if self.offspring is not None and len(x) > 1:
            mutation_results =  torch.einsum('pbi,poi->pbo', x[1], self.offspring)
            return [original_results, mutation_results]
        return [original_results]



class EvoModel(nn.Module):
    def __init__(self, hidden_size, mate_multiplier):
        super().__init__()
        self.fc1 = EvoLinear(28 * 28, hidden_size, mate_multiplier=mate_multiplier)
        self.fc2 = EvoLinear(hidden_size, 10, mate_multiplier=mate_multiplier)

    def forward(self, x):
        x = self.fc1.forward([_.flatten(2) for _ in x])
        return self.fc2.forward([F.relu(_) for _ in x])  

    def next_generation(self, population_size: int, lr: float):
        for m in self.modules():
            if isinstance(m, EvoLinear):
                m.next_generation(population_size, lr)

    def mate(self, parents: list[int], fitness_weights=None):
        for m in self.modules():
            if isinstance(m, EvoLinear):
                m.mate(parents, fitness_weights)

    def reset(self):
        for m in self.modules():
            if isinstance(m, EvoLinear):
                m.reset()


def upload_to_aws(experiment_name):
    os.system(f'aws s3 cp tensorboards/{experiment_name} s3://luke-genetics/{experiment_name} --recursive')
    os.system(f'aws s3 cp saves/{experiment_name}.pt s3://luke-genetics/{experiment_name}.pt --recursive')



@torch.inference_mode()
def main(config):
    
    # this slows things down when cpus are limited
    #tb = program.TensorBoard()
    #tb.configure(argv=[None, '--logdir', f'./tensorboards/{experiment_name}'])
    #url = tb.launch()
    


    hidden_size = config['hidden_size']
    same_batch = config['same_batch']
    lr = config['lr']
    mate_multiplier = config['mate_multiplier']
    population_size = config['population_size']
    num_parents_for_mating = config['num_parents_for_mating']
    batch_size = config['batch_size']
    same_batch = config['same_batch']
    random_seed = config['random_seed']

    experiment_name=f'min_{prefix}_hs{hidden_size}_sb{same_batch}_lr{lr}_mm{mate_multiplier}_bs{batch_size}_ps{population_size}_mn{num_parents_for_mating}_rs{random_seed}'

    tracking_address = f'./tensorboards/{experiment_name}'
    while os.path.exists(tracking_address):
        experiment_name = experiment_name + '_' + str(random.randint(0,9999999999999))
        tracking_address = f'./tensorboards/{experiment_name}'

    writer = SummaryWriter(log_dir=tracking_address)
    

    if device == 'cuda':
        mp.set_start_method('spawn')

    
    # TODO clean up the mps vs cuda logic
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    else:
        torch.mps.manual_seed(random_seed)
    
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train = datasets.MNIST('./data/mnist', train=True, download=False, transform=transform)
    val = datasets.MNIST('./data/mnist', train=False, transform=transform)
    
    # This complicated loader might not be necessary? 
    # works fine with only one worker and two workers doesn't seem to speed anything up
    #train_loader = SyncMPD(population_size, batch_size, device, same_batch) 

    train_loader = MPD(population_size, batch_size, 1, 2, device, same_batch) 
    #train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=8, persistent_workers=True, shuffle=True, pin_memory=True, drop_last=True)
    print("initialization data")
    t1 = time.time()
    #train_loader = DummyDataloader(batch_size * population_size, 'cpu') 
    #train_loader = DummyDataloader(batch_size * population_size, device) 
    print(f"done in {time.time() - t1}")
    val_loader = torch.utils.data.DataLoader(val, val_batch_size, shuffle=False, pin_memory=False)

    if not config['load_from']:
        model = EvoModel(hidden_size, mate_multiplier)
        model = model.to(device)
    else:
        model,_ = torch.load(config['load_from'], map_location=device)
        print(f"Loaded model from {config['load_from']}")

    generation_count = 0
    epoch = 0
    model.reset()
    loss, accuracy = evaluate(model, val_loader)
    print(f'epoch {epoch} | loss: {loss:.4f} | accuracy: {accuracy:.2%}')
    
    model.eval()
    t0 = time.time()
    
    done = False
    while not done:
        for input, target in train_loader:
            model.next_generation(population_size, lr)

            input, target = input.to(device), target.to(device)

        
            if not batch_size == 'all':
                #input = input.view((population_size, batch_size, *input.shape[1:]))
                #target = target.view((population_size, batch_size, *target.shape[1:]))

                input = input.unsqueeze(0).expand((population_size, *input.shape))
                target = target.unsqueeze(0).expand((population_size, *target.shape))

                original_output, mutation_output = model.forward([input,input])

            else:
                input = input.unsqueeze(0).expand((population_size, *input.shape))
                target = target.unsqueeze(0).expand((population_size, *target.shape))
                original_outputs = []
                mutation_outputs = []
                #for input, target in zip(torch.split(input, 500, dim=1), torch.split(target, 500, dim=1)):
                assert False
                for input in torch.split(input, 500, dim=1):
                    original_output, mutation_output = model.forward([input,input])
                    original_outputs.append(original_output)
                    mutation_outputs.append(mutation_output)
                original_output = torch.concat(original_outputs,dim=1)
                mutation_output = torch.concat(mutation_outputs,dim=1)
        

            generation_count += 1
            #print(generation_count)
        

            if config['fitness_type'] == 'accuracy':
                original_pred = original_output.argmax(dim=-1, keepdim=True) 
                original_correct = original_pred.eq(target.view_as(original_pred)).sum(dim=[1,2])
                mutation_pred = mutation_output.argmax(dim=-1, keepdim=True) 
                mutation_correct = mutation_pred.eq(target.view_as(mutation_pred)).sum(dim=[1,2])

                ave_fitness = mutation_correct.float().mean().item()
                improvement = mutation_correct.float() - original_correct.float()

            elif config['fitness_type'] == 'cross_entropy':
                original_loss = F.cross_entropy(original_output.flatten(0, 1), target.flatten(0,1), reduction='none') 
                mutation_loss = F.cross_entropy(mutation_output.flatten(0, 1), target.flatten(0,1), reduction='none') 
                original_loss = original_loss.unflatten(0, (population_size, batch_size)).mean(dim=-1) 
                mutation_loss = mutation_loss.unflatten(0, (population_size, batch_size)).mean(dim=-1) 
                ave_fitness = -1 * mutation_loss.mean().item()
                improvement = original_loss - mutation_loss

            elif config['fitness_type'] == 'sampled_acc':
                original_probs = torch.nn.functional.softmax(original_output, dim=-1)
                mutation_probs = torch.nn.functional.softmax(mutation_output, dim=-1)

                original_pred = torch.multinomial(original_probs.view(-1,10), num_samples=1).view((population_size, batch_size, 1))
                original_correct = original_pred.eq(target.view_as(original_pred)).sum(dim=[1,2])
                mutation_pred = torch.multinomial(mutation_probs.view(-1,10), num_samples=1).view((population_size, batch_size, 1))
                mutation_correct = mutation_pred.eq(target.view_as(mutation_pred)).sum(dim=[1,2])

                ave_fitness = mutation_correct.float().mean().item() / original_probs.shape[1]
                improvement = mutation_correct.float() - original_correct.float()
            else:
                assert False

            writer.add_scalar('ave_fitness', ave_fitness, generation_count)

            parents = torch.topk(improvement, k=num_parents_for_mating, largest=True).indices.tolist() 
            if config['fitness_weights']:
                model.mate(parents, improvement[parents])
            else:
                model.mate(parents)


            if generation_count % eval_every_generation == 0:
                dt = time.time() - t0
                model.reset()
                #if device == "mps":
                #    torch.mps.synchronize()
                #elif device == 'cuda':
                #    torch.cuda.synchronize()
                loss, accuracy = evaluate(model, val_loader)
                writer.add_scalar('best_val_loss', loss, generation_count)
                writer.add_scalar('best_val_acc', accuracy, generation_count)
                print(f'gen {generation_count} | epoch {epoch} | loss: {loss:.4f} | accuracy: {accuracy:.2%} | seconds per generation: {dt/eval_every_generation:.3f}')
                t0 = time.time()

                torch.save((model,config), open(f'saves/{experiment_name}.pt','wb'))

                if generation_count > config['max_generation'] or accuracy > config['target_acc']: 
                    done=True
                    break
            if generation_count % upload_every == 0:
                print("Uploading to aws...")
                upload_to_aws(experiment_name)
    upload_to_aws(experiment_name)


if __name__ == '__main__':

    #torch.set_num_threads(8)

    #tb = program.TensorBoard()
    #tb.configure(argv=[None, '--logdir', f'./tensorboards'])
    #url = tb.launch()
    for config in configs:
        main(config)
