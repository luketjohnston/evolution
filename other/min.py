
import os
import numpy as np
import multiprocessing as mp
import random
from tensorboard import program
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import time
from my_perfect_dataloader import MPD, SyncMPD, BinarizedMnistDataloader
from itertools import product

from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from binarized import EvoBinarizedMnistModel
from evo import EvoModel
from optimized_binary import EvoBinarizedOptimized

from torch.profiler import profile, record_function, ProfilerActivity

# TODO if I continue using this, make sure to acknowledge the reddit thread I got it from

configs = []
target_acc = 0.99
    
##same_batchs=[True, False, True, False, True, False, True, False]
same_batchs=[True]
##lrs = [2.7E-2, 2.7e-3]
#lrs = [2.7e-2]
mate_multipliers=[-1]
#population_sizes = [4]
##population_sizes = [256]
##num_parents_for_matings = [4,16,64]
#num_parents_for_matings = [64]
#batch_sizes = [500] 
#hidden_sizes=[128]
#max_generation = 30000*4
##max_generation = 200
##prefix = 'test'
#prefix = 'test'
#fitness_weights = True
#fitness_types = ['sampled_acc'] # 'cross_entropy','accuracy','sampled_acc'
#model_type = 'EvoModel'
#load_from = 'saves/copy/min_july22_hyperparam_search3_hs128_sbTrue_lr0.027_mm8_bs500_ps1024_mn256_rs5096758358.pt' # 93.5 val acc
#elite=False


# TODO for some reason we are getting "CUDA an illegal memory access was encountered"
# when pop = 1024 and hidden_sizes = 128. If we reduce either the error disappears.
# possibly just reaching max the gpu can handle?
# the weird thing is when I run with the same params in the kernel directly I get no error
# (see STRESS TEST)
# - still happens with pop 512, hs 128 mb 4096

# 5e-7 expects 20 mutations
# 5e-8 expects 2, not much point going any lower than this
# seems like 3e-8 actually works the best. 
lrs = [-1]
#lrs = [0.0001 * 0.03]
# Note that if elite = True, population_size needs to be more than 1
population_sizes = [16]
#population_sizes = [1024]
#num_parents_for_matings = ['all']
num_parents_for_matings = [1]
batch_sizes = ['all']
# This is already at full gpu-util so no need to increase further
minibatch_size = 512
hidden_sizes=[4096] 
fitness_weights=False
fitness_types = ['cross_entropy'] # 'cross_entropy','accuracy','sampled_acc'
#load_from = 'saves/copy/min_binary_oneflip_all_aug13_2_hs4096_sbTrue_lr-1_mm8_bsall_ps16_mn1_rs6723654141_dcuda.pt'
#load_from = 'saves/copy/min_binary_oneflip_all_aug13_1_hs4096_sbTrue_lr-1_mm8_bsall_ps16_mn1_rs2282136183_dcuda.pt'
load_from=''
#load_from = 'saves/min_optimized_test1.1_hs4096_sbTrue_lr-1_mm-1_bsall_ps16_np1_rs4414047074_dcuda_const.pt'
#load_from = 'saves/copy/min_binary_bnorm1_aug15_hs4096_sbTrue_lr-1_mm8_bsall_ps16_npall_rs6712504267_dcuda_batch_norm.pt'
elite=True
layers=2
max_generation = 200000000
#activation='batch_norm'
activation='const'

prefix = 'optimized_test1.2'
model_type = 'EvoBinarizedOptimized'

prefix = 'nonoptimized_test1'
model_type = 'EvoBinarized'


#load_from = 'saves/copy/min_july22_hyperparam_search3_hs128_sbTrue_lr0.027_mm8_bs500_ps1024_mn256_rs5096758358.pt' # 93.5 val acc

#fitness_types = ['cross_entropy'] # 'cross_entropy','accuracy','sampled_acc'

val_batch_size = 500
device = 'cuda' if torch.cuda.is_available() else 'mps'
#device = 'cpu'

eval_every_time = 300 # evaluation is done at specific time intervals, so it doesn't affect
                     # the time-based comparison between different hyperparams. In seconds.

eval_every_time=10
save_every_time=600


upload_every = 50000

for sb, lr, mm, ps, parents, bs, hs, ft  in product(same_batchs, lrs, mate_multipliers, population_sizes, num_parents_for_matings, batch_sizes, hidden_sizes, fitness_types):
    random_seed = random.randint(0,9999999999)

    if mm == 8 and parents == 0.25 and lr == 2.7e-2: 
        print("Skipping config, already tested...")
        continue 

    if (not type(parents) == str) and parents < 1:
        nump = int(parents * ps)
    else:
        nump = parents

    configs.append({
        'same_batch': sb,
        'lr': lr,
        'mate_multiplier': mm,
        'population_size': ps,
        'num_parents_for_mating': parents,
        'batch_size': bs,
        'hidden_size': hs,
        'device':device,
        'val_batch_size': val_batch_size,
        'random_seed': random_seed,
        'max_generation': max_generation,
        'target_acc': target_acc,
        'prefix': prefix,
        'fitness_weights': fitness_weights,
        'fitness_type': ft,
        'load_from': load_from,
        'elite': elite,
        'layers': layers,
        'minibatch_size': minibatch_size,
        })


# For profiling
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")


#@torch.inference_mode()
def evaluate(model: nn.Module, val_loader, verbose=True):
    t1 = time.time()
    model.eval()
    total = 0
    loss = 0
    correct = 0
    for input, target in val_loader:
        input, target = input.to(device), target.to(device)
        if len(input.shape) == 4:
          input=input.squeeze() # remove channel dimension

        # add population dimension
        input = input.unsqueeze(0)


        output = model.forward([input])
        if type(output) is list:
            output = output[0]
        output = output.squeeze()
        #print('output.shape:', output.shape)
        #print('target.shape:', target.shape)

        #print('output.shape:', output.shape)
        #print('target.shape:', target.shape)
        #print('output:', output)
        #print('target:', target)
 
        loss += F.cross_entropy(output, target, reduction='sum').item() 

        #print('output: ', output)
        probs = torch.softmax(output, dim=1)
        #print('probs: ', probs[0])
        pred = output.argmax(dim=-1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item() 
        total += target.size(0)

        del output # remove these so they no longer take memory, probably not important
        del input
        del target
        if total >= 10000: break
    if verbose: print(f"Eval time: {time.time() -t1}")

    return loss / total, correct / total





def upload_to_aws(experiment_name):
    os.system(f'aws s3 cp tensorboards/{experiment_name} s3://luke-genetics/{experiment_name} --recursive')
    os.system(f'aws s3 cp saves/{experiment_name}.pt s3://luke-genetics/{experiment_name}.pt')

# Make sure model is on cpu before saving otherwise it will take tons of space
def save_model(model, config, experiment_name, verbose=True):
    if verbose: print(f"Saving model to saves/{experiment_name}.pt ..."); t1 = time.time()
    torch.save((model,config), open(f'saves/{experiment_name}.pt','wb'))
    if verbose: print(f'done save in {time.time() - t1}')

def compute_loss(original_output, mutation_output, target, config):
    population_size = config['population_size']
    batch_size = original_output.shape[1]

    if config['fitness_type'] == 'accuracy':
        original_pred = original_output.argmax(dim=-1, keepdim=True) 
        original_correct = original_pred.eq(target.view_as(original_pred)).sum(dim=[1,2])
        mutation_pred = mutation_output.argmax(dim=-1, keepdim=True) 
        mutation_correct = mutation_pred.eq(target.view_as(mutation_pred)).sum(dim=[1,2])
    
        ave_fitness = mutation_correct.float().mean().item()
        improvements = mutation_correct.float() - original_correct.float()
    
    elif config['fitness_type'] == 'cross_entropy':
        #print("Original output shape:", original_output.shape)
        #print("Mutation  output shape:", mutation_output.shape)

        original_loss = F.cross_entropy(original_output.flatten(0, 1), target.flatten(0,1), reduction='none') 
        mutation_loss = F.cross_entropy(mutation_output.flatten(0, 1), target.flatten(0,1), reduction='none') 
        #print("Mutation output:", mutation_output.shape, mutation_output)

        original_loss = original_loss.unflatten(0, (population_size, batch_size)).mean(dim=-1) 
        mutation_loss = mutation_loss.unflatten(0, (population_size, batch_size)).mean(dim=-1) 
        #print("Mutation loss:", mutation_output.shape, mutation_output)
        ave_fitness = -1 * mutation_loss.mean().item()
    
    
        if model_type == 'EvoBinarized'  or model_type == 'EvoBinarizedOptimized':
            #print("ML:", mutation_loss)
            #print("ML[0]:", mutation_loss[0])
            improvements = - mutation_loss + mutation_loss[0] 
        else:
            improvements = original_loss - mutation_loss
    
    elif config['fitness_type'] == 'sampled_acc':
        original_probs = torch.nn.functional.softmax(original_output, dim=-1)
        mutation_probs = torch.nn.functional.softmax(mutation_output, dim=-1)
    
        original_pred = torch.multinomial(original_probs.view(-1,10), num_samples=1).view((population_size, batch_size, 1))
        original_correct = original_pred.eq(target.view_as(original_pred)).sum(dim=[1,2])
        mutation_pred = torch.multinomial(mutation_probs.view(-1,10), num_samples=1).view((population_size, batch_size, 1))
        mutation_correct = mutation_pred.eq(target.view_as(mutation_pred)).sum(dim=[1,2])
    
        ave_fitness = mutation_correct.float().mean().item() / original_probs.shape[1]
        improvements = mutation_correct.float() - original_correct.float()
    else:
        assert False
    return improvements, ave_fitness



#@torch.inference_mode()
def main(config):

 with torch.no_grad():
    
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

    experiment_name=f'min_{prefix}_hs{hidden_size}_sb{same_batch}_lr{lr}_mm{mate_multiplier}_bs{batch_size}_ps{population_size}_np{num_parents_for_mating}_rs{random_seed}_d{device}_{activation}'

    tracking_address = f'./tensorboards/{experiment_name}'
    while os.path.exists(tracking_address):
        experiment_name = experiment_name + '_' + str(random.randint(0,9999999999999))
        tracking_address = f'./tensorboards/{experiment_name}'

    writer = SummaryWriter(log_dir=tracking_address)
    


    
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
    
    #train = datasets.MNIST('./data/mnist', train=True, download=False, transform=transform)
    
    # This complicated loader might not be necessary? 
    # works fine with only one worker and two workers doesn't seem to speed anything up
    #train_loader = SyncMPD(population_size, batch_size, device, same_batch) 

    if model_type == 'EvoBinarizedOptimized':
      train_loader = BinarizedMnistDataloader(device, train=True)
      val_loader = BinarizedMnistDataloader(device, train=False)
      assert batch_size == 'all'
    else:
      train_loader = MPD(population_size, batch_size, 1, 2, device, same_batch) 
      val = datasets.MNIST('./data/mnist', train=False, transform=transform)
      val_loader = torch.utils.data.DataLoader(val, val_batch_size, shuffle=False, pin_memory=False)

    #train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=8, persistent_workers=True, shuffle=True, pin_memory=True, drop_last=True)
    print("initialization data")
    t1 = time.time()
    print(f"done in {time.time() - t1}")


    if not config['load_from']:
        if model_type == 'EvoModel':
            model = EvoModel(hidden_size, mate_multiplier)
            model = model.to(device)
        elif model_type == 'EvoBinarized':
            model = EvoBinarizedMnistModel(hidden_size=hidden_size, elite=config['elite'], layers=config['layers'], activation=activation)
            model = model.to(device)
        elif model_type == 'EvoBinarizedOptimized':
            model = EvoBinarizedOptimized(hidden_size=hidden_size, elite=config['elite'], layers=config['layers'], activation=activation)
            model = model.to(device)
        else:
            assert False
    else:
        model,_ = torch.load(config['load_from'], map_location=device)
        print(f"Loaded model from {config['load_from']}")

    # TODO make it so model.parameters() works
    param_count = sum([np.prod(layer.w.size()) for layer in model.layers])
    print('Param count: ', param_count)
    print('Expected flips per mutation: ', config['lr'] * param_count)

    generation_count = 0
    model.reset()
    #loss, accuracy = evaluate(model, val_loader)
    #print(f'loss: {loss:.4f} | accuracy: {accuracy:.2%}')
    
    model.eval()
    t0 = time.time()
    start = t0
    last_eval = start
    last_save = start
    last_eval_generation = 0


    #with profile(
    #       activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #       schedule=torch.profiler.schedule(
    #           wait=3,
    #           warmup=2,
    #           active=3),
    #       profile_memory=True,
    #       on_trace_ready=trace_handler
    #       ) as p:
    if True:
    
        done = False
        while not done:
            for input, target in train_loader:
                model.next_generation(population_size, lr)
                input, target = input.to(device), target.to(device)

            
                #if not batch_size == 'all':
                if True:
                    #input = input.view((population_size, batch_size, *input.shape[1:]))
                    #target = target.view((population_size, batch_size, *target.shape[1:]))

                    minibatch_x = torch.split(input, config['minibatch_size'], dim=0)
                    minibatch_y = torch.split(target, config['minibatch_size'], dim=0)
                    del input # probably not important to have these del lines
                    del target
                    num_minibatches = len(minibatch_x)


                    improvements = torch.zeros(population_size, device=device)
                    ave_fitness = 0

                    for input, target in zip(minibatch_x, minibatch_y):
                        input = input.unsqueeze(0).expand((population_size, *input.shape))
                        target = target.unsqueeze(0).expand((population_size, *target.shape))

                        r = model.forward([input,input])
                        #print("Output of model shape:", r.shape)
                        #print("Output of model:", r)
                        if type(r) is tuple or type(r) is list:
                            original_output, mutation_output = r
                        else:
                            original_output, mutation_output = r.float(), r.float()
                        #print("mutation output after convert to float:", mutation_output)

                        improvements_mb, ave_fitness = compute_loss(original_output, mutation_output, target, config)
                        #print("Improvements_mb:", improvements_mb)
                        improvements += improvements_mb
                        ave_fitness += ave_fitness / num_minibatches
                        del original_output # remove these so they don't take memory anymore, probably not important
                        del mutation_output
                        del input
                        del target
                        #print(improvements)


                    #assert(torch.sum(mutation_output[0,0] == mutation_output[0,1])  < mutation_output[0,1].shape[0])
                    #print('mutation_output[0,0] == mutation_output[0,2]:', mutation_output[0,0] == mutation_output[0,2])



                #else:
                #    input = input.unsqueeze(0).expand((population_size, *input.shape))
                #    target = target.unsqueeze(0).expand((population_size, *target.shape))
                #    original_outputs = []
                #    mutation_outputs = []
                #    #for input, target in zip(torch.split(input, 500, dim=1), torch.split(target, 500, dim=1)):
                #    assert False
                #    for input in torch.split(input, 500, dim=1):
                #        original_output, mutation_output = model.forward([input,input])
                #        original_outputs.append(original_output)
                #        mutation_outputs.append(mutation_output)
                #    original_output = torch.concat(original_outputs,dim=1)
                #    mutation_output = torch.concat(mutation_outputs,dim=1)
            

                generation_count += 1
                #print(generation_count)
            


                writer.add_scalar('ave_fitness', ave_fitness, generation_count)

                if num_parents_for_mating == 'all':
                    parents = (improvements > 0).nonzero().squeeze(dim=1).cpu()
                else:
                    parents = torch.topk(improvements, k=num_parents_for_mating, largest=True).indices.tolist() 
                #print("Improvements: ", improvements)
                #print("Parents:", parents)

                if config['fitness_weights']:
                    #num_parents_for_mating also serves to indicate what type of mating to do
                    model.mate(parents, num_parents_for_mating=num_parents_for_mating, fitness_weights=improvement[parents])
                else:
                    model.mate(parents, num_parents_for_mating=num_parents_for_mating)


                if (time.time() > last_eval + eval_every_time) or (generation_count > config['max_generation']):
                    last_eval = time.time()
                    dt = time.time() - t0

                    model.reset()
                    loss, accuracy = evaluate(model, val_loader)

                    writer.add_scalar('best_val_loss', loss, generation_count)
                    writer.add_scalar('best_val_acc', accuracy, generation_count)
                    writer.add_scalar('best_val_acc_time', accuracy, time.time() - start)
                    print(f'gen {generation_count} | loss: {loss:.4f} | accuracy: {accuracy:.2%} | elapsed time {time.time() - start:.1f} | seconds per generation: {dt/(generation_count - last_eval_generation):.3f}')
                    last_eval_generation = generation_count

                    if time.time() > last_save + save_every_time:
                        save_model(model.cpu(), config, experiment_name, verbose=True)
                        last_save = time.time()
                        model.to(device)

                    if generation_count > config['max_generation'] or accuracy > config['target_acc']:
                        done=True
                        break
                    t0 = time.time()

                if generation_count % upload_every == 0:
                    model.reset()
                    save_model(model.cpu(), config, experiment_name, verbose=True)
                    model.to(device)
                    print("Uploading to aws...")
                    upload_to_aws(experiment_name)

                #p.step()
    upload_to_aws(experiment_name)


if __name__ == '__main__':

    if device == 'cuda':
        mp.set_start_method('spawn')

    #torch.set_num_threads(8)

    #tb = program.TensorBoard()
    #tb.configure(argv=[None, '--logdir', f'./tensorboards'])
    #url = tb.launch()
    for config in configs:
        main(config)
