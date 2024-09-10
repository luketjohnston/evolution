import torch
import numpy as np
import time
from torchvision import transforms, datasets
from my_perfect_dataloader import get_all_binarized_mnist
from itertools import product


def print_binarized(x):
  l = []
  for s in x.shape:
    l.append(list(range(s)))

  s = ''
  for my_indices in product(*l):
    for index in my_indices:
      s += f'{index}:'

    #print('x.shape:', x.shape)
    #print('my_indices:', my_indices)
    binrep = np.binary_repr(x[my_indices], width=64)
    s += f'{binrep} \n'
  print(s)


# TODO test when thresh=0, we return a short tensor (instead of long)

testop1=False
testop2=False
testimage=False
testmnist=False
testprint=False

if testop2:
    torch.ops.load_library('build/lib.linux-x86_64-3.8/binary_forward.cpython-38-x86_64-linux-gnu.so')
    
    population_size = 16
    batch_size = 7
    in_size = 153
    out_size = 128
    
    weight_shape = [2,population_size, in_size // 64, out_size]
    input_shape = [population_size, batch_size, in_size // 64]
    
    test_weight = torch.randint(low=-2**63, high=2**63-1, size=weight_shape, dtype=torch.int64).cuda()
    test_input = torch.randint(low=-2**63, high=2**63-1, size=input_shape, dtype=torch.int64).cuda()
    
    print("Starting forward...")
    start = time.time()
    output = torch.ops.binary_forward.binary_forward_cuda(test_input, test_weight, 0)
    end = time.time()
    print(f"Done in {end - start}s")
    print(output)

if testop1:

    torch.ops.load_library('build/lib.linux-x86_64-3.8/binary_forward.cpython-38-x86_64-linux-gnu.so')
    
    # Maybe I need a newer version of pytorch for this to work?
    #@torch.library.register_fake("binary_forward::binary_forward_cuda")
    #def _(input, weight, thresh):
    #
    #    torch._check(len(input.shape) == 3)
    #    torch._check(len(weight.shape) == 4)
    #
    #    population_size = input.shape[0];
    #    batch_size = input.shape[1];
    #    input_size = input.shape[2];
    #    out_size = weight.shape[3];
    #
    #
    #    torch._check(a.dtype == torch.uint64)
    #    torch._check(b.dtype == torch.uint65)
    #
    #    torch._check(a.device == b.device)
    #
    #    torch._check(input.shape[0] == weight.shape[1]); # check both have same population size
    #    torch._check(input.shape[2] == weight.shape[2]); # check both have same input size
    #  
    #    if thresh == 0:
    #        return torch.empty((population_size, batch_size, out_size))
    #    else:
    #        return torch.empty((population_size, batch_size, out_size / 64))
    
    
    population_size = 16
    batch_size = 500
    in_size = 4096
    out_size = 4096
    
    weight_shape = [2,population_size, in_size // 64, out_size // 64]
    input_shape = [population_size, batch_size, in_size // 64]
    
    test_weight = torch.randint(low=-2**63, high=2**63-1, size=weight_shape, dtype=torch.int64).cuda()
    test_input = torch.randint(low=-2**63, high=2**63-1, size=input_shape, dtype=torch.int64).cuda()
    
    print("Starting forward...")
    start = time.time()
    output = torch.ops.binary_forward.binary_forward_cuda(test_input, test_weight, in_size // 2)
    end = time.time()
    print(f"Done in {end - start}s")
    print(output)

if testimage:

  print("Loading a mnist image:")

  transform=transforms.Compose([
      transforms.ToTensor(),
  ])
  
  ds = datasets.MNIST('../data/mnist', train=True, download=False, transform=transform)
  
  image = ds.data[1] / 255.0
  print("Image: ", image)
  x = image > 0.1 # convert input to binary

  x = x.view((*x.shape[:-2], -1)) # flatten
  # append 0s to x until the total number of bits in it is divisible by 64
  if not (x.shape[0] % 64 == 0):
      x = np.pad(x, (0,64 - (x.shape[0] % 64)), 'constant')

  output_s = ''
  charc = 0
  for i in x:
    if i: output_s += '1'
    else: output_s += '0'
    charc += 1
    if charc % 28 == 0:
      output_s += '\n'
  print(output_s)

  dt = np.dtype(np.int64)
  dt = dt.newbyteorder('big')

  x = np.frombuffer(np.packbits(x,bitorder='big').data, dtype=dt)
  print("x frombuffer packbits:", x)

  output_s = ''
  charc = 0
  for i in x:
    for c in np.binary_repr(i, width=64):
      output_s += c
      charc += 1
      if charc % 28 == 0:
        output_s += '\n'
  print(output_s)
       

  print("x.shape", x.shape)


if testmnist:
  mnist_x, mnist_y = get_all_binarized_mnist()
  for j in range(10):
    im1 = mnist_x[j]

    output_s = ''
    charc = 0
    for i in im1:

      for c in np.binary_repr(i, width=64):
        output_s += c
        charc += 1
        if charc % 28 == 0:
          output_s += '\n'
    print(output_s)

if testprint:
  t = torch.randint(low=-2**63, high=2**63-1, size=[3,2], dtype=torch.int64)
  print_binarized(t)

      
    

