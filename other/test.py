import torch
import time
import torch.library


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


population_size = 64
batch_size = 500
in_size = 4096
out_size = 4096

weight_shape = [2,population_size, in_size // 64, out_size // 64]
input_shape = [population_size, batch_size, in_size // 64]

test_weight = torch.randint(low=-2**63, high=2**63-1, size=weight_shape, dtype=torch.int64).cuda()
test_input = torch.randint(low=-2**63, high=2**63-1, size=input_shape, dtype=torch.int64).cuda()

print("Starting forward...")
start = time.time()
torch.ops.binary_forward.binary_forward_cuda(test_input, test_weight, in_size // 2)
end = time.time()
print(f"Done in {end - start}s")

