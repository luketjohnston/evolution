#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"
#include "debug.h"
#include <time.h> // for srand init
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define BBP "%c%c%c%c%c%c%c%c"
#define BB(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0') 

#define  torch_device_inttype torch::kInt64
#define  torch_output_inttype torch::kInt32

namespace binary_forward {

// we can do 64 bit operations the fastest so we want to primarily 
// work with those
// TODO update above comment, we are testing with 32 bit instead
typedef int32_t device_inttype;
const int DEVICE_INTTYPE_BYTES{sizeof(device_inttype)};
// However, there are only 32 warps that can collect bit results with __ballot_sync
// So we need the output to have 32-bit precision
typedef int32_t output_inttype;
const int OUTPUT_INTTYPE_BITS{sizeof(output_inttype) * 8};

const int OUT_TILE_X_MULTIPLICITY{4};
const int OUT_TILE_Y_MULTIPLICITY{8};




__host__ __device__ void printbinary(device_inttype i);
__host__ __device__ void printbinary(char* input, device_inttype i);
__host__ __device__ void printbinary(unsigned char i);

typedef unsigned int intType_t;


at::Tensor host_helper(at::Tensor input, at::Tensor weight, int thresh, bool verbose);

at::Tensor binary_forward_cuda(
        const at::Tensor& input, 
        const at::Tensor& weight, 
        const int64_t thresh,
        const bool verbose
        );

}

