#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"
#include "debug.h"
#include <time.h> // for srand init
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "binary.cuh"




namespace binary_forward {

__host__ __device__ void printbinary(device_inttype i) {
    printf(BBP BBP BBP BBP "\n", BB(i>>24), BB(i>>16), BB(i>>8), BB(i));
}

__host__ __device__ void printbinary(char* input, device_inttype i) {
    printf("%s" BBP BBP BBP BBP "\n", input, BB(i>>24), BB(i>>16), BB(i>>8), BB(i));
}

__host__ __device__ void printbinary(unsigned char i) {
    printf(BBP "\n", BB(i));
}


/**
In order to use all 32 warps at the final ballot_sync, we need the output
to be a multiple of 32.
This means that each of the 32 warps must compute popcount(w * input)
*/



// note that i is the index into the input as an array of integers, NOT as an array of bits.
__host__ __device__ intType_t i_ind(const intType_t p, const intType_t b, const intType_t i, const device_inttype batch_size, const device_inttype in_ints) {
 
  return p * batch_size * in_ints + b * in_ints + i;
}

// note that o is the index into output as an array of integers, NOT as an array of bits.
__host__ __device__ intType_t o_ind(const intType_t p, const intType_t b, const intType_t o, const device_inttype batch_size, const device_inttype out_ints) {
  return p * batch_size * out_ints + b * out_ints + o;
}

// note the "o" arg to w_ind should be different than the "o" arg to o_ind above.
// (because w has a full integer entry for each output bit o)
__host__ __device__ intType_t w_ind(const intType_t p, const intType_t i, const intType_t t,  const intType_t o, const device_inttype in_ints, const device_inttype out_size) {
  return p * in_ints * out_size * 2 + i * out_size * 2 + t * out_size + o;
}



// Computes the binary matrix multiplication on the host and saves the output
// out should be zeroed
void host_op(
    device_inttype const * const input, 
    device_inttype const * const weight, 
    output_inttype * out, 
    int thresh,
    const device_inttype population_size,
    const device_inttype batch_size,
    const device_inttype in_ints,
    const device_inttype out_size,
    bool verbose
    ) {
  const device_inttype out_int32s = out_size / OUTPUT_INTTYPE_BITS;

  //if (thresh == 0) {
  //  out = (device_inttype*) out;
  //} else {
  //  out = (output_inttype*) out;
  //}

  for( intType_t b = 0; b < batch_size; b++) {
    for( intType_t p = 0; p < population_size; p++) {
      for( intType_t o = 0; o < out_size; o++) {

        output_inttype temp = 0;

        for( intType_t i = 0; i < in_ints; i++) {
          temp += __builtin_popcount( (input[i_ind(p,b,i,batch_size,in_ints)] & weight[w_ind(p,i,0,o,in_ints,out_size)]));
          temp += __builtin_popcount((~input[i_ind(p,b,i,batch_size,in_ints)]) & weight[w_ind(p,i,1,o,in_ints,out_size)]);
        }
        // NOTE that the way we shift here, the outputs are ordered low-precision bits first
       
        if (thresh > 0) {
          device_inttype bit_to_set = (temp >= thresh) << (o % (OUTPUT_INTTYPE_BITS));
          device_inttype o_int_index = o / (OUTPUT_INTTYPE_BITS);
          //if (verbose && b == 1) {printf("Setting output o: %u o_int_index: %u to bit_to_set: %u\n", o, o_int_index, bit_to_set); };
          out[o_ind(p,b,o_int_index,batch_size,out_int32s)] = out[o_ind(p,b,o_int_index,batch_size,out_int32s)] | bit_to_set;
        } else {
          //if (verbose) {printf("Setting output b: %u o: %u temp: %d\n", b, o, temp); };
          // need to multiple o by 2 since we will eventually convert this back to (int64_t *) 
          // since that is what pytorch uses
          // (meaning, we only want to set every other 32-bit entry)
          out[o_ind(p,b,2*o,batch_size,out_size*2)] = temp;
        }
      }
    }
  }
}

at::Tensor host_helper(at::Tensor input, at::Tensor weight, int thresh, bool verbose) {
  const unsigned int population_size = input.sizes()[0];
  const unsigned int batch_size = input.sizes()[1];
  const unsigned int in_ints = input.sizes()[2] * 2; // * 2 because input is int64_t tensor, we want number of int32s
  const unsigned int out_size = weight.sizes()[3];
  TORCH_CHECK(input.sizes().size() == 3); // population, batch, input
  TORCH_CHECK(weight.sizes().size() == 4); // population, input, 2, output
  // TODO we shouldn't need this check, need to make kernel work for any batch size etc.
  //TORCH_CHECK(input.sizes()[1] >= 32); 
  TORCH_CHECK(input.sizes()[0] == weight.sizes()[0]); // check both have same population size
  TORCH_CHECK(input.sizes()[2] == weight.sizes()[1]); // check both have same input size

  TORCH_CHECK(input.dtype() == torch_device_inttype); 
  TORCH_CHECK(weight.dtype() == torch_device_inttype);

  at::Tensor input_contig = input.contiguous();
  at::Tensor weight_contig = weight.contiguous();
  const device_inttype* input_ptr = (device_inttype *) input_contig.data_ptr();
  const device_inttype* weight_ptr = (device_inttype *) weight_contig.data_ptr();
  at::Tensor output;
  auto options = input_contig.options().dtype(torch_device_inttype);
  const unsigned int out_int64s  = (out_size + 63) / 64;
  if (thresh == 0) {
    output = torch::zeros({population_size, batch_size, out_size}, options);
  } else {
    output = torch::zeros({population_size, batch_size, out_int64s}, options);
  }
  output_inttype* output_ptr = (output_inttype *) output.data_ptr();
  host_op(input_ptr, weight_ptr, output_ptr, thresh, population_size, batch_size, in_ints, out_size, verbose);
  return output;
}

/**
o n w o1 on
0 1 0 0 0
0 1 1 0 1
1 0 0 0 0
1 0 1 1 0
Don't see a way to do this in one op, need both the "not" and the "and".
Later could try replacing both with xor or something like that
*/

/**
Note that the input size is always a multiple of DEVICE_INTTYPE_BITS. If at a higher level we want to use an 
input size that isn't a multiple of DEVICE_INTTYPE_BITS, we must make it into a multiple of DEVICE_INTTYPE_BITS by extending
it with constants.

batch_size, on the other hand, does not need to be a multiple of the block size. So
we need to wrap all functionality within if(batch_id < batch_size) checks.
*/
__global__ void binary_forward(
        device_inttype const * const input, 
        device_inttype const * const weight, 
        int64_t * out, 
        const unsigned int out_tile_y_ints, 
        const unsigned int in_ints, 
        const unsigned int warp_size,
        const unsigned int population_size,
        const unsigned int batch_size,
        const unsigned int out_size,
        const bool verbose
        ) {

  extern __shared__ device_inttype shared[];

  device_inttype* weight_tile{&shared[0]};
  device_inttype* not_weight_tile{&shared[warp_size * warp_size]};
  device_inttype* input_tile{&shared[2 * warp_size * warp_size]};

  // TODO distinguish between warp_size and threadblock.y.size
  const intType_t output_tile_x = blockIdx.x * warp_size;
  const intType_t output_tile_y = blockIdx.y * out_tile_y_ints;
  const intType_t b = output_tile_y + threadIdx.y;

  const intType_t p = blockIdx.z;

  device_inttype acc = 0;

  for (unsigned int tile_i = 0; tile_i < (in_ints + warp_size - 1) / warp_size; ++tile_i) {

    __syncthreads();
    intType_t i = tile_i * warp_size + threadIdx.x;

    // Load input into shared memory
    //if (threadIdx.x < warp_size) { 
    if (threadIdx.x < warp_size && b < batch_size && i < in_ints) { 
      input_tile[threadIdx.y * warp_size + threadIdx.x] = input[i_ind(p,b, i, batch_size, in_ints)]; 
    }
    //if (true) {
    // load weight into shared memory
    for (int weight_y_offset = 0; weight_y_offset < warp_size; weight_y_offset += blockDim.y) {

      intType_t w_i_to_load =  tile_i * warp_size + threadIdx.y + weight_y_offset;
      intType_t w_o_to_load = blockIdx.x * warp_size + threadIdx.x;

      if (w_i_to_load < in_ints) {
        // "w_o_to_load" varies with threadIdx.x and is the last coord so access is coalesced
        weight_tile[    (threadIdx.y + weight_y_offset) * warp_size +  threadIdx.x] = weight[w_ind( p, w_i_to_load,0, w_o_to_load,  in_ints, out_size)]; 
        not_weight_tile[(threadIdx.y + weight_y_offset) * warp_size + threadIdx.x] = weight[w_ind( p, w_i_to_load,1, w_o_to_load,  in_ints, out_size)]; 
      }
    }

    __syncthreads(); // wait until shared memory load is finished before continuing with compute

    for (unsigned int j = 0; j < warp_size; ++j) {

      //if (true) {
      //if (verbose && b < 2 && j < 2) {printf("Checking b %u < batch_size %u and j %u + tile_i %u * 32 < in_ints %u\n",b,batch_size,j,tile_i,in_ints);};
        
      // Note that we will be computing this section for some invalid outputs,
      // but we jsut don't write them into out[] in the final if statement 
      if (b < batch_size && (j + tile_i * warp_size < in_ints)) {
        //device_inttype tmp = acc + 0;

        //device_inttype i1 = input_tile[threadIdx.y * warp_size + j];
        //// Note that the compiler optimizes away the ~i operation, it pre-computes a lookup table and uses a lop3.lut op
        //// (binary 3-register operation, look-up-table)
        //device_inttype not_i1 = ~i1;
        //device_inttype w1 = weight_tile[j * warp_size + threadIdx.x];
        //device_inttype not_w1 = not_weight_tile[j * warp_size + threadIdx.x];
        //device_inttype and1 = i1 & w1;
        //device_inttype and2 = not_i1 & not_w1;
        //acc += __popc(and1);
        //acc += __popc(and2);


        acc += __popc(input_tile[threadIdx.y * warp_size + j] & weight_tile[j * warp_size + threadIdx.x]);
        //device_inttype d1 = acc - tmp;
        acc += __popc((~input_tile[threadIdx.y * warp_size + j]) & not_weight_tile[j * warp_size + threadIdx.x]);


        //device_inttype d2 = acc - tmp - d1;
        //if (verbose) {printf("tmp: %ld, acc: %ld, d1: %ld, d2: %ld, x: %u, y: %u, b: %u\n", tmp, acc, d1, d2, threadIdx.x, threadIdx.y, b);};
        //if (verbose) {printf("b: %u, x:%u y:%u j:%u Added d1: %ld and d2: %ld to acc to get acc %ld\n", b, threadIdx.x, threadIdx.y, j, d1, d2, acc);};
        //if (verbose) {printf("b: %u, x: %u, y: %u, j: %u, ~input_tile[threadIdx.y * warp_size + j]: %ld, not_weight_tile[j*warp_size + threadIdx.x]: %ld\n", b, threadIdx.x, threadIdx.y, j, ~input_tile[threadIdx.y * warp_size + j], not_weight_tile[j * warp_size + threadIdx.x]);};
        //if (verbose) {
        //  std::cout << "b: " << b << "x:" << threadIdx.x << "y: " << threadIdx.y << "j: " << "d1: " << d1 << "d2: " << d2 << "acc: " << acc << std::endl;
        //}

          //printf("acc: %d, b: %u, x:%u y:%u j:%u Added d1: %d and d2: %d to acc to get acc %d\n", acc, b, threadIdx.x, threadIdx.y, j, d1, d2, acc);};
        //if (verbose) {printf("ACC:%d\n", acc);};
      }
    }
    
    __syncthreads(); // wait until this compute is finished before next loop replaces shared memory

  }
  //if (true) { 
  if (b < batch_size && output_tile_x + threadIdx.x < out_size) { 
    out[o_ind(p,b,output_tile_x + threadIdx.x,batch_size,out_size)] = acc;
  }
}

__global__ void binary_forward_with_threshold(
        device_inttype const * const input, 
        device_inttype const * const weight, 
        output_inttype * out, 
        const unsigned int thresh, 
        const unsigned int out_tile_y_ints, 
        const unsigned int in_ints, 
        const unsigned int warp_size,
        const unsigned int population_size,
        const unsigned int batch_size,
        const unsigned int out_size,
        const bool verbose
        ) {

  extern __shared__ device_inttype shared[];

  device_inttype* weight_tile{&shared[0]};
  // TODO replace 32 with a constant or something
  device_inttype* not_weight_tile{&shared[warp_size * warp_size]};
  device_inttype* input_tile{&shared[2 * warp_size * warp_size]};

  const intType_t output_tile_x = blockIdx.x;
  const intType_t output_tile_y = blockIdx.y * out_tile_y_ints;
  const intType_t b = output_tile_y + threadIdx.y;

  const unsigned int out_int32s = out_size / OUTPUT_INTTYPE_BITS;

  const intType_t p = blockIdx.z;

  unsigned int acc = 0;

  
  for (unsigned int tile_i = 0; tile_i < (in_ints + warp_size - 1) / warp_size; ++tile_i) {

    intType_t i = tile_i * warp_size + threadIdx.x;

    // Load input into shared memory
    //if (threadIdx.x < warp_size) { 
    if (threadIdx.x < warp_size &&  b < batch_size && i < in_ints) { 
      //if (verbose && threadIdx.x < 3 && threadIdx.y < 1) {printf("threadIdx.x %u threadIdx.y %u setting input\n", threadIdx.x, threadIdx.y);};
      input_tile[threadIdx.y * warp_size + threadIdx.x] = input[i_ind(p, b, i, batch_size, in_ints)]; 
    }

    //if (true) {
    // load weight into shared memory
    for (int weight_y_offset = 0; weight_y_offset < warp_size; weight_y_offset += blockDim.y) {

      intType_t w_i_to_load =  tile_i * warp_size + threadIdx.y + weight_y_offset;
      intType_t w_o_to_load = blockIdx.x * warp_size + threadIdx.x;
      intType_t weight_tile_i = (threadIdx.y + weight_y_offset) * warp_size + threadIdx.x;

      //printf("x:%u, y:%u, weight_tile_i: %u, wyo: %d, blockDim.y: %u, w_i_to_load: %u,  w_o_to_load: %u\n", threadIdx.x, threadIdx.y, weight_tile_i, weight_y_offset, blockDim.y, w_i_to_load, w_o_to_load);
      //if (verbose && blockIdx.x == 1 && threadIdx.x < 3 && threadIdx.y < 1) {printf("threadIdx.x %u threadIdx.y %u w_i_to_load: %u in_ints: %u\n", threadIdx.x, threadIdx.y, w_i_to_load, in_ints);};
      if (w_i_to_load < in_ints) {
        // "o" varies with threadIdx.x and is the last coord so access is coalesced

        //if (verbose && blockIdx.x == 1 && threadIdx.x < 3 && threadIdx.y < 1) {printf("threadIdx.x %u threadIdx.y %u setting shared mem weights w_i_to_load %u w_o_to_load %u \n", threadIdx.x, threadIdx.y, w_i_to_load, w_o_to_load);};

        weight_tile[    (threadIdx.y + weight_y_offset) * warp_size +  threadIdx.x] = weight[w_ind( p, w_i_to_load,0, w_o_to_load,  in_ints, out_size)]; 
        not_weight_tile[(threadIdx.y + weight_y_offset) * warp_size + threadIdx.x] = weight[w_ind( p, w_i_to_load,1, w_o_to_load,  in_ints, out_size)]; 
      }
    }


    __syncthreads(); // wait until shared memory load is finished before continuing with compute

    for (unsigned int j = 0; j < warp_size; ++j) {

      //if (true) {
      
      //if (verbose && blockIdx.x == 1 && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u, j:%u,  j+tile_i*warp_size:%u \n" , threadIdx.x, threadIdx.y, j, j + tile_i * warp_size);};

      if (b < batch_size && (j + tile_i * warp_size < in_ints)) { // TODO double check?

        //device_inttype x1 = input_tile[threadIdx.y * warp_size + j];
        //if (verbose && blockIdx.x == 1 && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u j:%u: input_tile[...]: " BBP BBP BBP BBP "\n", threadIdx.x, threadIdx.y, j, BB(x1>>24), BB(x1>>16), BB(x1>>8), BB(x1));};

        //x1 = weight_tile[j * warp_size + threadIdx.x];
        //if (verbose && blockIdx.x == 1 && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u j:%u: weight_tile[...]: " BBP BBP BBP BBP "\n", threadIdx.x, threadIdx.y, j, BB(x1>>24), BB(x1>>16), BB(x1>>8), BB(x1));};
        //x1 = not_weight_tile[j * warp_size + threadIdx.x];
        //if (verbose && blockIdx.x == 1 && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u j:%u: not_weight_tile[...]: " BBP BBP BBP BBP "\n", threadIdx.x, threadIdx.y, j, BB(x1>>24), BB(x1>>16), BB(x1>>8), BB(x1));};

        //unsigned int tmp = acc;

        //device_inttype i1 = input_tile[threadIdx.y * warp_size + j];
        //device_inttype not_i1 = ~i1;
        //device_inttype w1 = weight_tile[j * warp_size + threadIdx.x];
        //device_inttype not_w1 = not_weight_tile[j * warp_size + threadIdx.x];
        //device_inttype and1 = i1 & w1;
        //device_inttype and2 = not_i1 & not_w1;
        //device_inttype popc1 = __popc(and1);
        //device_inttype popc2 = __popc(and2);
        //acc += popc1;
        //acc += popc2;

        acc += __popc(input_tile[threadIdx.y * warp_size + j] & weight_tile[j * warp_size + threadIdx.x]);
        acc += __popc((~input_tile[threadIdx.y * warp_size + j]) & not_weight_tile[j * warp_size + threadIdx.x]);

        //if (verbose && blockIdx.x == 1 && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u j:%u adding to acc: %u \n" , threadIdx.x, threadIdx.y, j, acc - tmp);};

      } 
    }
    
    __syncthreads(); // wait until this compute is finished before next loop replaces shared memory

  }


  //if (verbose && blockIdx.x == 1 && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u acc:%u:\n" , threadIdx.x, threadIdx.y, acc);};

  // activemask will always be 1111..., TODO 
  //unsigned int r = __ballot_sync(__activemask(), acc >= thresh);
  unsigned int r = __ballot_sync(-1, acc >= thresh);
  // after ballot_sync, only one thread per warp needs to write to output
  //if (threadIdx.x % warp_size == 0) {
  if (threadIdx.x % warp_size == 0 && b < batch_size) {
    out[o_ind(p,b,output_tile_x,batch_size,out_int32s)] = r;
  }
}


at::Tensor binary_forward_cuda(
        const at::Tensor& input, 
        const at::Tensor& weight, 
        const int64_t thresh,
        const bool verbose
        ) {


  const unsigned int population_size = input.sizes()[0];
  const unsigned int batch_size = input.sizes()[1];
  const unsigned int in_ints = input.sizes()[2] * 2; // * 2 because input is an int64 tensor, and we want to count the int32s
  const unsigned int out_size = weight.sizes()[3];
  
  TORCH_CHECK(input.sizes().size() == 3); // population, batch, input
  TORCH_CHECK(weight.sizes().size() == 4); // population, input, 2, output

  TORCH_CHECK(weight.sizes()[2] == 2); // population, input, 2, output

  TORCH_CHECK(input.sizes()[0] == weight.sizes()[0]); // check both have same population size
  TORCH_CHECK(input.sizes()[2] == weight.sizes()[1]); // check both have same input size

  TORCH_CHECK(input.dtype() == torch_device_inttype); 
  TORCH_CHECK(weight.dtype() == torch_device_inttype);

  TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(weight.device().type() == at::DeviceType::CUDA);

  at::Tensor input_contig = input.contiguous();
  at::Tensor weight_contig = weight.contiguous();

  const device_inttype* input_ptr = (device_inttype *) input_contig.data_ptr();
  const device_inttype* weight_ptr = (device_inttype *) weight_contig.data_ptr();

  const unsigned int out_tile_y_ints = 8; 
  const dim3 threads(32, out_tile_y_ints, 1); 

  const unsigned int warp_size = 32;
  const unsigned int sharedMemSize = DEVICE_INTTYPE_BYTES * (((warp_size * out_tile_y_ints) + 2 * (warp_size * warp_size))); 

  at::Tensor output;

  const unsigned int out_int64s  = (out_size + 63) / 64;
  const unsigned int out_int32s  = out_int64s * 2;
  // out_ints * 2 because out_ints is with respect to 64-bit precision, whereas the output at the kernel
  // level is 32-bit precision, see below comment as well
  const dim3 blocks(out_int32s, (batch_size + out_tile_y_ints - 1) / out_tile_y_ints, population_size);

  //printf("Checking thresh");
  if (thresh > 0) {

      // note that the tensor is created with 64-bit precisions here,
      // because pytorch expects integer tensors to have 64-bit precision,
      // but the data pointer will be cast to 32-bit precisions when sent to the kernel.
      // This is because outputs are generated one bit per warp and collected with __ballot_sync,
      // so we need the output to have 32-bit precision.
      auto options = input_contig.options().dtype(torch_device_inttype);

      output = torch::empty({population_size, batch_size, out_int64s}, options);
      output_inttype* output_ptr = (output_inttype *) output.data_ptr();

      //printf("Calling binary forward with threshold...");

      binary_forward_with_threshold<<<blocks, threads, sharedMemSize>>>(
              input_ptr, 
              weight_ptr, 
              output_ptr, 
              thresh,
              out_tile_y_ints,
              in_ints,
              warp_size,
              population_size,
              batch_size,
              out_size,
              verbose
              );
  } else {
      // here we have no thresholding, so the output tensor has to be larger
      // I think the output has to be torch::kInt64, TODO investigate
      auto options = input_contig.options().dtype(torch_device_inttype); 
      output = torch::zeros({population_size, batch_size, out_size}, options);
      int64_t* output_ptr = (int64_t*) output.data_ptr();

      //printf("Calling binary forward...");
      binary_forward<<<blocks, threads, sharedMemSize>>>(
              input_ptr, 
              weight_ptr, 
              output_ptr, 
              out_tile_y_ints,
              in_ints,
              warp_size,
              population_size,
              batch_size,
              out_size,
              verbose

              );
  }
      

  return output;
}

TORCH_LIBRARY(binary_forward, m) {
  m.def("binary_forward_cuda", &binary_forward_cuda);
}

}

