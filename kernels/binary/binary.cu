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
__host__ __device__ intType_t w_ind(const intType_t p, const intType_t i, const intType_t t,  const intType_t o, const device_inttype in_ints, const device_inttype out_size, const intType_t wdim2) {
  return p * in_ints * out_size * wdim2 + i * out_size * wdim2 + t * out_size + o;
}

at::Tensor host_consolidate(
    at::Tensor input) {
    

  const unsigned int population_size = input.sizes()[0];
  const unsigned int batch_size = input.sizes()[1];
  const unsigned int in_ints = input.sizes()[2]; 

  unsigned int out_total_int64s = (in_ints + 63) / 64;
  //printf("Out total ints: %u\n", out_total_int64);
  //printf("In ints: %u\n", in_ints);

  auto options = input.options().dtype(torch_device_inttype);

  at::Tensor output = torch::zeros({population_size, batch_size, out_total_int64s}, options);

  for( int64_t p = 0; p < population_size; p++) {
    for( int64_t b = 0; b < batch_size; b++) {
      for( int64_t i = 0; i < in_ints; i++) {

        int64_t temp = input.index({p,b,i}).item<int64_t>();

        int64_t bit_to_set = ((int64_t) (temp >= 0)) << (i % (64));
        int64_t o_int_index = i / 64;

        
        //printf("bit_to_set: %ld o_int_index: %ld \n", bit_to_set, o_int_index);
        //printf("output.index[p,b,o_int_index] before: %ld\n", output.index({p,b,o_int_index}).item<int64_t>());
        output.index({p,b,o_int_index}) =  output.index({p,b,o_int_index}).item<int64_t>() | bit_to_set;
        //printf("output.index[p,b,o_int_index] after: %ld\n", output.index({p,b,o_int_index}).item<int64_t>());
      }
    }
  }
  return output;
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
          temp += __builtin_popcountll( (input[i_ind(p,b,i,batch_size,in_ints)] & weight[w_ind(p,i,0,o,in_ints,out_size,2)]));
          temp += __builtin_popcountll((~input[i_ind(p,b,i,batch_size,in_ints)]) & weight[w_ind(p,i,1,o,in_ints,out_size,2)]);
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
  const unsigned int in_ints = input.sizes()[2]; // * 2 because input is int64_t tensor, we want number of int32s
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


__global__ void consolidate_bits(
      const int64_t * const input, 
      output_inttype * out,
      const unsigned int in_ints,
      const unsigned int batch_size,
      const unsigned int out_total_int32s) {

  const intType_t p = blockIdx.z;
  const intType_t b = blockIdx.y * 32 + threadIdx.y;
  const intType_t o_int = blockIdx.x;
  const intType_t i = blockIdx.x * WARP_SIZE + threadIdx.x;

  const intType_t input_index = p * (batch_size * in_ints) + b * (in_ints) + i;

  if (b < batch_size) {
    // TODO maybe can remove the >= 0 below if we change input; ballot_sync reduces 1 iff non-zero
    unsigned int r = __ballot_sync(-1, input[input_index] >= 0); // TODO fix -1 type conversion warning

    // after ballot_sync, only one thread per warp needs to write to output
    if (threadIdx.x % WARP_SIZE == 0 && b < batch_size) {
      out[o_ind(p,b,o_int,batch_size,out_total_int32s)] = r;
    }
  }
}



/**
Note that the input size is always a multiple of DEVICE_INTTYPE_BITS. If at a higher level we want to use an 
input size that isn't a multiple of DEVICE_INTTYPE_BITS, we must make it into a multiple of DEVICE_INTTYPE_BITS by extending
it with constants.


batch_size, on the other hand, does not need to be a multiple of the block size. So
we need to wrap all functionality within if(batch_id < batch_size) checks.
*/
__global__ void binary_forward_AND(
        device_inttype const * const input, 
        device_inttype const * const weight, 
        void * out, 
        int64_t thresh,
        const unsigned int out_tile_y_size, 
        const unsigned int out_tile_x_ints, 
        const unsigned int in_ints, 
        const unsigned int population_size,
        const unsigned int batch_size,
        const unsigned int out_size,
        const bool verbose
        ) {

  extern __shared__ device_inttype shared[];

  device_inttype* weight_tile{&shared[0]};
  device_inttype* not_weight_tile{&shared[blockDim.x * out_tile_x_ints * blockDim.y]};
  device_inttype* input_tile{&shared[2 * blockDim.x * out_tile_x_ints * blockDim.y]};

  // TOOD probably a lot of the warp_size here should be replaced with blockDim.x ( which 
  // should be a multiple of warp size)

  intType_t output_tile_x; // TODO how to make this const and init later?
  const intType_t output_tile_y = blockIdx.y * out_tile_y_size;

  const unsigned int out_valid_int32s = out_size / OUTPUT_INTTYPE_BITS;
  const unsigned int out_total_int32s = ((out_valid_int32s + 1) / 2) * 2;

  if (thresh == 0) { // do not threshold, just output activations.
    output_tile_x = blockIdx.x * OUT_TILE_X_MULTIPLICITY * blockDim.x;
  } else { // threshold activations and consolidate bits.
    output_tile_x = blockIdx.x * OUT_TILE_X_MULTIPLICITY;
  }

  const intType_t p = blockIdx.z;

  int64_t acc[OUT_TILE_X_MULTIPLICITY][OUT_TILE_Y_MULTIPLICITY] = { 0 }; // double check this initializes to 0

  for (unsigned int tile_offset = 0; tile_offset < in_ints; tile_offset += blockDim.x) {

    // Load input into shared memory
    for (int input_y_offset = 0; input_y_offset < out_tile_y_size; input_y_offset += blockDim.y) {

      intType_t b = output_tile_y + threadIdx.y + input_y_offset;
      intType_t i = tile_offset + threadIdx.x; // CORRECT

      if (b < batch_size && i < in_ints) { 
        //if (verbose && threadIdx.x < 3 && threadIdx.y < 1) {printf("threadIdx.x %u threadIdx.y %u setting input\n", threadIdx.x, threadIdx.y);};
        input_tile[(threadIdx.y + input_y_offset) * blockDim.x + threadIdx.x] = input[i_ind(p, b, i, batch_size, in_ints)]; 
      }
    }

    // load weight into shared memory
    // TODO probably remove this first for loop, it never loops since blockDim.y = warp_size always?
    for (int weight_x_offset = 0; weight_x_offset < OUT_TILE_X_MULTIPLICITY * blockDim.x; weight_x_offset += blockDim.x) {

      intType_t w_i_to_load =  tile_offset + threadIdx.y;
      intType_t w_o_to_load = blockIdx.x * blockDim.x * OUT_TILE_X_MULTIPLICITY + threadIdx.x + weight_x_offset;

      //if (verbose && blockIdx.x == 1 && threadIdx.x < 3 && threadIdx.y < 1) {printf("threadIdx.x %u threadIdx.y %u w_i_to_load: %u in_ints: %u\n", threadIdx.x, threadIdx.y, w_i_to_load, in_ints);};
      if (w_i_to_load < in_ints && w_o_to_load < out_size) {
        // "o" varies with threadIdx.x and is the last coord so access is coalesced
        weight_tile[threadIdx.y * blockDim.x * OUT_TILE_X_MULTIPLICITY +  weight_x_offset + threadIdx.x] = weight[w_ind( p, w_i_to_load,0, w_o_to_load,  in_ints, out_size,2)]; 
        not_weight_tile[threadIdx.y * blockDim.x * OUT_TILE_X_MULTIPLICITY + weight_x_offset + threadIdx.x] = weight[w_ind( p, w_i_to_load,1, w_o_to_load,  in_ints, out_size,2)]; 
      }
    }

    __syncthreads(); // wait until shared memory load is finished before continuing with compute

    for (unsigned int j = 0; j < blockDim.x; ++j) {

      // read shared memory into registers
      device_inttype input_regs[OUT_TILE_Y_MULTIPLICITY];
      device_inttype weight_regs[OUT_TILE_X_MULTIPLICITY];
      device_inttype not_weight_regs[OUT_TILE_X_MULTIPLICITY];
      for (unsigned int yi = 0; yi < OUT_TILE_Y_MULTIPLICITY; yi += 1) {
        input_regs[yi] = input_tile[yi * blockDim.x * blockDim.y + threadIdx.y * blockDim.x  + j];
      }
      for (unsigned int xi = 0; xi < OUT_TILE_X_MULTIPLICITY; xi += 1) {
        weight_regs[xi] = weight_tile[j * (blockDim.x * OUT_TILE_X_MULTIPLICITY) + xi * blockDim.x + threadIdx.x];
        not_weight_regs[xi] = not_weight_tile[j * (blockDim.x * OUT_TILE_X_MULTIPLICITY) + xi * blockDim.x + threadIdx.x];
      }
        
      // Do compute
      for (unsigned int xi = 0; xi < OUT_TILE_X_MULTIPLICITY; xi += 1) {
        for (unsigned int yi = 0; yi < OUT_TILE_Y_MULTIPLICITY; yi += 1) {

          intType_t b = output_tile_y + yi * blockDim.y + threadIdx.y;
          intType_t o = output_tile_x + xi * blockDim.x + threadIdx.x;

          if (b < batch_size && (j + tile_offset < in_ints) && o < out_size) { // TODO double check?
            acc[xi][yi] += __popcll((input_regs[yi] & weight_regs[xi]) | (~input_regs[yi] & not_weight_regs[xi]));
          } 
        }
      }
    }
    __syncthreads(); // wait until this compute is finished before next loop replaces shared memory
  }

  if (thresh == 0) {
    for (unsigned int xi = 0; xi < OUT_TILE_X_MULTIPLICITY; xi += 1) {
      for (unsigned int yi = 0; yi < OUT_TILE_Y_MULTIPLICITY; yi += 1) {

        intType_t b = output_tile_y + yi * blockDim.y + threadIdx.y;
        intType_t o = output_tile_x + xi * blockDim.x + threadIdx.x;

        if (b < batch_size && o < out_size) { 
          ((int64_t * ) out)[o_ind(p,b,o,batch_size,out_size)] = acc[xi][yi];
        }
      }
    }
  } else {
    for (unsigned int xi = 0; xi < OUT_TILE_X_MULTIPLICITY; xi += 1) {
      for (unsigned int yi = 0; yi < OUT_TILE_Y_MULTIPLICITY; yi += 1) {

        intType_t b = output_tile_y + yi * blockDim.y + threadIdx.y;
        intType_t o_int = output_tile_x + xi;

        //unsigned int r = __ballot_sync(__activemask(), acc >= thresh);
        unsigned int r = __ballot_sync(-1, acc[xi][yi] >= thresh); // TODO fix -1 type conversion warning
        // after ballot_sync, only one thread per warp needs to write to output
        if (threadIdx.x % blockDim.x == 0 && b < batch_size && o_int < out_valid_int32s) {
          ((output_inttype * ) out)[o_ind(p,b,o_int,batch_size,out_total_int32s)] = r;
        }
      }
    }
  }
}

__global__ void binary_forward_XOR(
        device_inttype const * const input, 
        device_inttype const * const weight, 
        void * out, 
        int64_t thresh,
        const unsigned int out_tile_y_size, 
        const unsigned int out_tile_x_ints, 
        const unsigned int in_ints, 
        const unsigned int population_size,
        const unsigned int batch_size,
        const unsigned int out_size,
        const bool verbose
        ) {

  extern __shared__ device_inttype shared[];

  device_inttype* weight_tile{&shared[0]};
  device_inttype* input_tile{&shared[blockDim.x * out_tile_x_ints * blockDim.y]};

  // TOOD probably a lot of the warp_size here should be replaced with blockDim.x ( which 
  // should be a multiple of warp size)

  intType_t output_tile_x; // TODO how to make this const and init later?
  const intType_t output_tile_y = blockIdx.y * out_tile_y_size;

  const unsigned int out_valid_int32s = out_size / OUTPUT_INTTYPE_BITS;
  const unsigned int out_total_int32s = ((out_valid_int32s + 1) / 2) * 2;

  if (thresh == 0) { // do not threshold, just output activations.
    output_tile_x = blockIdx.x * OUT_TILE_X_MULTIPLICITY * blockDim.x;
  } else { // threshold activations and consolidate bits.
    output_tile_x = blockIdx.x * OUT_TILE_X_MULTIPLICITY;
  }

  const intType_t p = blockIdx.z;

  int64_t acc[OUT_TILE_X_MULTIPLICITY][OUT_TILE_Y_MULTIPLICITY] = { 0 }; // double check this initializes to 0

  for (unsigned int tile_offset = 0; tile_offset < in_ints; tile_offset += blockDim.x) {

    // Load input into shared memory
    for (int input_y_offset = 0; input_y_offset < out_tile_y_size; input_y_offset += blockDim.y) {

      intType_t b = output_tile_y + threadIdx.y + input_y_offset;
      intType_t i = tile_offset + threadIdx.x; // CORRECT

      if (b < batch_size && i < in_ints) { 
        //if (verbose && threadIdx.x < 3 && threadIdx.y < 1) {printf("threadIdx.x %u threadIdx.y %u setting input\n", threadIdx.x, threadIdx.y);};
        input_tile[(threadIdx.y + input_y_offset) * blockDim.x + threadIdx.x] = input[i_ind(p, b, i, batch_size, in_ints)]; 
      }
    }

    // load weight into shared memory
    // TODO probably remove this first for loop, it never loops since blockDim.y = warp_size always?
    for (int weight_x_offset = 0; weight_x_offset < OUT_TILE_X_MULTIPLICITY * blockDim.x; weight_x_offset += blockDim.x) {

      intType_t w_i_to_load =  tile_offset + threadIdx.y;
      intType_t w_o_to_load = blockIdx.x * blockDim.x * OUT_TILE_X_MULTIPLICITY + threadIdx.x + weight_x_offset;

      //if (verbose && blockIdx.x == 1 && threadIdx.x < 3 && threadIdx.y < 1) {printf("threadIdx.x %u threadIdx.y %u w_i_to_load: %u in_ints: %u\n", threadIdx.x, threadIdx.y, w_i_to_load, in_ints);};
      if (w_i_to_load < in_ints && w_o_to_load < out_size) {
        // "o" varies with threadIdx.x and is the last coord so access is coalesced
        weight_tile[threadIdx.y * blockDim.x * OUT_TILE_X_MULTIPLICITY +  weight_x_offset + threadIdx.x] = weight[w_ind( p, w_i_to_load,0, w_o_to_load,  in_ints, out_size,1)]; 
      }
    }

    __syncthreads(); // wait until shared memory load is finished before continuing with compute

    for (unsigned int j = 0; j < blockDim.x; ++j) {

      // read shared memory into registers
      device_inttype input_regs[OUT_TILE_Y_MULTIPLICITY];
      device_inttype weight_regs[OUT_TILE_X_MULTIPLICITY];
      for (unsigned int yi = 0; yi < OUT_TILE_Y_MULTIPLICITY; yi += 1) {
        input_regs[yi] = input_tile[yi * blockDim.x * blockDim.y + threadIdx.y * blockDim.x  + j];
      }
      for (unsigned int xi = 0; xi < OUT_TILE_X_MULTIPLICITY; xi += 1) {
        weight_regs[xi] = weight_tile[j * (blockDim.x * OUT_TILE_X_MULTIPLICITY) + xi * blockDim.x + threadIdx.x];
      }
        
      // Do compute
      for (unsigned int xi = 0; xi < OUT_TILE_X_MULTIPLICITY; xi += 1) {
        for (unsigned int yi = 0; yi < OUT_TILE_Y_MULTIPLICITY; yi += 1) {

          intType_t b = output_tile_y + yi * blockDim.y + threadIdx.y;
          intType_t o = output_tile_x + xi * blockDim.x + threadIdx.x;

          if (b < batch_size && (j + tile_offset < in_ints) && o < out_size) { // TODO double check?
            acc[xi][yi] += __popcll((input_regs[yi] ^ weight_regs[xi]));
          } 
        }
      }
    }
    __syncthreads(); // wait until this compute is finished before next loop replaces shared memory
  }

  if (thresh == 0) {
    for (unsigned int xi = 0; xi < OUT_TILE_X_MULTIPLICITY; xi += 1) {
      for (unsigned int yi = 0; yi < OUT_TILE_Y_MULTIPLICITY; yi += 1) {

        intType_t b = output_tile_y + yi * blockDim.y + threadIdx.y;
        intType_t o = output_tile_x + xi * blockDim.x + threadIdx.x;

        if (b < batch_size && o < out_size) { 
          ((int64_t * ) out)[o_ind(p,b,o,batch_size,out_size)] = acc[xi][yi];
        }
      }
    }
  } else {
    for (unsigned int xi = 0; xi < OUT_TILE_X_MULTIPLICITY; xi += 1) {
      for (unsigned int yi = 0; yi < OUT_TILE_Y_MULTIPLICITY; yi += 1) {

        intType_t b = output_tile_y + yi * blockDim.y + threadIdx.y;
        intType_t o_int = output_tile_x + xi;

        //unsigned int r = __ballot_sync(__activemask(), acc >= thresh);
        unsigned int r = __ballot_sync(-1, acc[xi][yi] >= thresh); // TODO fix -1 type conversion warning
        // after ballot_sync, only one thread per warp needs to write to output
        if (threadIdx.x % blockDim.x == 0 && b < batch_size && o_int < out_valid_int32s) {
          ((output_inttype * ) out)[o_ind(p,b,o_int,batch_size,out_total_int32s)] = r;
        }
      }
    }
  }
}


at::Tensor consolidate_bits_cuda(
        const at::Tensor& input
        ) {
  const unsigned int population_size = input.sizes()[0];
  const unsigned int batch_size = input.sizes()[1];
  const unsigned int in_ints = input.sizes()[2]; // * 2 because input is an int64 tensor, and we want to count the int32s
  const unsigned int out_valid_int32s = (in_ints + 31) / 32;
  const unsigned int out_total_int64s = (out_valid_int32s + 1) / 2;
  const unsigned int out_total_int32s = out_total_int64s * 2;

  TORCH_CHECK(input.dtype() == torch_device_inttype); 
  TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);

  at::Tensor input_contig = input.contiguous();
  const int64_t* input_ptr = (int64_t *) input_contig.data_ptr();

  auto options = input_contig.options();
  at::Tensor output = torch::zeros({population_size, batch_size, out_total_int64s}, options);

  const dim3 blocks(out_valid_int32s, (batch_size + 31) / 32, population_size);

  output_inttype* output_ptr = (output_inttype *) output.data_ptr();
  const dim3 threads(32, 32, 1); 

  consolidate_bits<<<blocks, threads>>>(
          input_ptr, 
          output_ptr, 
          in_ints,
          batch_size,
          out_total_int32s
          );
  return output;
}



at::Tensor binary_forward_cuda(
        const at::Tensor& input, 
        const at::Tensor& weight, 
        const int64_t thresh,
        const bool verbose,
        const bool use_xor
        ) {


  const unsigned int population_size = input.sizes()[0];
  const unsigned int batch_size = input.sizes()[1];
  const unsigned int in_ints = input.sizes()[2]; // * 2 because input is an int64 tensor, and we want to count the int32s
  const unsigned int out_tile_y_size = OUT_TILE_Y_MULTIPLICITY * 32; 
  const unsigned int out_tile_x_ints = OUT_TILE_X_MULTIPLICITY; 
  const dim3 threads(32, 32, 1); 

  const unsigned int warp_size = 32;

  unsigned int sharedMemSize; // TODO how to make const and init in if statement?

  TORCH_CHECK(weight.sizes().size() == 4); // (population, input, 2, output) or (p,i,1,o)
  const unsigned int out_size = weight.sizes()[3];

  if (use_xor) {
    TORCH_CHECK(weight.sizes()[2] == 1); // population, input, 1, output
    sharedMemSize = DEVICE_INTTYPE_BYTES * (((warp_size * out_tile_y_size) + (warp_size * warp_size * out_tile_x_ints))); 
  } else {
    TORCH_CHECK(weight.sizes()[2] == 2); // population, input, 2, output
    sharedMemSize = DEVICE_INTTYPE_BYTES * (((warp_size * out_tile_y_size) + 2 * (warp_size * warp_size * out_tile_x_ints))); 
  }

  
  TORCH_CHECK(input.sizes().size() == 3); // population, batch, input
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


  at::Tensor output;

  const unsigned int out_int64s  = (out_size + 63) / 64;
  const unsigned int out_int32s  = out_int64s * 2;
  // out_ints * 2 because out_ints is with respect to 64-bit precision, whereas the output at the kernel
  // level is 32-bit precision, see below comment as well TODO update comment
  const dim3 blocks((out_int32s + out_tile_x_ints - 1) / out_tile_x_ints, (batch_size + out_tile_y_size - 1) / out_tile_y_size, population_size);


  auto options = input_contig.options().dtype(torch_device_inttype);
  if (thresh > 0) {
      // note that the tensor is created with 64-bit precisions here, TODO update comment
      // because pytorch expects integer tensors to have 64-bit precision,
      // but the data pointer will be cast to 32-bit precisions when sent to the kernel.
      // This is because outputs are generated one bit per warp and collected with __ballot_sync,
      // so we need the output to have 32-bit precision.
      output = torch::zeros({population_size, batch_size, out_int64s}, options);
  } else {
      // here we have no thresholding, so the output tensor has to be larger
      output = torch::zeros({population_size, batch_size, out_size}, options);
  }
      
  void * output_ptr = (void *) output.data_ptr();


  int maxbytes = 65536; // TODO is this actually necessary
  if (use_xor) {
    cudaFuncSetAttribute(binary_forward_XOR, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    binary_forward_XOR<<<blocks, threads, sharedMemSize>>>(
            input_ptr, 
            weight_ptr, 
            output_ptr, 
            thresh,
            out_tile_y_size,
            out_tile_x_ints,
            in_ints,
            population_size,
            batch_size,
            out_size,
            verbose
            );
  } else {
    cudaFuncSetAttribute(binary_forward_AND, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    binary_forward_AND<<<blocks, threads, sharedMemSize>>>(
            input_ptr, 
            weight_ptr, 
            output_ptr, 
            thresh,
            out_tile_y_size,
            out_tile_x_ints,
            in_ints,
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
  m.def("consolidate_bits_cuda", &consolidate_bits_cuda);
}

}

