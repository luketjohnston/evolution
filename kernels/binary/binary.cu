#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"
#include "debug.h"
#include <time.h> // for srand init
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <bitset>


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

typedef int32_t device_inttype;
const int DEVICE_INTTYPE_BITS{sizeof(device_inttype) * 8};
const int DEVICE_INTTYPE_BYTES{sizeof(device_inttype)};

const bool VERBOSE{false};
const bool COMPARE_WITH_CPU{true};
const unsigned int BATCH_SIZE{7}; // TODO should we make threads be max of batch_size and 32?
const unsigned int POPULATION_SIZE{7};
// IN_SIZE must be large enough for a full warp to load integers into memory.
// so it has to be at least 32*32
const unsigned int IN_SIZE{33}; // in bits
const unsigned int OUT_SIZE{32*4}; // in bits
const unsigned int IN_INTS{(IN_SIZE + DEVICE_INTTYPE_BITS - 1) / DEVICE_INTTYPE_BITS}; 
//const unsigned int OUT_INTS{(OUT_SIZE + DEVICE_INTTYPE_BITS - 1) / DEVICE_INTTYPE_BITS};

// 32 * 32 = 1024, the max threads per block.
// a full warp (32 threads) is used to compute a single output
// haven't tested anything besides 1
// const unsigned int OUT_TILE_X_INTS{1};
// needs to stay at 32 to match warp size
// const unsigned int OUT_TILE_Y_INTS{32};

const unsigned int WEIGHT_SIZE{POPULATION_SIZE * IN_SIZE * OUT_SIZE * 2}; // *2 since we have a weight for the input and for the inverted input
// const unsigned int WEIGHT_INTS{WEIGHT_SIZE / DEVICE_INTTYPE_BITS};

void printInput(at::Tensor input) {
  const unsigned int population_size = input.sizes()[0];
  const unsigned int batch_size = input.sizes()[1];
  const unsigned int input_size = input.sizes()[2];
  //std::cout << "Sizes:" << input.sizes() << std::endl;
  ///std::cout << "input[p,b,i]:" << input[p,b,i] << std::endl;

  for (int p = 0; p < population_size; p++) {
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < input_size; i++) {
        std::cout << "p" << p << "b" << b << ":" << i << ":" << std::bitset<DEVICE_INTTYPE_BITS>( input.index({p,b,i}).item<device_inttype>()) << std::endl;
      }
    }
  }
}
  
    

void printWeight(at::Tensor weight) {
  const unsigned int population_size = weight.sizes()[1];
  const unsigned int input_size = weight.sizes()[2];
  const unsigned int out_size = weight.sizes()[3];
  for (int p = 0; p < population_size; p++) {
    for (int i = 0; i < input_size; i++) {
      for (int o = 0; o < out_size; o++) {
        for (int t = 0; t < 2; t++) {
          std::cout << "p" << p << "i" << i << "o" << o << "t" << t << ":" << std::bitset<DEVICE_INTTYPE_BITS>( weight.index({t,p,i,o}).item<device_inttype>()) << std::endl;
        }
      }
    }
  }
}

void printOut(at::Tensor out) {
  const unsigned int population_size = out.sizes()[0];
  const unsigned int batch_size = out.sizes()[1];
  const unsigned int out_size = out.sizes()[2];
  for (int p = 0; p < population_size; p++) {
    for (int b = 0; b < batch_size; b++) {
      for (int o = 0; o < out_size; o++) {
        std::cout << "p" << p << "b" << b << "o" << o << ":" << std::bitset<DEVICE_INTTYPE_BITS>( out.index({p,b,o}).item<device_inttype>()) << std::endl;
      }
    }
  }
}
      

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


typedef unsigned int intType_t;


const intType_t WARP_SIZE{32};


/**
In order to use all 32 warps at the final ballot_sync, we need the output
to be a multiple of 32.
This means that each of the 32 warps must compute popcount(w * input)
*/




// note that i is the index into the input as an array of integers, NOT as an array of bits.
__host__ __device__ intType_t i_ind(const intType_t b, const intType_t p, const intType_t i, const device_inttype population_size, const device_inttype in_ints) {
 
  return b * population_size * in_ints + p * in_ints + i;
}

// note that o is the index into output as an array of integers, NOT as an array of bits.
__host__ __device__ intType_t o_ind(const intType_t b, const intType_t p, const intType_t o, const device_inttype population_size, const device_inttype out_ints) {
  return b * population_size * out_ints + p * out_ints + o;
}

// note the "o" arg to w_ind should be different than the "o" arg to o_ind above.
// (because w has a full integer entry for each output bit o)
__host__ __device__ intType_t w_ind(const intType_t t, const intType_t p, const intType_t i, const intType_t o, const device_inttype population_size, const device_inttype in_ints, const device_inttype out_size) {
  return t * in_ints * population_size * out_size + p * in_ints * out_size + i * out_size + o;
}



// Computes the binary matrix multiplication on the host and saves the output
// out should be zeroed
void host_op(
    device_inttype const * const input, 
    device_inttype const * const weight, 
    device_inttype * out, 
    int thresh,
    const device_inttype population_size,
    const device_inttype batch_size,
    const device_inttype in_ints,
    const device_inttype out_size
    ) {
  const device_inttype out_ints = out_size / 8 / sizeof(device_inttype);

  for( intType_t b = 0; b < batch_size; b++) {
    for( intType_t p = 0; p < population_size; p++) {
      for( intType_t o = 0; o < out_size; o++) {

        unsigned int temp = 0;

        for( intType_t i = 0; i < in_ints; i++) {
          temp += __builtin_popcount( (input[i_ind(b,p,i,population_size,in_ints)] & weight[w_ind(0,p,i,o,population_size,in_ints,out_size)]));
          temp += __builtin_popcount((~input[i_ind(b,p,i,population_size,in_ints)]) & weight[w_ind(1,p,i,o,population_size,in_ints,out_size)]);
        }
        // NOTE that the way we shift here, the outputs are ordered low-precision bits first
        device_inttype bit_to_set = (temp >= thresh) << (o % (8 * sizeof(device_inttype)));
        device_inttype o_int_index = o / (8 * sizeof(device_inttype));
        out[o_ind(b,p,o_int_index,population_size,out_ints)] = out[o_ind(b,p,o_int_index,population_size,out_ints)] | bit_to_set;
      }
    }
  }
}

at::Tensor host_helper(at::Tensor input, at::Tensor weight, int thresh) {
  const unsigned int population_size = input.sizes()[0];
  const unsigned int batch_size = input.sizes()[1];
  const unsigned int in_ints = input.sizes()[2];
  const unsigned int out_size = weight.sizes()[3];
  TORCH_CHECK(input.sizes().size() == 3); // population, batch, input
  TORCH_CHECK(weight.sizes().size() == 4); // 2, population, input, output
  // TODO we shouldn't need this check, need to make kernel work for any batch size etc.
  //TORCH_CHECK(input.sizes()[1] >= 32); 
  TORCH_CHECK(input.sizes()[0] == weight.sizes()[1]); // check both have same population size
  TORCH_CHECK(input.sizes()[2] == weight.sizes()[2]); // check both have same input size

  TORCH_CHECK(input.dtype() == torch::kInt32); 
  TORCH_CHECK(weight.dtype() == torch::kInt32);

  at::Tensor input_contig = input.contiguous();
  at::Tensor weight_contig = weight.contiguous();
  const device_inttype* input_ptr = (device_inttype *) input_contig.data_ptr();
  const device_inttype* weight_ptr = (device_inttype *) weight_contig.data_ptr();
  at::Tensor output;
  auto options = input_contig.options().dtype(torch::kInt32);
  const unsigned int out_ints  = (out_size + DEVICE_INTTYPE_BITS - 1) / DEVICE_INTTYPE_BITS;
  output = torch::zeros({population_size, batch_size, out_ints}, options);
  device_inttype* output_ptr = (device_inttype *) output.data_ptr();
  host_op(input_ptr, weight_ptr, output_ptr, thresh, population_size, batch_size, in_ints, out_size);
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
        unsigned short * out, 
        const unsigned int out_tile_x_ints,
        const unsigned int out_tile_y_ints, 
        const unsigned int in_ints, 
        const unsigned int warp_size,
        const unsigned int population_size,
        const unsigned int batch_size,
        const unsigned int out_size
        ) {

  extern __shared__ device_inttype shared[];

  device_inttype* weight_tile{&shared[0]};
  // TODO replace 32 with a constant or something
  device_inttype* not_weight_tile{&shared[out_tile_x_ints * warp_size * warp_size]};
  device_inttype* input_tile{&shared[2 * out_tile_x_ints * warp_size * warp_size]};

  const intType_t output_tile_x = blockIdx.x * out_tile_x_ints;
  const intType_t output_tile_y = blockIdx.y * out_tile_y_ints;
  const intType_t b = output_tile_y + threadIdx.y;
  const unsigned int out_ints = out_size / 8 / sizeof(device_inttype);

  const intType_t p = blockIdx.z;

  unsigned int acc = 0;

  for (unsigned int tile_i = 0; tile_i < (in_ints + warp_size - 1) / warp_size; ++tile_i) {

    intType_t i = tile_i * warp_size + threadIdx.x;

    // Load input into shared memory
    //if (threadIdx.x < warp_size) { 
    if (threadIdx.x < warp_size && b < batch_size && i < in_ints) { 
      input_tile[threadIdx.y * warp_size + threadIdx.x] = input[i_ind(b, p, i, population_size, in_ints)]; 
    }
    //if (true) {
    // load weight into shared memory
    intType_t w_i_to_load =  tile_i * warp_size + threadIdx.y;
    intType_t w_o_to_load = blockIdx.x * out_tile_x_ints * 32 + threadIdx.x;
    if (b < batch_size && w_i_to_load < in_ints) {
      // "o" varies with threadIdx.x and is the last coord so access is coalesced
      weight_tile[    threadIdx.y * 32 +  threadIdx.x] = weight[w_ind(0, p, w_i_to_load, w_o_to_load, population_size, in_ints, out_size)]; 
      not_weight_tile[threadIdx.y * 32 +  threadIdx.x] = weight[w_ind(1, p, w_i_to_load, w_o_to_load, population_size, in_ints, out_size)]; 
    }

    __syncthreads(); // wait until shared memory load is finished before continuing with compute

    for (unsigned int j = 0; j < warp_size; ++j) {

      //if (true) {
      if (b < batch_size && (j + tile_i * warp_size < in_ints)) {
        acc += __popc(input_tile[threadIdx.y * warp_size + j] & weight_tile[j * warp_size + threadIdx.x]);
        acc += __popc((~input_tile[threadIdx.y * warp_size + j]) & not_weight_tile[j * warp_size + threadIdx.x]);
      }
    }
    
    __syncthreads(); // wait until this compute is finished before next loop replaces shared memory

  }
  //if (true) { 
  if (b < batch_size) { 
    out[o_ind(b,p,output_tile_x + threadIdx.x,population_size,out_ints)] = acc;
  }
}

__global__ void binary_forward_with_threshold(
        device_inttype const * const input, 
        device_inttype const * const weight, 
        device_inttype * out, 
        const unsigned int thresh, 
        const unsigned int out_tile_x_ints,
        const unsigned int out_tile_y_ints, 
        const unsigned int in_ints, 
        const unsigned int warp_size,
        const unsigned int population_size,
        const unsigned int batch_size,
        const unsigned int out_size
        ) {

  extern __shared__ device_inttype shared[];

  device_inttype* weight_tile{&shared[0]};
  // TODO replace 32 with a constant or something
  device_inttype* not_weight_tile{&shared[out_tile_x_ints * warp_size * warp_size]};
  device_inttype* input_tile{&shared[2 * out_tile_x_ints * warp_size * warp_size]};

  const intType_t output_tile_x = blockIdx.x * out_tile_x_ints;
  const intType_t output_tile_y = blockIdx.y * out_tile_y_ints;
  const intType_t b = output_tile_y + threadIdx.y;
  const unsigned int out_ints = out_size / 8 / sizeof(device_inttype);

  const intType_t p = blockIdx.z;

  unsigned int acc = 0;

  
  for (unsigned int tile_i = 0; tile_i < (in_ints + warp_size - 1) / warp_size; ++tile_i) {

    intType_t i = tile_i * warp_size + threadIdx.x;

    // Load input into shared memory
    //if (threadIdx.x < warp_size) { 
    if (threadIdx.x < warp_size &&  b < batch_size && i < in_ints) { 
      if (VERBOSE && threadIdx.x < 3 && threadIdx.y < 1) {printf("threadIdx.x %u threadIdx.y %u setting input\n", threadIdx.x, threadIdx.y);};
      input_tile[threadIdx.y * warp_size + threadIdx.x] = input[i_ind(b, p, i, population_size, in_ints)]; 
    }

    //if (true) {
    // load weight into shared memory
    intType_t w_i_to_load =  tile_i * warp_size + threadIdx.y;
    // TODO need to set output integers to be 32 bits and the rest to be 64 
    intType_t w_o_to_load = blockIdx.x * out_tile_x_ints * warp_size + threadIdx.x;

    if (VERBOSE && threadIdx.x < 3 && threadIdx.y < 1) {printf("threadIdx.x %u threadIdx.y %u w_i_to_load: %u in_ints: %u\n", threadIdx.x, threadIdx.y, w_i_to_load, in_ints);};
    if (w_i_to_load < in_ints) {
      // "o" varies with threadIdx.x and is the last coord so access is coalesced

      if (VERBOSE && threadIdx.x < 3 && threadIdx.y < 1) {printf("threadIdx.x %u threadIdx.y %u setting shared mem weights\n", threadIdx.x, threadIdx.y);};

      weight_tile[    threadIdx.y * warp_size +  threadIdx.x] = weight[w_ind(0, p, w_i_to_load, w_o_to_load, population_size, in_ints, out_size)]; 
      not_weight_tile[threadIdx.y * warp_size +  threadIdx.x] = weight[w_ind(1, p, w_i_to_load, w_o_to_load, population_size, in_ints, out_size)]; 
    }

    __syncthreads(); // wait until shared memory load is finished before continuing with compute

    for (unsigned int j = 0; j < warp_size; ++j) {

      //if (true) {
      
      if (VERBOSE && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u, j:%u,  j+tile_i*warp_size:%u \n" , threadIdx.x, threadIdx.y, j, j + tile_i * warp_size);};

      if (b < batch_size && (j + tile_i * warp_size < in_ints)) { // TODO double check?

        device_inttype x1 = input_tile[threadIdx.y * warp_size + j];
        if (VERBOSE && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u j:%u: input_tile[...]: " BBP BBP BBP BBP "\n", threadIdx.x, threadIdx.y, j, BB(x1>>24), BB(x1>>16), BB(x1>>8), BB(x1));};

        x1 = weight_tile[j * warp_size + threadIdx.x];
        if (VERBOSE && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u j:%u: weight_tile[...]: " BBP BBP BBP BBP "\n", threadIdx.x, threadIdx.y, j, BB(x1>>24), BB(x1>>16), BB(x1>>8), BB(x1));};
        x1 = not_weight_tile[j * warp_size + threadIdx.x];
        if (VERBOSE && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u j:%u: not_weight_tile[...]: " BBP BBP BBP BBP "\n", threadIdx.x, threadIdx.y, j, BB(x1>>24), BB(x1>>16), BB(x1>>8), BB(x1));};

        unsigned int tmp = acc;

        acc += __popc(input_tile[threadIdx.y * warp_size + j] & weight_tile[j * warp_size + threadIdx.x]);
        acc += __popc((~input_tile[threadIdx.y * warp_size + j]) & not_weight_tile[j * warp_size + threadIdx.x]);

        if (VERBOSE && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u j:%u adding to acc: %u \n" , threadIdx.x, threadIdx.y, j, acc - tmp);};

      } 
    }
    
    __syncthreads(); // wait until this compute is finished before next loop replaces shared memory

  }

  __syncthreads(); // TODO I dont think we need this

  if (VERBOSE && threadIdx.x < 3 && threadIdx.y < 1) {printf("%u.%u acc:%u:\n" , threadIdx.x, threadIdx.y, acc);};

  // activemask will always be 1111..., TODO 
  unsigned int r = __ballot_sync(__activemask(), acc >= thresh);
  // after ballot_sync, only one thread per warp needs to write to output
  //if (threadIdx.x % warp_size == 0) {
  if (threadIdx.x % warp_size == 0 && b < batch_size) {
    out[o_ind(b,p,output_tile_x,population_size,out_ints)] = r;
  }
  __syncthreads(); // TODO I dont think we need this
}


at::Tensor binary_forward_cuda(
        const at::Tensor& input, 
        const at::Tensor& weight, 
        const device_inttype thresh
        ) {


  const unsigned int population_size = input.sizes()[0];
  const unsigned int batch_size = input.sizes()[1];
  const unsigned int in_ints = input.sizes()[2];
  const unsigned int out_size = weight.sizes()[3];
  
  TORCH_CHECK(input.sizes().size() == 3); // population, batch, input
  TORCH_CHECK(weight.sizes().size() == 4); // 2, population, input, output

  // TODO we shouldn't need this check, need to make kernel work for any batch size etc.
  //TORCH_CHECK(input.sizes()[1] >= 32); 

  TORCH_CHECK(input.sizes()[0] == weight.sizes()[1]); // check both have same population size
  TORCH_CHECK(input.sizes()[2] == weight.sizes()[2]); // check both have same input size

  TORCH_CHECK(input.dtype() == torch::kInt32); 
  TORCH_CHECK(weight.dtype() == torch::kInt32);

  TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(weight.device().type() == at::DeviceType::CUDA);

  at::Tensor input_contig = input.contiguous();
  at::Tensor weight_contig = weight.contiguous();


  // TODO can we have result be shaped correctly or does it have to be contiguous?
  //unsigned int output_tensor_size[1] = {population_size*batch_size*out_size};
  //unsigned int output_tensor_size_array_ref = ArrayRef(&output_tensor_size, 1);

  const device_inttype* input_ptr = (device_inttype *) input_contig.data_ptr();
  const device_inttype* weight_ptr = (device_inttype *) weight_contig.data_ptr();

  const dim3 threads(32, 32, 1); 
  const unsigned int out_tile_y_ints = WARP_SIZE; // keep as warp size

  // TODO replace 32's with the appropriate constant

  const unsigned int warp_size = WARP_SIZE;
  const unsigned int sharedMemSize = DEVICE_INTTYPE_BYTES * (((warp_size * out_tile_y_ints) + 2 * (warp_size * warp_size))); 

  at::Tensor output;

  printf("Checking thresh");
  if (thresh > 0) {
      // TODO change 64s to sizeof(device_inttype)
      const unsigned int out_ints  = (out_size + DEVICE_INTTYPE_BITS - 1) / DEVICE_INTTYPE_BITS;

      // TODO I don't think this works when out_size is not a multiple of 64
      auto options = input_contig.options().dtype(torch::kInt32);

      output = torch::empty({population_size, batch_size, out_ints}, options);
      device_inttype* output_ptr = (device_inttype *) output.data_ptr();

      // TODO check that this integer division works like I expect
      const dim3 blocks(out_ints, (batch_size + out_tile_y_ints - 1) / out_tile_y_ints, population_size);
      printf("Calling binary forward with threshold...");

      binary_forward_with_threshold<<<blocks, threads, sharedMemSize>>>(
              input_ptr, 
              weight_ptr, 
              output_ptr, 
              thresh,
              1,
              out_tile_y_ints,
              in_ints,
              warp_size,
              population_size,
              batch_size,
              out_size
              );
  } else {
      // here we have no thresholding, so the output tensor has to be larger
      auto options = input_contig.options().dtype(torch::kInt16); // 2**16 is plenty large enough hold activations
      output = torch::zeros({population_size, batch_size, out_size}, options);
      unsigned short* output_ptr = (unsigned short *) output.data_ptr();
      const unsigned int out_tile_x_ints = warp_size;
     
      const dim3 blocks(out_size, (batch_size + out_tile_y_ints - 1) / out_tile_y_ints, population_size);

      printf("Calling binary forward...");
      binary_forward<<<blocks, threads, sharedMemSize>>>(
              input_ptr, 
              weight_ptr, 
              output_ptr, 
              out_tile_x_ints,
              out_tile_y_ints,
              in_ints,
              warp_size,
              population_size,
              batch_size,
              out_size
              );
  }
      

  return output;
}

//TORCH_LIBRARY(binary_forward, m) {
//  m.def("binary_forward_cuda", &binary_forward_cuda);
//}

}



int main( int argc, char *argv[] )
{

  srand(time(NULL));;
  /* get GPU device number and name */
  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

  fprintf(stdout, "Input size is %d\n",IN_SIZE);
  fprintf(stdout, "Batch size is %d\n",BATCH_SIZE);
  fprintf(stdout, "Population size is %d\n",POPULATION_SIZE);
  fprintf(stdout, "sizeof(int)%lu\n",sizeof(int));
  fprintf(stdout, "sizeof(device_inttype)%lu\n",sizeof(device_inttype));

  auto options =
      torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);


  const device_inttype maxlong = std::numeric_limits<device_inttype>::max();
  const device_inttype minlong = std::numeric_limits<device_inttype>::min();


  const at::Tensor h_input = torch::randint(minlong,maxlong,{POPULATION_SIZE,BATCH_SIZE,IN_INTS},options);
  const at::Tensor h_weight = torch::randint(minlong,maxlong,{2,POPULATION_SIZE,IN_INTS,OUT_SIZE},options);

  const at::Tensor d_input = h_input.to(torch::kCUDA);
  const at::Tensor d_weight = h_weight.to(torch::kCUDA);

  // Note this is different than IN_SIZE / 2 because when IN_SIZE is not divisible
  // by DEVICE_INTTYPE_BITS, we add extend IN_SIZE until it is. 
  const int threshold = IN_INTS * DEVICE_INTTYPE_BITS / 2;

  // start timers
  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );


  /////////
  // GPU //
  /////////

  checkCUDA( cudaEventRecord( start, 0 ) );

  at::Tensor d_out = binary_forward::binary_forward_cuda(
    d_input,
    d_weight,
    threshold);

  checkKERNEL();

  // stop timer and print time
  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GBop/s\n", ( ( (double) BATCH_SIZE *
    (double) POPULATION_SIZE * 
    (double) OUT_SIZE * (double) IN_SIZE * 2.0 / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 ))); // TODO check this computation , havent checked

  if (COMPARE_WITH_CPU) {
    // start timer
    checkCUDA( cudaEventRecord( start, 0 ) );

  
    // do convolution on cpu
    at::Tensor h_out = binary_forward::host_helper(h_input, h_weight, threshold);
  
    // stop timers
    checkCUDA( cudaEventRecord( stop, 0 ) );
    checkCUDA( cudaEventSynchronize( stop ) );
    checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  
    // print time taken
    fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
    checkCUDA( cudaEventDestroy( start ) );
    checkCUDA( cudaEventDestroy( stop ) );
  
    // compare GPU implementation results with CPU results

    //std::cout << "Host out:" << std::endl << h_out << std::endl;
    //printOut(h_out);
    //std::cout << "Device out:" << std::endl << d_out.to(torch::kCPU) << std::endl;
    //printOut(d_out.to(torch::kCPU));
    //printInput(h_input);

    //printf("Weight:\n");
    //printWeight(h_weight);
    //printf("Device weight:\n");
    //printWeight(d_weight.to(torch::kCPU));

    device_inttype diff = torch::sum(torch::abs(h_out - d_out.to(torch::kCPU))).item<device_inttype>();
    
    printf("error is %ld\n",diff);
    if( diff != 0 ) printf("FAIL\n");
    else printf("PASS\n");
  }
    

  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

  return 0;


}
