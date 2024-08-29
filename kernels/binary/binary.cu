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

const bool VERBOSE{false};
const bool COMPARE_WITH_CPU{true};
const unsigned int BATCH_SIZE{128}; // TODO when we make 32 again, change actual kernel call threads also
const unsigned int POPULATION_SIZE{16};
// IN_SIZE must be large enough for a full warp to load integers into memory.
// so it has to be at least 32*32
const unsigned int IN_SIZE{32*32}; // in bits
const unsigned int OUT_SIZE{128}; // in bits
const unsigned int IN_INTS{IN_SIZE / 8 / sizeof(int64_t)}; 
const unsigned int OUT_INTS{OUT_SIZE / 8 / sizeof(int64_t)};

// 32 * 32 = 1024, the max threads per block.
// a full warp (32 threads) is used to compute a single output
// haven't tested anything besides 1
const unsigned int OUT_TILE_X_INTS{1};
// needs to stay at 32 to match warp size
const unsigned int OUT_TILE_Y_INTS{32};

// Needs to stay at 32 to match warp size
const unsigned int SUBTILE_INTS{32};

const unsigned int WEIGHT_SIZE{POPULATION_SIZE * IN_SIZE * OUT_SIZE * 2}; // *2 since we have a weight for the input and for the inverted input
const unsigned int WEIGHT_INTS{WEIGHT_SIZE / 8 / sizeof(int64_t)};

namespace binary_forward {

typedef int64_t myIntType_t;

__host__ __device__ void printbinary(myIntType_t i) {
    printf(BBP BBP BBP BBP "\n", BB(i>>24), BB(i>>16), BB(i>>8), BB(i));
}

__host__ __device__ void printbinary(char* input, myIntType_t i) {
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
__host__ __device__ intType_t i_ind(const intType_t b, const intType_t p, const intType_t i, const myIntType_t population_size, const myIntType_t in_ints) {
 
  return b * population_size * in_ints + p * in_ints + i;
}

// note that o is the index into output as an array of integers, NOT as an array of bits.
__host__ __device__ intType_t o_ind(const intType_t b, const intType_t p, const intType_t o, const myIntType_t population_size, const myIntType_t out_ints) {
  return b * population_size * out_ints + p * out_ints + o;
}

// note the "o" arg to w_ind should be different than the "o" arg to o_ind above.
// (because w has a full integer entry for each output bit o)
__host__ __device__ intType_t w_ind(const intType_t t, const intType_t p, const intType_t i, const intType_t o, const myIntType_t population_size, const myIntType_t in_ints, const myIntType_t out_size) {
  return t * in_ints * population_size * out_size + p * in_ints * out_size + i * out_size + o;
}



// Computes the binary matrix multiplication on the host and saves the output
// out should be zeroed
void host_op(
    myIntType_t const * const input, 
    myIntType_t const * const weight, 
    myIntType_t * out, 
    int thresh,
    const myIntType_t population_size,
    const myIntType_t batch_size,
    const myIntType_t in_ints,
    const myIntType_t out_size
    ) {
  const myIntType_t out_ints = out_size / 8 / sizeof(int64_t);

  for( intType_t b = 0; b < batch_size; b++) {
    for( intType_t p = 0; p < population_size; p++) {
      for( intType_t o = 0; o < out_size; o++) {

        unsigned int temp = 0;

        for( intType_t i = 0; i < in_ints; i++) {
          temp += __builtin_popcount( (input[i_ind(b,p,i,population_size,in_ints)] & weight[w_ind(0,p,i,o,population_size,in_ints,out_size)]));
          temp += __builtin_popcount((~input[i_ind(b,p,i,population_size,in_ints)]) & weight[w_ind(1,p,i,o,population_size,in_ints,out_size)]);
        }
        // NOTE that the way we shift here, the outputs are ordered low-precision bits first
        myIntType_t bit_to_set = (temp >= thresh) << (o % (8 * sizeof(myIntType_t)));
        myIntType_t o_int_index = o / (8 * sizeof(myIntType_t));
        out[o_ind(b,p,o_int_index,population_size,out_ints)] = out[o_ind(b,p,o_int_index,population_size,out_ints)] | bit_to_set;
      }
    }
  }
}

at::Tensor host_helper(at::Tensor input, at::Tensor weight, int thresh) {
  const unsigned int population_size = input.sizes()[0];
  const unsigned int batch_size = input.sizes()[1];
  const unsigned int input_size = input.sizes()[2];
  const unsigned int out_size = weight.sizes()[3];
  TORCH_CHECK(input.sizes().size() == 3); // population, batch, input
  TORCH_CHECK(weight.sizes().size() == 4); // 2, population, input, output
  // TODO we shouldn't need this check, need to make kernel work for any batch size etc.
  TORCH_CHECK(input.sizes()[1] >= 32); 
  TORCH_CHECK(input.sizes()[0] == weight.sizes()[1]); // check both have same population size
  TORCH_CHECK(input.sizes()[2] == weight.sizes()[2]); // check both have same input size
  TORCH_CHECK(input.dtype() == at::kLong); 
  TORCH_CHECK(weight.dtype() == at::kLong);

  at::Tensor input_contig = input.contiguous();
  at::Tensor weight_contig = weight.contiguous();
  const int64_t* input_ptr = (int64_t *) input_contig.data_ptr();
  const int64_t* weight_ptr = (int64_t *) weight_contig.data_ptr();
  at::Tensor output;
  auto options = input_contig.options().dtype(torch::kInt64);
  const unsigned int out_ints  = (out_size + 63) / 64;
  const unsigned int in_ints  = (input_size + 63) / 64;
  output = torch::empty({population_size*batch_size*out_ints}, options);
  int64_t* output_ptr = (int64_t *) output.data_ptr();
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
  TODO docs
*/
__global__ void binary_forward(
        myIntType_t const * const input, 
        myIntType_t const * const weight, 
        unsigned short * out, 
        const unsigned int out_tile_x_ints,
        const unsigned int out_tile_y_ints, 
        const unsigned int subtile_ints,
        const unsigned int in_ints, 
        const unsigned int warp_size,
        const unsigned int population_size,
        const unsigned int batch_size,
        const unsigned int out_size
        ) {

  extern __shared__ myIntType_t shared[];

  myIntType_t* weight_tile{&shared[0]};
  // TODO replace 32 with a constant or something
  myIntType_t* not_weight_tile{&shared[out_tile_x_ints * subtile_ints * 32]};
  myIntType_t* input_tile{&shared[2 * out_tile_x_ints * subtile_ints * 32]};

  const intType_t output_tile_x = blockIdx.x * out_tile_x_ints;
  const intType_t output_tile_y = blockIdx.y * out_tile_y_ints;
  const intType_t b = output_tile_y + threadIdx.y;
  const unsigned int out_ints = out_size / 8 / sizeof(int64_t);

  const intType_t p = blockIdx.z;

  unsigned int acc = 0;

  for (unsigned int tile_i = 0; tile_i < in_ints / subtile_ints; ++tile_i) {

    intType_t i = tile_i * subtile_ints + threadIdx.x;

    // Load input into shared memory
    if (threadIdx.x < subtile_ints) { 
      input_tile[threadIdx.y * subtile_ints + threadIdx.x] = input[i_ind(b, p, i, population_size, in_ints)]; 
    }
    // load weight into shared memory
    intType_t w_i_to_load =  tile_i * subtile_ints + threadIdx.y;
    intType_t w_o_to_load = blockIdx.x * out_tile_x_ints * 32 + threadIdx.x;

    // "o" varies with threadIdx.x and is the last coord so access is coalesced
    weight_tile[    threadIdx.y * 32 +  threadIdx.x] = weight[w_ind(0, p, w_i_to_load, w_o_to_load, population_size, in_ints, out_size)]; 
    not_weight_tile[threadIdx.y * 32 +  threadIdx.x] = weight[w_ind(1, p, w_i_to_load, w_o_to_load, population_size, in_ints, out_size)]; 

    __syncthreads(); // wait until shared memory load is finished before continuing with compute

    for (unsigned int j = 0; j < subtile_ints; ++j) {

      acc += __popc(input_tile[threadIdx.y * subtile_ints + j] & weight_tile[j * subtile_ints + threadIdx.x]);
      acc += __popc((~input_tile[threadIdx.y * subtile_ints + j]) & not_weight_tile[j * subtile_ints + threadIdx.x]);
    }
    
    __syncthreads(); // wait until this compute is finished before next loop replaces shared memory

  }
  out[o_ind(b,p,output_tile_x + threadIdx.x,population_size,out_ints)] = acc;
}

__global__ void binary_forward_with_threshold(
        myIntType_t const * const input, 
        myIntType_t const * const weight, 
        myIntType_t * out, 
        const unsigned int thresh, 
        const unsigned int out_tile_x_ints,
        const unsigned int out_tile_y_ints, 
        const unsigned int subtile_ints,
        const unsigned int in_ints, 
        const unsigned int warp_size,
        const unsigned int population_size,
        const unsigned int batch_size,
        const unsigned int out_size
        ) {

  extern __shared__ myIntType_t shared[];

  myIntType_t* weight_tile{&shared[0]};
  // TODO replace 32 with a constant or something
  myIntType_t* not_weight_tile{&shared[out_tile_x_ints * subtile_ints * 32]};
  myIntType_t* input_tile{&shared[2 * out_tile_x_ints * subtile_ints * 32]};

  const intType_t output_tile_x = blockIdx.x * out_tile_x_ints;
  const intType_t output_tile_y = blockIdx.y * out_tile_y_ints;
  const intType_t b = output_tile_y + threadIdx.y;
  const unsigned int out_ints = out_size / 8 / sizeof(int64_t);

  const intType_t p = blockIdx.z;

  unsigned int acc = 0;

  for (unsigned int tile_i = 0; tile_i < in_ints / subtile_ints; ++tile_i) {

    intType_t i = tile_i * subtile_ints + threadIdx.x;

    // Load input into shared memory
    if (threadIdx.x < subtile_ints) { 
      input_tile[threadIdx.y * subtile_ints + threadIdx.x] = input[i_ind(b, p, i, population_size, in_ints)]; 
    }
    // load weight into shared memory
    intType_t w_i_to_load =  tile_i * subtile_ints + threadIdx.y;
    intType_t w_o_to_load = blockIdx.x * out_tile_x_ints * 32 + threadIdx.x;

    // "o" varies with threadIdx.x and is the last coord so access is coalesced
    weight_tile[    threadIdx.y * 32 +  threadIdx.x] = weight[w_ind(0, p, w_i_to_load, w_o_to_load, population_size, in_ints, out_size)]; 
    not_weight_tile[threadIdx.y * 32 +  threadIdx.x] = weight[w_ind(1, p, w_i_to_load, w_o_to_load, population_size, in_ints, out_size)]; 

    __syncthreads(); // wait until shared memory load is finished before continuing with compute

    for (unsigned int j = 0; j < subtile_ints; ++j) {

      acc += __popc(input_tile[threadIdx.y * subtile_ints + j] & weight_tile[j * subtile_ints + threadIdx.x]);
      acc += __popc((~input_tile[threadIdx.y * subtile_ints + j]) & not_weight_tile[j * subtile_ints + threadIdx.x]);
    }
    
    __syncthreads(); // wait until this compute is finished before next loop replaces shared memory

  }

  // activemask will always be 1111..., TODO 
  unsigned int r = __ballot_sync(__activemask(), acc >= thresh);
  // after ballot_sync, only one thread per warp needs to write to output
  if (threadIdx.x % warp_size == 0) {
    out[o_ind(b,p,output_tile_x,population_size,out_ints)] = r;
  }
}


at::Tensor binary_forward_cuda(
        const at::Tensor& input, 
        const at::Tensor& weight, 
        const int64_t thresh
        ) {


  const unsigned int population_size = input.sizes()[0];
  const unsigned int batch_size = input.sizes()[1];
  const unsigned int input_size = input.sizes()[2];
  const unsigned int out_size = weight.sizes()[3];
  
  TORCH_CHECK(input.sizes().size() == 3); // population, batch, input
  TORCH_CHECK(weight.sizes().size() == 4); // 2, population, input, output

  // TODO we shouldn't need this check, need to make kernel work for any batch size etc.
  TORCH_CHECK(input.sizes()[1] >= 32); 

  TORCH_CHECK(input.sizes()[0] == weight.sizes()[1]); // check both have same population size
  TORCH_CHECK(input.sizes()[2] == weight.sizes()[2]); // check both have same input size

  TORCH_CHECK(input.dtype() == at::kLong); 
  TORCH_CHECK(weight.dtype() == at::kLong);

  TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(weight.device().type() == at::DeviceType::CUDA);

  at::Tensor input_contig = input.contiguous();
  at::Tensor weight_contig = weight.contiguous();


  // TODO can we have result be shaped correctly or does it have to be contiguous?
  //unsigned int output_tensor_size[1] = {population_size*batch_size*out_size};
  //unsigned int output_tensor_size_array_ref = ArrayRef(&output_tensor_size, 1);

  const int64_t* input_ptr = (int64_t *) input_contig.data_ptr();
  const int64_t* weight_ptr = (int64_t *) weight_contig.data_ptr();

  const dim3 threads(32, 32, 1); // set to 1 while we are testing batch size 1 TODO
  const unsigned int out_tile_y_ints = 32; // stay 32 to match warp size
  const unsigned int subtile_ints = 32;

  // TODO replace 32's with the appropriate constant
  const unsigned int sharedMemSize = 64 * ((subtile_ints * out_tile_y_ints) + 2 * (subtile_ints * 32)); 
  const unsigned int in_ints = (input_size + 63) / 64;
  const unsigned int warp_size = 32;
  at::Tensor output;

  if (thresh > 0) {
      // TODO change 64s to sizeof(int64_t)
      const unsigned int out_ints  = (out_size + 63) / 64;

      // TODO I don't think this works when out_size is not a multiple of 64
      auto options = input_contig.options().dtype(torch::kInt64);

      output = torch::empty({population_size*batch_size*out_ints}, options);
      int64_t* output_ptr = (int64_t *) output.data_ptr();

      // TODO check that this integer division works like I expect
      const dim3 blocks(out_ints, (batch_size + out_tile_y_ints - 1) / out_tile_y_ints, population_size);

      binary_forward_with_threshold<<<blocks, threads, sharedMemSize>>>(
              input_ptr, 
              weight_ptr, 
              output_ptr, 
              thresh,
              1,
              out_tile_y_ints,
              subtile_ints,
              in_ints,
              warp_size,
              population_size,
              batch_size,
              out_size
              );
  } else {
      // here we have no thresholding, so the output tensor has to be larger
      auto options = input_contig.options().dtype(torch::kInt16); // 2**16 is plenty large enough hold activations
      output = torch::zeros({population_size*batch_size*out_size}, options);
      unsigned short* output_ptr = (unsigned short *) output.data_ptr();
      const unsigned int out_tile_x_ints = 32;
     
      const dim3 blocks(out_size, (batch_size + out_tile_y_ints - 1) / out_tile_y_ints, population_size);

      binary_forward<<<blocks, threads, sharedMemSize>>>(
              input_ptr, 
              weight_ptr, 
              output_ptr, 
              out_tile_x_ints,
              out_tile_y_ints,
              subtile_ints,
              in_ints,
              warp_size,
              population_size,
              batch_size,
              out_size
              );
  }
      

  return output;
}

TORCH_LIBRARY(binary_forward, m) {
  m.def("binary_forward_cuda", &binary_forward_cuda);
}

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
  fprintf(stdout, "sizeof(myIntType_t)%lu\n",sizeof(binary_forward::myIntType_t));

  auto options =
      torch::TensorOptions()
        .dtype(torch::kInt64)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);


  const int64_t maxlong = std::numeric_limits<int64_t>::max();
  const int64_t minlong = std::numeric_limits<int64_t>::min();


  const at::Tensor h_input = torch::randint(minlong,maxlong,{IN_INTS,BATCH_SIZE,POPULATION_SIZE},options);
  const at::Tensor h_weight = torch::randint(minlong,maxlong,{2,POPULATION_SIZE,IN_INTS,OUT_SIZE},options);

  const at::Tensor d_input = h_input.to(torch::kCUDA);
  const at::Tensor d_weight = h_weight.to(torch::kCUDA);

  // start timers
  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

  printf("HERE 9\n");

  /////////
  // GPU //
  /////////

  checkCUDA( cudaEventRecord( start, 0 ) );

  at::Tensor d_out = binary_forward::binary_forward_cuda(
    d_input,
    d_weight,
    IN_SIZE / 2);

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

    printf("HERE 10\n");
  
    // do convolution on cpu
    at::Tensor h_out = binary_forward::host_helper(h_input, h_weight, IN_SIZE / 2);
    printf("HERE 11\n");
  
    // stop timers
    checkCUDA( cudaEventRecord( stop, 0 ) );
    checkCUDA( cudaEventSynchronize( stop ) );
    checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  
    // print time taken
    fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
    checkCUDA( cudaEventDestroy( start ) );
    checkCUDA( cudaEventDestroy( stop ) );
  
    // compare GPU implementation results with CPU results
    int diff = torch::sum(torch::abs(h_out - d_out.to(torch::kCPU))).item<int>();
    
    printf("error is %d\n",diff);
    if( diff > 0 ) printf("FAIL\n");
    else printf("PASS\n");
  }
    

  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

  return 0;


}
