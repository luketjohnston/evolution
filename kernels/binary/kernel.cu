#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"
#include "debug.h"
#include <time.h> // for srand init

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

typedef unsigned long myIntType_t;

__host__ __device__ void printbinary(myIntType_t i) {
    printf(BBP BBP BBP BBP "\n", BB(i>>24), BB(i>>16), BB(i>>8), BB(i));
}

__host__ __device__ void printbinary(char* input, myIntType_t i) {
    printf("%s" BBP BBP BBP BBP "\n", input, BB(i>>24), BB(i>>16), BB(i>>8), BB(i));
}

__host__ __device__ void printbinary(unsigned char i) {
    printf(BBP "\n", BB(i));
}

const bool VERBOSE{false};

typedef unsigned int intType_t;


const intType_t WARP_SIZE{32};

const bool COMPARE_WITH_CPU{false};

/**
In order to use all 32 warps at the final ballot_sync, we need the output
to be a multiple of 32.
This means that each of the 32 warps must compute popcount(w * input)
*/


const intType_t BATCH_SIZE{500}; // TODO when we make 32 again, change actual kernel call threads also
const intType_t POPULATION_SIZE{16};

const intType_t IN_SIZE{4096}; // in bits
const intType_t OUT_SIZE{4096}; // in bits

// IN_SIZE must be large enough for a full warp to load integers into memory.
// so it has to be at least 32*32
//const intType_t IN_SIZE{32*32}; // in bits
//const intType_t OUT_SIZE{32*32}; // in bits

// NOTE when allocating etc we still need to multiply this by BATCH_SIZE and POPULATION_SIZE when 
// appropriate
const intType_t IN_BYTES{IN_SIZE / 8};
const intType_t OUT_BYTES{OUT_SIZE / 8};

const intType_t IN_INTS{IN_BYTES / sizeof(myIntType_t)}; // TODO  can we use 64 bit ints?
const intType_t OUT_INTS{OUT_BYTES / sizeof(myIntType_t)};

// 32 * 32 = 1024, the max threads per block.
// a full warp (32 threads) is used to compute a single output
// haven't tested anything besides 1
const intType_t OUT_TILE_X_INTS{1};
// needs to stay at 32 to match warp size
const intType_t OUT_TILE_Y_INTS{32};

// Needs to stay at 32 to match warp size
const intType_t SUBTILE_INTS{32};

const intType_t WEIGHT_SIZE{POPULATION_SIZE * IN_SIZE * OUT_SIZE * 2}; // *2 since we have a weight for the input and for the inverted input
const intType_t WEIGHT_BYTES{WEIGHT_SIZE / 8};
const intType_t WEIGHT_INTS{WEIGHT_BYTES / sizeof(myIntType_t)};


// note that i is the index into the input as an array of integers, NOT as an array of bits.
__host__ __device__ intType_t i_ind(const intType_t b, const intType_t p, const intType_t i) {
 
  return b * POPULATION_SIZE * IN_INTS + p * IN_INTS + i;
}

// note that o is the index into output as an array of integers, NOT as an array of bits.
__host__ __device__ intType_t o_ind(const intType_t b, const intType_t p, const intType_t o) {
  return b * POPULATION_SIZE * OUT_INTS + p * OUT_INTS + o;
}

// note the "o" arg to w_ind should be different than the "o" arg to o_ind above.
// (because w has a full integer entry for each output bit o)
__host__ __device__ intType_t w_ind(const intType_t t, const intType_t p, const intType_t i, const intType_t o) {
  return t * IN_INTS * POPULATION_SIZE * OUT_SIZE + p * IN_INTS * OUT_SIZE + i * OUT_SIZE + o;
}


// Computes the binary matrix multiplication on the host and saves the output
// out should be zeroed
void host_op(myIntType_t const * const input, myIntType_t const * const weight, myIntType_t * out, int thresh) {

  for( intType_t b = 0; b < BATCH_SIZE; b++) {
    for( intType_t p = 0; p < POPULATION_SIZE; p++) {
      for( intType_t o = 0; o < OUT_SIZE; o++) {

        unsigned int temp = 0;

        for( intType_t i = 0; i < IN_INTS; i++) {

          //intType_t in = input[i_ind(b,p,i)];
          //intType_t w1 = weight[w_ind(0,p,i,o)];
          //intType_t w2 = weight[w_ind(1,p,i,o)];

          //intType_t first_and =   in & w1;
          //intType_t second_and =  (~in) & w2;

          //printf("Adding to temp 1 : " BBP " & " BBP"  =  " BBP "\b\n",BB(in), BB(w1), BB(first_and)); 
          //printf("popcount of "BBP": %u\n", BB(first_and), __builtin_popcount( first_and));

          temp += __builtin_popcount( (input[i_ind(b,p,i)] & weight[w_ind(0,p,i,o)]));

          //printf("Adding to temp 2 : " BBP " & " BBP"  =  " BBP "\b\n",BB(~in), BB(w2), BB( second_and)); 
          //printf("popcount of "BBP": %u\n", BB(second_and), __builtin_popcount( second_and));

          temp += __builtin_popcount((~input[i_ind(b,p,i)]) & weight[w_ind(1,p,i,o)]);
        }
        
        //printf("done adding to temp");

        //printf("activation for output %u: %u\n", o, temp);

        // NOTE that the way we shift here, the outputs are ordered low-precision bits first
        myIntType_t bit_to_set = (temp >= thresh) << (o % (8 * sizeof(myIntType_t)));
        //printf("bit_to_set: %d\n", bit_to_set);
        myIntType_t o_int_index = o / (8 * sizeof(myIntType_t));
        out[o_ind(b,p,o_int_index)] = out[o_ind(b,p,o_int_index)] | bit_to_set;

      }
    }
  }
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
  Indexing assumptions:
  TODO docs
*/
__global__ void gpu_op(
        myIntType_t const * const input, 
        myIntType_t const * const weight, 
        myIntType_t * out, 
        const unsigned int thresh, 
        const unsigned int out_tile_x_ints,
        const unsigned int out_tile_y_ints, 
        const unsigned int subtile_ints,
        const unsigned int in_ints, 
        const unsigned int in_bytes, 
        const unsigned int in_size,
        const unsigned int warp_size
        ) {

  // shared memory for input and for weights
  // shared memory needs to contain 
  // and window of image that we are convolving, of size
  // (WINDOW_X_SIZE, WINDOW_Y_SIZE)
  extern __shared__ myIntType_t shared[];

  myIntType_t* weight_tile{&shared[0]};
  // TODO replace 32 with a constant or something
  myIntType_t* not_weight_tile{&shared[out_tile_x_ints * subtile_ints * 32]};
  myIntType_t* input_tile{&shared[2 * out_tile_x_ints * subtile_ints * 32]};


  const intType_t output_tile_x = blockIdx.x * out_tile_x_ints;
  const intType_t output_tile_y = blockIdx.y * out_tile_y_ints;
  const intType_t b = output_tile_y + threadIdx.y;

  const intType_t p = blockIdx.z;
  //const intType_t o = output_tile_x + threadIdx.x;

  // TODO replace 32 with a constant or something
  



  //if (blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
  //  //printf("HERE BEFORE FOR IN GPU\n");
  //  //printf("IN_INTS: %u, IN_BYTES: %u, IN_SIZE: %u\n", in_ints, in_bytes, in_size);
  //        //printf("valid window sizes: %d, %d\n", valid_window_x_size, valid_window_y_size);
  //        //printf("tile_x, tile_y: %d, %d\n", tile_x, tile_y);
  //        //printf("tile sizes: %d, %d\n", tile_x_size, tile_y_size);
  //        //printf("threadIdx.x: %d\n", threadIdx.x);
  //}
  

  unsigned int acc = 0;

  for (unsigned int tile_i = 0; tile_i < in_ints / subtile_ints; ++tile_i) {

    intType_t i = tile_i * subtile_ints + threadIdx.x;

    // Load input into shared memory
    if (threadIdx.x < subtile_ints) { 
      //printf("%d.%d : i_ind: %d\n",threadIdx.x, threadIdx.y, i_ind(b, p, i));
      input_tile[threadIdx.y * subtile_ints + threadIdx.x] = input[i_ind(b, p, i)]; 
      //if (VERBOSE) {printf("loading i: %u\n", &input_tile[threadIdx.y * subtile_ints + threadIdx.x]);};
    }

    //if (VERBOSE) {printf("loading  w: %u\n", &weight_tile[threadIdx.y * subtile_ints + threadIdx.x]);};
    //if (VERBOSE) {printf("loading nw: %u\n", &not_weight_tile[threadIdx.y * subtile_ints + threadIdx.x]);};

    // load weight into shared memory
    intType_t w_i_to_load =  tile_i * subtile_ints + threadIdx.y;
    intType_t w_o_to_load = blockIdx.x * out_tile_x_ints * 32 + threadIdx.x;

    //if (VERBOSE) {printf("Setting weight_tile[%u] with weight index %u\n",     threadIdx.y * subtile_ints * 32 + j * 32 + threadIdx.x, w_ind(0, p, w_i_to_load, w_o_to_load));};

    //if (VERBOSE) {printf("Setting not_weight_tile[%u] with weight index %u\n",threadIdx.y * subtile_ints * 32 + j * 32 + threadIdx.x,w_ind(1, p, w_i_to_load, w_o_to_load) );};

    // "o" varies with threadIdx.x and is the last coord so access is coalesced
   
    weight_tile[    threadIdx.y * 32 +  threadIdx.x] = weight[w_ind(0, p, w_i_to_load, w_o_to_load)]; 
    not_weight_tile[threadIdx.y * 32 +  threadIdx.x] = weight[w_ind(1, p, w_i_to_load, w_o_to_load)]; 

    __syncthreads(); // wait until shared memory load is finished before continuing with compute


    for (unsigned int j = 0; j < subtile_ints; ++j) {

      if (VERBOSE) {printf("DONE %u %u", threadIdx.x, threadIdx.y);};

      myIntType_t x1 = input_tile[threadIdx.y * subtile_ints + j];

      //printf("x1: %u\n", x1);
      //printf("%u.%u j:%u: input_tile[...]: " BBP BBP BBP BBP "\n\n", threadIdx.x, threadIdx.y, j, BB(x1>>24), BB(x1>>16), BB(x1>>8), BB(x1));
      if (VERBOSE) {printf("j: %u i: " BBP BBP BBP BBP "\n", j, BB(x1 >> 24), BB(x1 >> 16), BB(x1>>8), BB(x1));};
      x1 = weight_tile[j * subtile_ints + threadIdx.x];
      //printf("%u.%u j:%u: weight_tile[...]: " BBP BBP BBP BBP "\n\n", threadIdx.x, threadIdx.y, j, BB(x1>>24), BB(x1>>16), BB(x1>>8), BB(x1));

      if (VERBOSE) {printf("j: %u w1: " BBP BBP BBP BBP "\n", j, BB(x1>>24), BB(x1>>16), BB(x1>>8), BB(x1));};
      x1 = not_weight_tile[j * subtile_ints + threadIdx.x];
      if (VERBOSE) {printf("j: %u w2: " BBP BBP BBP BBP "\n", j, BB(x1>>24), BB(x1>>16), BB(x1>>8), BB(x1));};

      acc += __popc(input_tile[threadIdx.y * subtile_ints + j] & weight_tile[j * subtile_ints + threadIdx.x]);
      acc += __popc((~input_tile[threadIdx.y * subtile_ints + j]) & not_weight_tile[j * subtile_ints + threadIdx.x]);
    }
    
    __syncthreads(); // wait until this compute is finished before next loop replaces shared memory

  }

  //if (blockIdx.x == 1) {printf("%u.%u: acc: %u\n",threadIdx.x, threadIdx.y, acc);};
  // activemask will always be 1111..., TODO 
  unsigned int r = __ballot_sync(__activemask(), acc >= thresh);
  // after ballot_sync, only one thread per warp needs to write to output
  if (threadIdx.x % warp_size == 0) {
    if (VERBOSE) {printf("r for blockIdx.x %u threadIdx.x %u threadIdx.y %u: " BBP "\n",blockIdx.x, threadIdx.x, threadIdx.y, BB(r));};
    //out[o_ind(b,p,o / sizeof(myIntType_t))] = r;
    out[o_ind(b,p,output_tile_x)] = r;
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
  fprintf(stdout, "sizeof(myIntType_t)%lu\n",sizeof(myIntType_t));

  // arrays for input, weight, and output on host and device
  myIntType_t *h_input, *h_weight, *h_gpu_out, *h_cpu_out;
  myIntType_t *d_input, *d_weight, *d_out;

  printf("HERE 1\n");

  // allocate space for image and convolutional filter
  cudaError_t status;
  status = cudaMallocHost((void**) &h_input, IN_BYTES * BATCH_SIZE * POPULATION_SIZE);
  if (status != cudaSuccess) {
    printf("Error allocating image memory.\n");
  }
  printf("HERE 2\n");

  status = cudaMallocHost((void**) &h_weight, WEIGHT_BYTES * POPULATION_SIZE);
  if (status != cudaSuccess) {
    printf("Error allocating weight memory.\n");
  }
  printf("HERE 3\n");


  // allocate space for output from both CPU and GPU
  printf("Allocating h_cpu_out with %u bytes.", OUT_BYTES * BATCH_SIZE * POPULATION_SIZE);
  h_cpu_out = (myIntType_t *) malloc(OUT_BYTES * BATCH_SIZE * POPULATION_SIZE);
  status = cudaMallocHost((void**) &h_gpu_out, OUT_BYTES * BATCH_SIZE * POPULATION_SIZE);
  if (status != cudaSuccess) {
    printf("Error allocating output memory.\n");
  }

  printf("HERE 4\n");

  // check if there was a error in host malloc
  if( h_input == NULL || h_weight == NULL || h_cpu_out == NULL ||
      h_gpu_out == NULL) {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }
  printf("HERE 5\n");

  // initialize output to 0 (since we will be computing it by adding to this
  // output
  memset(h_cpu_out, 0, OUT_BYTES * BATCH_SIZE * POPULATION_SIZE);
  memset( h_gpu_out, 0, OUT_BYTES * BATCH_SIZE * POPULATION_SIZE);

  printf("HERE 6\n");

  // randomly initialize the image and the filter
  // TODO verify this gives random bits

  //fprintf(stdout, "Randomly initializing input");
  for( intType_t i = 0; i < IN_INTS; ++i ) {
    h_input[i] = (myIntType_t) rand();
  }
  printf("HERE 7\n");
  //fprintf(stdout, "Randomly initializing weight");
  for(intType_t i = 0; i < WEIGHT_INTS; ++i ) {
    h_weight[i] = (myIntType_t) rand();
  }
  printf("HERE 8\n");


  // start timers
  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

  printf("HERE 9\n");

  /////////
  // GPU //
  /////////

  // allocate image, filter, and output in GPU memory
  checkCUDA( cudaMalloc( (void **)&d_input, IN_BYTES * BATCH_SIZE * POPULATION_SIZE ) );
  checkCUDA( cudaMalloc( (void **)&d_weight, WEIGHT_BYTES * POPULATION_SIZE ) );
  checkCUDA( cudaMalloc( (void **)&d_out, OUT_BYTES * BATCH_SIZE * POPULATION_SIZE) );

  // copy image and filter to device
  checkCUDA( cudaMemcpy( d_input, h_input, IN_BYTES, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemcpy( d_weight, h_weight, WEIGHT_BYTES, cudaMemcpyHostToDevice ) );

  // setup grid and block sizes
  //const dim3 threads(32, 32, 1); // max possible for me. one thread per entry in 32x32 tile 
  const dim3 threads(32, 32, 1); // set to 1 while we are testing batch size 1 TODO

  const dim3 blocks((OUT_INTS / OUT_TILE_X_INTS), (BATCH_SIZE / OUT_TILE_Y_INTS), POPULATION_SIZE);

  // *3 because there is the input tile, the weight tile, and the ~weight tile.
  const intType_t sharedMemSize = sizeof(myIntType_t) * ((SUBTILE_INTS * OUT_TILE_Y_INTS) + 2 * (SUBTILE_INTS * OUT_TILE_X_INTS * 32)); // TODO try to remove 32 and replace with the appropriate connstant

  // start timer
  printf("blocks.z: %d\n", blocks.z);
  printf("blocks.x: %d\n", blocks.x);
  printf("blocks.y: %d\n", blocks.y);
  printf("blocks.z: %d\n", blocks.z);
  printf("threads.x: %d\n", threads.x);
  printf("threads.y: %d\n", threads.y);
  printf("shmem size: %d\n", sharedMemSize);


  // initialize output to 0, since compute it by adding to it. 
  // TODO: is this better than just having each thread initialize its loc to 0?
  checkCUDA( cudaMemset( d_out, 0, OUT_BYTES * BATCH_SIZE * POPULATION_SIZE ) );

  checkCUDA( cudaEventRecord( start, 0 ) );

  gpu_op<<< blocks, threads, sharedMemSize >>> (
      d_input, 
      d_weight, 
      d_out, 
      IN_SIZE / 2,
      OUT_TILE_X_INTS,
      OUT_TILE_Y_INTS,
      SUBTILE_INTS,
      IN_INTS,
      IN_BYTES,
      IN_SIZE,
      WARP_SIZE);

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

        
  // copy output back to host to compare with CPU
  checkCUDA( cudaMemcpy( h_gpu_out, d_out, OUT_BYTES * POPULATION_SIZE * BATCH_SIZE, cudaMemcpyDeviceToHost ) );

  if (COMPARE_WITH_CPU) {
    // start timer
    checkCUDA( cudaEventRecord( start, 0 ) );

    printf("HERE 10\n");
  
    // do convolution on cpu
    host_op(h_input, h_weight, h_cpu_out, IN_SIZE / 2);
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
    float temp = 0.0;

    if (VERBOSE) {
      //for( intType_t i = 0; i < 16; i++)
      for( intType_t i = 0; i < IN_INTS; i++)
      {
        printf("h_input %d:", i); 
        printbinary(h_input[i]);
      }

      //for( intType_t i = 0; i < 64; i++)
      for( intType_t i = 0; i < WEIGHT_INTS; i++)
      {
        printf("weight  %d:", i);
        printbinary(h_weight[i]);
      }
    }


    //for( intType_t i = 0; i < 4; i++ )
    for( intType_t i = 0; i < OUT_INTS; i++ )
    {

      if (VERBOSE) {
        printf("output (cpu first) %d:", i);
        printbinary(h_cpu_out[i]);
        printbinary(h_gpu_out[i]);
      }
      temp += ( h_cpu_out[i] - h_gpu_out[i] ) * ( h_cpu_out[i] - h_gpu_out[i] );
    } /* end for */

    printf("error is %f\n",temp);
    if( temp > 0 ) printf("FAIL\n");
    else printf("PASS\n");
  }
    


  //int testx = 10;
  //int testy = 29;

  //printf("test index: %d, %d\n", testx, testy);
  //printf("cpu out: %f\n", h_cpu_out[testy * OUT_Y_SIZE + testx]);
  //printf("gpu out: %f\n", h_gpu_out[testy * OUT_Y_SIZE + testx]);


  // cleanup
  cudaFreeHost( h_input );
  cudaFreeHost( h_weight );
  cudaFreeHost( h_gpu_out );

  cudaFree(d_out);
  cudaFree(d_input);
  cudaFree(d_weight);

  free( h_cpu_out );


  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

  return 0;


}
