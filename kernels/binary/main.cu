
#include <tclap/CmdLine.h>
#include <bitset>
#include <limits>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"
#include "debug.h"
#include <time.h> // for srand init
#include "binary.cuh"
#include "cuda_profiler_api.h"
#define  torch_device_inttype torch::kInt64
#define  torch_output_inttype torch::kInt32

const bool COMPARE_THRESHOLD{true};
const bool COMPARE_NOTHRESHOLD{true};
const bool STRESS_TEST{false};
const bool TEST_CONSOLIDATE{true};
const bool TEST_NOTHRESH_CONSOLIDATE{true};

const unsigned int BATCH_SIZE{1}; // TODO should we make threads be max of batch_size and 32?
const unsigned int POPULATION_SIZE{2};
// IN_SIZE must be large enough for a full warp to load integers into memory.
// so it has to be at least 32*32 TODO is this still true?
const unsigned int IN_SIZE{64}; // in bits
const unsigned int OUT_SIZE{64}; // in bits

const int DEVICE_INTTYPE_BITS = binary_forward::DEVICE_INTTYPE_BYTES * 8;

typedef binary_forward::device_inttype device_inttype;
typedef binary_forward::output_inttype output_inttype;

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

void printOut(at::Tensor out, bool bits) {
  const unsigned int population_size = out.sizes()[0];
  const unsigned int batch_size = out.sizes()[1];
  const unsigned int out_size = out.sizes()[2];
  for (int p = 0; p < population_size; p++) {
    for (int b = 0; b < batch_size; b++) {
      for (int o = 0; o < out_size; o++) {
        if (bits) {
          std::cout << "p" << p << "b" << b << "o" << o << ":" << std::bitset<64>( out.index({p,b,o}).item<device_inttype>()) << std::endl;
        } else {

          std::cout << "p" << p << "b" << b << "o" << o << ":" << out.index({p,b,o}).item<device_inttype>() << std::endl;
        }
      }
    }
  }
}


int main( int argc, char *argv[] )
{
  //argparse::ArgumentParser program("program_name");
  //program.add_argument("-p", "--population").default_value(POPULATION_SIZE);
  //program.add_argument("-i", "--input").default_value(IN_SIZE);
  //program.add_argument("-o", "--output").default_value(OUT_SIZE);
  //program.parse_args(argc, argv);
  //input = program.get<int>("b");


  //unsigned int batch_size = program.get<int>("b");
  //unsigned int population_size = program.get<int>("p");
  //unsigned int in_size = program.get<int>("i");
  //unsigned int out_size = program.get<int>("o");

  TCLAP::CmdLine cmd("TODO descriptor", ' ', "0.1");
  TCLAP::ValueArg<int> batchArg("b","batch_size","Batch size",false,BATCH_SIZE,"int");
  TCLAP::ValueArg<int>   popArg("p","population_size","Population size",false,POPULATION_SIZE,"int");
  TCLAP::ValueArg<int>    inArg("i","in_size","Input size",false,IN_SIZE,"int");
  TCLAP::ValueArg<int>   outArg("o","out_size","Output size",false,OUT_SIZE,"int");

  TCLAP::SwitchArg   vArg("v","verbose","Verbose",false);
  cmd.add(batchArg);
  cmd.add(popArg);
  cmd.add(inArg);
  cmd.add(outArg);
  cmd.add(vArg);
  cmd.parse( argc, argv );




  unsigned int batch_size = batchArg.getValue();
  unsigned int population_size = popArg.getValue();
  unsigned int in_size = inArg.getValue();
  unsigned int out_size = outArg.getValue();
  bool verbose = vArg.getValue();

  //po::options_description desc("Allowed options");
  //desc.add_options()
  //    ("b", po::value<int>(), "Batch size")
  //    ("p", po::value<int>(), "Population size size")
  //    ("i", po::value<int>(), "Input size")
  //    ("o", po::value<int>(), "Output size")
  //;
  //
  //po::variables_map vm;
  //po::store(po::parse_command_line(argc, argv, desc), vm);
  //po::notify(vm);    
  //
  //if (vm.count("b")) {
  //    batch_size = vm["b"].as<int>();
  //} 

  //if (vm.count("p")) {
  //    population_size = vm["p"].as<int>();
  //} 

  unsigned int in_int64s = (in_size + 63) / 64; 


  srand(time(NULL));;
  /* get GPU device number and name */
  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

  fprintf(stdout, "Input size is %d\n",in_size);
  fprintf(stdout, "Batch size is %d\n",batch_size);
  fprintf(stdout, "Population size is %d\n",population_size);
  fprintf(stdout, "sizeof(int)%lu\n",sizeof(int));
  fprintf(stdout, "sizeof(device_inttype)%lu\n",sizeof(device_inttype));
  
  fprintf(stdout, "X multiplicity is %d\n", binary_forward::OUT_TILE_X_MULTIPLICITY);
  fprintf(stdout, "Y multiplicity is %d\n", binary_forward::OUT_TILE_Y_MULTIPLICITY);
  std::cout << "Verbose is: " << verbose << std::endl;

  auto options =
      torch::TensorOptions()
        .dtype(torch_device_inttype)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);


  const device_inttype maxlong = std::numeric_limits<device_inttype>::max();
  const device_inttype minlong = std::numeric_limits<device_inttype>::min();
  printf("HERE A\n");


  const at::Tensor h_input = torch::randint(minlong,maxlong,{population_size,batch_size,in_int64s},options);
  const at::Tensor h_weight = torch::randint(minlong,maxlong,{population_size,in_int64s,2,out_size},options);

  printf("HERE B\n");
  const at::Tensor d_input = h_input.to(torch::kCUDA);
  const at::Tensor d_weight = h_weight.to(torch::kCUDA);

  printf("HERE C\n");

  // Note this is different than IN_SIZE / 2 because when IN_SIZE is not divisible
  // by DEVICE_INTTYPE_BITS, we add extend IN_SIZE until it is. 
  const int64_t threshold = in_int64s * 64 / 2;

  printf("HERE D\n");

  // start timers
  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

  printf("HERE E\n");

  float elapsedTime;

  /////////
  // GPU //
  /////////
  printf("HERE F");

  if (TEST_CONSOLIDATE) {
    printf("YAY HERE 1\n");

    auto options =
        torch::TensorOptions()
          .dtype(torch_device_inttype)
          .layout(torch::kStrided)
          .device(torch::kCPU)
          .requires_grad(false);
    printf("YAY HERE 2\n");
    const at::Tensor consolidate_h_input1 = torch::randint(-256,256,{2,4096,128},options);
    printf("Made h input1\n");
    const at::Tensor consolidate_d_input1 = consolidate_h_input1.to(torch::kCUDA);
    printf("Made d input1\n");

    //checkCUDA( cudaEventRecord( start, 0 ) );

    at::Tensor h_consolidated = binary_forward::host_consolidate(consolidate_h_input1);
    printf("computed h_consolidated\n");
    //printInput(h_consolidated);

    at::Tensor d_consolidated = binary_forward::consolidate_bits_cuda(consolidate_d_input1);
    //checkKERNEL();
    printf("computed d_consolidated\n");

    //checkCUDA( cudaEventRecord( stop, 0 ) );
    //checkCUDA( cudaEventSynchronize( stop ) );

    //checkCUDA( cudaEventDestroy( start ) );
    //checkCUDA( cudaEventDestroy( stop ) );
    //checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    int64_t diff3 = torch::sum(torch::abs(h_consolidated - d_consolidated.to(torch::kCPU))).item<device_inttype>();
    printf("Consolidation error is %ld\n",diff3);
  }

  if (TEST_NOTHRESH_CONSOLIDATE) {
    // TODO 
    auto options =
        torch::TensorOptions()
          .dtype(torch_device_inttype)
          .layout(torch::kStrided)
          .device(torch::kCUDA)
          .requires_grad(false);

    const int64_t threshold = (13 * 64) / 2;
    at::Tensor i = torch::randint(minlong,maxlong,{2,4096,13},options);
    at::Tensor w = torch::randint(minlong,maxlong,{2,13,2,128},options);

    at::Tensor full_output = binary_forward::binary_forward_cuda(i, w, threshold, false);

    at::Tensor step_output1 = binary_forward::binary_forward_cuda(i,w,0,false) - threshold;
    at::Tensor step_output2 = binary_forward::consolidate_bits_cuda(step_output1);

    int64_t diff = torch::sum(torch::abs(full_output - step_output2)).item<device_inttype>();
    printf("Nothresh + consolidate error is %ld\n",diff);

  }

  if (STRESS_TEST) {
    at::Tensor i1;
    at::Tensor w1;
    at::Tensor w2;
    at::Tensor i2;
    at::Tensor r;

    //cudaProfilerStart();
    
 
    for (int i = 0; i < 2; i++) {
      auto options =
          torch::TensorOptions()
            .dtype(torch_device_inttype)
            .layout(torch::kStrided)
            .device(torch::kCUDA)
            .requires_grad(false);
      i1 = torch::randint(minlong,maxlong,{16,4096,64},options);
      w1 = torch::randint(minlong,maxlong,{16,64,2,4096},options);

      i2 = binary_forward::binary_forward_cuda(
          i1,
          w1,
          (64*64)/2,
          false);

      //w2 = torch::randint(minlong,maxlong,{2,1024,2,10},options);

      //r = binary_forward::binary_forward_cuda(
      //    i2,
      //    w2,
      //    0,
      //    false);
      //printOut(r.index({torch::indexing::Slice(-1,torch::indexing::None),torch::indexing::Slice(-1,torch::indexing::None),torch::indexing::Slice()}), false);
      //std::cout << i << std::endl;
    }

    //cudaProfilerStop();

  }



       

  checkCUDA( cudaEventRecord( start, 0 ) );

  at::Tensor d_out_thresh = binary_forward::binary_forward_cuda(
    d_input,
    d_weight,
    threshold,
    verbose);

  at::Tensor d_out_nothresh = binary_forward::binary_forward_cuda(
    d_input,
    d_weight,
    0,
    verbose);

  checkKERNEL();

  // stop timer and print time
  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GBop/s\n", ( ( (double) BATCH_SIZE *
    (double) POPULATION_SIZE * 
    (double) OUT_SIZE * (double) IN_SIZE * 2.0 / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 ))); // TODO check this computation , havent checked

  if (verbose) {
    std::cout << "Input:" << std::endl;
    printInput(h_input);
    //printf("Device weight:\n");
    //printWeight(d_weight.to(torch::kCPU));
    std::cout << "Device out:" << std::endl;
    printOut(d_out_thresh.to(torch::kCPU), true);
    std::cout << "Device NOTHRESH out:" << std::endl;
    printOut(d_out_nothresh.to(torch::kCPU), false);
  }

  if (COMPARE_THRESHOLD) {
    // start timer
    checkCUDA( cudaEventRecord( start, 0 ) );
  
    // do convolution on cpu
    at::Tensor h_out_thresh = binary_forward::host_helper(h_input, h_weight, threshold, verbose);
    //at::Tensor h_out_nothresh = binary_forward::host_helper(h_input, h_weight, threshold);
  
    // stop timers
    checkCUDA( cudaEventRecord( stop, 0 ) );
    checkCUDA( cudaEventSynchronize( stop ) );
    checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  
    // print time taken
    fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
    checkCUDA( cudaEventDestroy( start ) );
    checkCUDA( cudaEventDestroy( stop ) );
  
    // compare GPU implementation results with CPU results

    int64_t diff1 = torch::sum(torch::abs(h_out_thresh - d_out_thresh.to(torch::kCPU))).item<device_inttype>();
    printf("Threshold error is %ld\n",diff1);

    if (verbose) {

      //printf("Weight:\n");
      //printWeight(h_weight);
      std::cout << "Host out:" << std::endl << h_out_thresh << std::endl;
      printOut(h_out_thresh, true);
    }
    
    printf("Threshold error is %ld\n",diff1);

    if( diff1 == 0 ) {
      printf("PASS\n");
    } else {
      printf("FAIL\n");
    }

  }

  if (COMPARE_NOTHRESHOLD) {
    checkCUDA( cudaEventRecord( start, 0 ) );

    at::Tensor h_out_nothresh = binary_forward::host_helper(h_input, h_weight, 0, verbose);
    checkCUDA( cudaEventRecord( stop, 0 ) );
    checkCUDA( cudaEventSynchronize( stop ) );

    // print time taken
    fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
    checkCUDA( cudaEventDestroy( start ) );
    checkCUDA( cudaEventDestroy( stop ) );
    checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    int64_t diff2 = torch::sum(torch::abs(h_out_nothresh - d_out_nothresh.to(torch::kCPU))).item<device_inttype>();
    printf("No threshold error is %ld\n",diff2);

    if (verbose) {
      std::cout << "NOTHRESH Host out:" << std::endl;
      printOut(h_out_nothresh, false);
    }
    printf("No threshold error is %ld\n",diff2);

    if( diff2 == 0 ) {
      printf("PASS\n");
    } else {
      printf("FAIL\n");
    }
  }

  printf("HERE 1");

    


    

  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

  return 0;
}
