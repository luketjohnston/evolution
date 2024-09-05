BIN=x.conv
NVCC=nvcc
# the ept-relaxed-constexpr disables some warnings that I think are coming from torch libraries... 
NVOPTS=-O3 $(ARCH) -DDEBUG -lineinfo -lcudnn -I "/home/luke/.local/lib/python3.8/site-packages/torch/include" -I "/home/luke/.local/lib/python3.8/site-packages/torch/include/torch/csrc/api/include" -I "/usr/include/python3.8" --expt-relaxed-constexpr
#NVOPTS=-O3 $(ARCH) -lineinfo

# used to have lcublas, dont think I need it anymore
#$(NVCC) $(NVOPTS) -lcublas kernel.cu -o $(BIN) 

$(BIN): 
	$(NVCC) $(NVOPTS) kernel.cu -o $(BIN) 

clean:
	rm -rf kernel.o $(BIN)
