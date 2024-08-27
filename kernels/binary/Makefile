BIN=x.conv
NVCC=nvcc
NVOPTS=-O3 $(ARCH) -DDEBUG -lineinfo -lcudnn
#NVOPTS=-O3 $(ARCH) -lineinfo

# used to have lcublas, dont think I need it anymore
#$(NVCC) $(NVOPTS) -lcublas kernel.cu -o $(BIN) 

$(BIN): 
	$(NVCC) $(NVOPTS) kernel.cu -o $(BIN) 

clean:
	rm -rf kernel.o $(BIN)
