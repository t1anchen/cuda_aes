NVCC = nvcc
NVCCFLAGS = -G -g -deviceemu
OBJS = helloworld.o aes.o main.o
BINS = helloworld au9aes

all: $(BINS)

helloworld: helloworld.o
	$(NVCC) -o $@ $^
helloworld.o: helloworld.cu
	$(NVCC) -c -o $@ $^

au9aes: main.o aes.o
	$(NVCC) -o $@ $^
main.o: main.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
aes.o: aes.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -vf $(OBJS) $(BINS)
