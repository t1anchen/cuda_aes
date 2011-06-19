NVCC = nvcc
OBJS = helloworld.o aes.o main.o
BINS = helloworld aes

all: $(BINS)

helloworld: helloworld.o
	$(NVCC) -o $@ $^
helloworld.o: helloworld.cu
	$(NVCC) -c -o $@ $^

aes: main.o aes.o
	$(NVCC) -o $@ $^
main.o: main.cu aes.h
	$(NVCC) -c $< -o $@
aes.o: aes.cu aes.h
	$(NVCC) -c $< -o $@

clean:
	rm -vf $(OBJS) $(BINS)
