NVCC = nvcc
NVCCFLAGS = -G -g
OBJS = add_round_key.o \
	cipher.o \
	helloworld.o \
	key_expansion.o \
	main.o \
	mix_columns.o \
	shift_rows.o \
	sub_bytes.o \
	utils.o
BINS = au9aes

all: $(BINS)

au9aes: add_round_key.o \
	cipher.o \
	key_expansion.o \
	main.o \
	mix_columns.o \
	shift_rows.o \
	sub_bytes.o \
	utils.o
	$(NVCC) -o $@ $^

add_round_key.o: add_round_key.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
cipher.o: cipher.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
key_expansion.o: key_expansion.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
main.o: main.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
mix_columns.o: mix_columns.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
shift_rows.o: shift_rows.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
sub_bytes.o: sub_bytes.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
utils.o: utils.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm -vf $(OBJS) $(BINS) gdb.txt
