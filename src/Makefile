NVCC = nvcc
NVCCFLAGS = -G -g --compiler-options '-fPIC'
OBJS = add_round_key.o \
	cipher.o \
	helloworld.o \
	key_expansion.o \
	main.o \
	mix_columns.o \
	shift_rows.o \
	sub_bytes.o \
	utils.o \
    cuda_aes.o \
    cuda_aes_wrap.o
BINS = au9aes
LIBS = _cuda_aes_for_py.so
all: $(LIBS) $(BINS)

# Pack it as binary executable
au9aes: add_round_key.o 		\
	cipher.o 					\
	key_expansion.o		\
	mix_columns.o 						\
	shift_rows.o						\
	sub_bytes.o							\
	utils.o	\
    main.o
	gcc -o $@ $^ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda

# pack it as as shared library
# libcuda_aes.so: add_round_key.o 		\
# 	cipher.o 					\
# 	key_expansion.o		\
# 	mix_columns.o 						\
# 	shift_rows.o						\
# 	sub_bytes.o							\
# 	utils.o						
# 	$(NVCC) --shared -o $@ $^

# Compilation
add_round_key.o: add_round_key.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
cipher.o: cipher.cu cipher.h key_expansion.h aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
key_expansion.o: key_expansion.cu key_expansion.h
	$(NVCC) $(NVCCFLAGS) -c $<
main.o: main.c cipher.h utils.h
	gcc -g -fPIC -I/usr/include -c $<
mix_columns.o: mix_columns.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
shift_rows.o: shift_rows.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
sub_bytes.o: sub_bytes.cu aes.h
	$(NVCC) $(NVCCFLAGS) -c $<
utils.o: utils.c utils.h
	gcc -g -fPIC -c $<

# SWIG Python module
cuda_aes_for_py_wrap.c: cuda_aes_for_py.i
	swig -python $<
# cuda_aes_for_py_wrap.cxx: cuda_aes_for_py.i
# 	swig -c++ -python $<
# cuda_aes_for_py.o: cuda_aes_for_py.c
# 	gcc -g -fPIC -c $<
cuda_aes_for_py.o: cuda_aes_for_py.c
	gcc -g -fPIC -I/usr/include -c $<
cuda_aes_for_py_wrap.o: cuda_aes_for_py_wrap.c
	gcc -g -fPIC -I/usr/include/python2.7 -c $^
# cuda_aes_for_py_wrap.o: cuda_aes_for_py_wrap.cxx
# 	$(NVCC) $(NVCCFLAGS) -I/usr/include/python2.7 -c $^
_cuda_aes_for_py.so: add_round_key.o 		\
	cipher.o 					\
	key_expansion.o		\
	mix_columns.o 						\
	shift_rows.o						\
	sub_bytes.o							\
	utils.o	\
    cuda_aes_for_py.o \
    cuda_aes_for_py_wrap.o					
	gcc --shared -o $@ $^ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda

clean:
	rm -vf $(OBJS) $(BINS) *_wrap.c *.so *.a *.py *.pyc gdb.txt
