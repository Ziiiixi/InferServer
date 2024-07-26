CC=g++
NVCC=/usr/local/cuda-11.8/bin/nvcc
CFLAGS=-O3
CUDAINCLUDE=/usr/local/cuda-11.8/include/
CUDALIB=/usr/local/cuda-11.8/lib64
CUDNNINCLUDE=/usr/include/
CUDNNLIB=/usr/lib/x86_64-linux-gnu/

utils_sched.o: utils_sched.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC -x cu -shared -c utils_sched.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler.o: scheduler.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC  -x cu -shared -c scheduler.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler_eval.o: scheduler_eval.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC  -x cu -shared -c scheduler_eval.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler_eval.so: scheduler_eval.o utils_sched.o
	 $(CC) $(CFLAGS) -fPIC -shared utils_sched.o scheduler_eval.o -I$(CUDAINCLUDE) -L$(CUDALIB) -L$(CUDNNLIB) -lcudart -lpthread -lcublas -o scheduler_eval.so
all:
	make scheduler_eval.so

clean:
	rm -rf *.o *.so
