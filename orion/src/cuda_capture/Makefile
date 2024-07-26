CC=g++
CUDAINCLUDE=/usr/local/cuda-11.8/include/
CUDALIB=/usr/local/cuda-11.8/lib64

# CUBLASINCLUDE=/usr/local/cuda-11.8/targets/x86_64-linux/include/
# CUBLASLIB=/usr/local/cuda-11.8/targets/x86_64-linux/lib/

libinttemp.so: utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp
	$(CC) -O3 -fPIC -shared utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -ldl -L/usr/local/cuda/lib64 -lcublas -lcudart -lpthread  -o libinttemp.so

all:
	make libinttemp.so

clean:
	rm -rf *.o libinttemp.so
