#include <dlfcn.h>
#include <stdio.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include <queue>
#include <string>
#include <pthread.h>
#include <cuda.h>
#include <cublas.h>
// #include <cublas_v2.h>
#include <assert.h>
#include <vector>
#include <sys/types.h>
#include <syscall.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "../system_utils.h"

typedef struct kernel_record {

	const void* func;
	dim3 gridDim;
	dim3 blockDim;
	void** args;
	size_t sharedMem;
	cudaStream_t stream;
	volatile bool run;
	volatile cudaStream_t sched_stream;
} kernel_record;


typedef struct memcpy_record {

	void* dst;
	const void* src;
	size_t count;
	enum cudaMemcpyKind kind;
	cudaStream_t stream;
	bool async;
} memcpy_record;


typedef struct malloc_record {

	void** devPtr;
    size_t size;

} malloc_record;

typedef struct free_record {
	void* devPtr;

} free_record;

typedef struct memset_record {

	void* devPtr;
	int value;
	size_t count;
	cudaStream_t stream;
	bool async;

} memset_record;

// CUBLAS

typedef struct cublasSgemm_record {

	cublasHandle_t handle;
	cublasOperation_t transa;
	cublasOperation_t transb;
	int m;
	int n;
	int k;
	const float *alpha;
	const float  *A;
       	int lda;
	const float *B;
	int ldb;
	const float *beta;
	float *C;
	int ldc;

	cublasSgemm_record(cublasHandle_t handle_arg, cublasOperation_t transa_arg, cublasOperation_t transb_arg, int m_arg, int n_arg, int k_arg, const float *alpha_arg, const float *A_arg, int lda_arg, const float *B_arg, int ldb_arg, const float *beta_arg, float *C_arg, int ldc_arg) {

		handle = handle_arg;
		transa = transa_arg;
		transb = transb_arg;
		m = m_arg;
		n = n_arg;
		k = k_arg;
		alpha = alpha_arg;
		A = A_arg;
		lda = lda_arg;
		B = B_arg;
		ldb = ldb_arg;
		beta = beta_arg;
		C = C_arg;
		ldc = ldc_arg;
	}

	~cublasSgemm_record() {}

} cublasSgemm_record;


typedef struct cublasSgemmStridedBatched_record {

	cublasHandle_t handle;
	cublasOperation_t transa;
	cublasOperation_t transb;
	int m;
	int n;
	int k;
	const float *alpha;
       	const float *A;
       	int lda;
       	long long int strideA;
       	const float *B;
       	int ldb;
       	long long int strideB;
       	const float *beta;
       	float *C;
       	int ldc;
       	long long int strideC;
       	int batchCount;

	cublasSgemmStridedBatched_record(cublasHandle_t handle_arg, cublasOperation_t transa_arg, cublasOperation_t transb_arg, int m_arg, int n_arg, int k_arg, const float *alpha_arg, const float *A_arg, int lda_arg, long long int strideA_arg, const float *B_arg, int ldb_arg, long long int strideB_arg, const float *beta_arg, float *C_arg, int ldc_arg, long long int strideC_arg, int batchCount_arg) {

		handle = handle_arg;
		transa = transa_arg;
		transb = transb_arg;
		m = m_arg;
		n = n_arg;
		k = k_arg;
		alpha = alpha_arg;
		A = A_arg;
		lda = lda_arg;
		strideA = strideA_arg;
		B = B_arg;
		ldb = ldb_arg;
		strideB = strideB_arg;
		beta = beta_arg;
		C = C_arg;
		ldc = ldc_arg;
		strideC = strideC_arg;
		batchCount = batchCount_arg;
	}


	~cublasSgemmStridedBatched_record() {}


} cublasSgemmStridedBatched_record;

//////////////////////////////////////////////////


enum func_type {
	KERNEL_RECORD,
	MEMCPY_RECORD,
	MALLOC_RECORD,
	FREE_RECORD,
	MEMSET_RECORD,
	CUBLAS_SGEMM_RECORD,
	CUBLAS_SGEMM_STRIDED_RECORD
};

union func_data {

	kernel_record krecord;
	cublasSgemm_record cublasSgemmRecord;
	cublasSgemmStridedBatched_record cublasSgemmStridedRecord;
	memcpy_record mrecord;
	malloc_record malrecord;
	free_record frecord;
	memset_record msetrecord;

	 func_data() {}
	 ~func_data() {};
};

typedef struct func_record {

	enum func_type type;
	union func_data data;

} func_record;


// globals
extern volatile pid_t* thread_ids; // 2*N threads + scheduler

extern queue<func_record>** kqueues;
extern pthread_mutex_t** mutexes;

extern int* func_indexes;
extern int* num_total_clients;

extern volatile bool** client_request_status;
extern volatile bool* client_stop;
extern volatile bool* client_stop_ack;
extern volatile bool* affinity_set;

// functions
extern cudaError_t (*kernel_func)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
extern cudaError_t (*memcpy_func)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t (*memcpy_async_func)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t (*malloc_func)(void** devPtr, size_t size);
extern cudaError_t (*free_func)(void* devPtr);
extern cudaError_t (*memset_func)(void* devPtr, int  value, size_t count);
extern cudaError_t (*memset_async_func)(void* devPtr, int  value, size_t count, cudaStream_t stream);

extern cublasStatus_t (*cublas_sgemm_func)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
extern cublasStatus_t (*cublas_sgemm_strided_func)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);


// util functions
int get_idx();
void block(int idx, pthread_mutex_t** mutexes, queue<func_record>** kqueues);


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
		           const int line) {
	if (err != cudaSuccess)
	{
		printf("CUDA Runtime Error at: %s:%d\n", file, line);
		printf("Error %d, %s\n", err, cudaGetErrorString(err));
	}
	assert (err == cudaSuccess);
}
