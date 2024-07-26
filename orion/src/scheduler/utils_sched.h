#include <cuda_runtime.h>
#include <cuda.h>

#include "../system_utils.h"
#include "../cuda_capture/intercept_temp.h"

extern cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
extern cudaError_t (*memcpy_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t (*memcpy_async_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t (*malloc_function)(void** devPtr, size_t size);
extern cudaError_t (*free_function)(void* devPtr);
extern cudaError_t (*memset_function)(void* devPtr, int  value, size_t count);
extern cudaError_t (*memset_async_function)(void* devPtr, int  value, size_t count, cudaStream_t stream);

extern cublasStatus_t (*cublas_sgemm_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
extern cublasStatus_t (*cublas_sgemm_strided_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);


typedef struct op_info {

	string name;
	int profile; // 1: compute-bound, 0: mem-bound, -1: unclear
	int mem;
	int sm_used;
	float duration;

} op_info;

void register_functions();
void schedule_kernel(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid);
void schedule_pair(
	vector<func_record*> &frecords,
	queue<struct func_record>** &buffers,
	pthread_mutex_t** &mutexes,
	vector<vector<op_info>> &op_info_vector,
	int* seen, int max_sms,
	cudaStream_t** sched_streams,
	int* streams,
	cudaEvent_t*** events,
	int num_events,
	int* event_ids
);
void schedule_pair_kernel_padding(
	vector<func_record*> &frecords,
	queue<struct func_record>** &cbuffers,
	pthread_mutex_t** &cmutexes,
	vector<vector<op_info>> &op_info_vector,
	int* seen, int max_sms,
	cudaStream_t** sched_streams,
	int* streams,
	cudaEvent_t*** events,
	int num_events,
	int* event_ids
);
void pop_from_queue(queue<struct func_record>* client_queue, pthread_mutex_t* client_mutex, int idx);
void create_streams(cudaStream_t** sched_streams, int num, bool reef);
void create_events(cudaEvent_t*** events, int num);
void wait_for_stream(int idx, int profile, int current_prio, int prev_prio, cudaStream_t* sched_stream, cudaEvent_t*** events, int num_events, int* event_ids);
void wait_all_streams(int idx, cudaStream_t* sched_stream, cudaEvent_t*** events, int num_events, int* event_ids);
void process_eval(vector<vector<float>> &client_durations);