#include "utils_sched.h"

cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
cudaError_t (*memcpy_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t (*memcpy_async_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t (*malloc_function)(void** devPtr, size_t size);
cudaError_t (*free_function)(void* devPtr);
cudaError_t (*memset_function)(void* devPtr, int  value, size_t count);
cudaError_t (*memset_async_function)(void* devPtr, int  value, size_t count, cudaStream_t stream);


cublasStatus_t (*cublas_sgemm_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
cublasStatus_t (*cublas_sgemm_strided_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);


extern int status;
extern int* seen;
extern int* num_client_kernels;

void process_eval(vector<vector<float>> &client_durations) {

	int num_clients = client_durations.size();
	for (int i=0; i<num_clients; i++) {
		vector<float> client_stats = client_durations[i];
		// remove first two iters
		client_stats.erase(client_stats.begin());
		client_stats.erase(client_stats.begin());

		sort(client_stats.begin(), client_stats.end());
		int client_len = client_stats.size();
		float p50 = client_stats[client_len/2];
		float p95 = client_stats[(client_len*95/100)];
		float p99 = client_stats[(client_len*99/100)];
		printf("Client %d, p50=%f, p95=%f, p99=%f\n", i, p50, p95, p99);
	}
}


void pop_from_queue(queue<struct func_record>* client_queue, pthread_mutex_t* client_mutex, int idx) {
	pthread_mutex_lock(client_mutex);
	client_queue->pop();
	pthread_mutex_unlock(client_mutex);
}

void create_streams(cudaStream_t** sched_streams, int num, bool reef) {

	int* lp = (int*)malloc(sizeof(int));
	int* hp = (int*)malloc(sizeof(int));

	CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(lp, hp));

	DEBUG_PRINT("Highest stream priority is %d, lowest stream priority is %d\n", *hp, *lp);
	assert(*lp==0);

	for (int i=0; i<num-1; i++) {
		sched_streams[i] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[i], cudaStreamNonBlocking, 0);
	}

	// client num-1 is high priority
	if (!reef) {
		sched_streams[num-1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[num-1], cudaStreamNonBlocking, *hp);
	}
	else {
		sched_streams[num-1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[num-1], cudaStreamNonBlocking, 0);
	}

}

void create_events(cudaEvent_t*** events, int num) {

	// per-stream event
	for (int i=0; i<num; i++) {
		events[i] = (cudaEvent_t**)malloc(30000*sizeof(cudaEvent_t*));
		for (int j=0; j<30000; j++) {
			//printf("create %d, %d\n", i, j);
			events[i][j] = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
			CHECK_CUDA_ERROR(cudaEventCreateWithFlags(events[i][j], cudaEventDisableTiming));
		}
	}
}

void register_functions() {

    // for kernel
	*(void **)(&kernel_function) = dlsym(RTLD_DEFAULT, "cudaLaunchKernel");
	assert(kernel_function != NULL);

	// for memcpy
	*(void **)(&memcpy_function) = dlsym (RTLD_DEFAULT, "cudaMemcpy");
	assert(memcpy_function != NULL);

	// for memcpy_async
	*(void **)(&memcpy_async_function) = dlsym (RTLD_DEFAULT, "cudaMemcpyAsync");
	assert(memcpy_async_function != NULL);

	// for malloc
	*(void **)(&malloc_function) = dlsym (RTLD_DEFAULT, "cudaMalloc");
	assert(malloc_function != NULL);

	// for free
	*(void **)(&free_function) = dlsym (RTLD_DEFAULT, "cudaFree");
	assert(free_function != NULL);

	// for memset
	*(void **)(&memset_function) = dlsym (RTLD_DEFAULT, "cudaMemset");
	assert (memset_function != NULL);

	// for memset_async
	*(void **)(&memset_async_function) = dlsym (RTLD_DEFAULT, "cudaMemsetAsync");
	assert (memset_async_function != NULL);

	// CUBLAS sgemm
	*(void **)(&cublas_sgemm_function) = dlsym(RTLD_DEFAULT, "cublasSgemm_v2");
	assert(cublas_sgemm_function != NULL);

	// CUBLAS sgemm strided
	*(void **)(&cublas_sgemm_strided_function) = dlsym(RTLD_DEFAULT, "cublasSgemmStridedBatched");
	assert(&cublas_sgemm_strided_function != NULL);

}

void schedule_kernel(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid) {

	switch (frecord.type) {
		case KERNEL_RECORD: {
			DEBUG_PRINT("found a new kernel record from idx %d! kernel func is %p\n", idx, kernel_function);
			kernel_record record = frecord.data.krecord;
			(*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, *sched_stream);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case MEMCPY_RECORD: {
			memcpy_record record = frecord.data.mrecord;
			if (!record.async) {
				(*memcpy_function)(record.dst, record.src, record.count, record.kind);
			} else {
				(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, *sched_stream);
			}
			break;
		}
		case MALLOC_RECORD: {
			DEBUG_PRINT("found a new malloc record from idx %d!\n", idx);
			malloc_record record = frecord.data.malrecord;
			(*malloc_function)(record.devPtr, record.size);
			break;
		}
		case FREE_RECORD: {
			DEBUG_PRINT("found a new FREE record from idx %d!\n", idx);
			free_record record = frecord.data.frecord;
			//(*free_function)(record.devPtr);
			break;
		}
		case MEMSET_RECORD: {
			memset_record record = frecord.data.msetrecord;
			if (record.async) {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_async_function)(record.devPtr, record.value, record.count, *sched_stream);
			}
			else {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_function)(record.devPtr, record.value, record.count);
			}
			break;
		}
		case CUBLAS_SGEMM_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm record from idx %d!\n", idx);

			cublasSgemm_record record = frecord.data.cublasSgemmRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			(*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_SGEMM_STRIDED_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm strided record from idx %d!\n", idx);

			cublasSgemmStridedBatched_record record = frecord.data.cublasSgemmStridedRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			(*cublas_sgemm_strided_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.strideA, record.B, record.ldb, record.strideB, record.beta, record.C, record.ldc, record.strideC, record.batchCount);
			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		default:
			printf("UNSUPPORTED OPERATION - ABORT\n");
			abort();

	}
	DEBUG_PRINT("Return from schedule, seen[%d] is %d!\n", idx, seen[idx]);
	CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
	//event_ids[evid] += 1;
}