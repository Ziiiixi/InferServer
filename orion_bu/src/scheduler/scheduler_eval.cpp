#include "scheduler.h"
#include <vector>
#include <algorithm>
#include <cmath>
#define MIN_DURATION 100000 // might need to change this - emperical

using namespace std;
int priority_client = 1;
// globals
void* klib;
vector<vector<op_info>> op_info_vector;
int* fidx;
int* num_client_kernels;
int* num_client_max_iters;
int* num_client_cur_iters;

bool* locked;

std::chrono::time_point<std::chrono::high_resolution_clock>* client_starts;
std::chrono::time_point<std::chrono::high_resolution_clock>* total_client_starts;
bool** client_starts_set;
vector<vector<float>> client_durations;
int num_tpcs = 24;
int max_sms = 48; // v100
queue<struct func_record>** client_buffers;
pthread_mutex_t** client_mutexes;
queue<struct func_record>** buffers;
int* seen;
vector<int> client_progress;
vector<int> func_progress;
// cudnnHandle_t* global_handle0;
// cudnnHandle_t* global_handle1;

// fifo-globals
cudaStream_t sched_stream;
cudaStream_t sync_stream;
cudaEvent_t sched_event;

// profile-globals
cudaStream_t** sched_streams;
cudaStream_t** sync_streams;
cudaEvent_t*** events;
int* streams;
int* event_ids;
int status;
vector<int> max_sms_clients;
vector<bool> is_train;

// reef
int lp_idx = 0;
int penalty = 0;
bool** request_status;
bool* stops;
bool* stop_ack;

//spacial

vector<int> num_cur_clients;
bool *client_finished;
uint32_t mask;
uint32_t *localMask;
uint32_t *localMask_O;
vector<int> client_mask;
vector<int> client_block;
std::vector<std::vector<std::pair<int, int>>> current_schedule;
queue<int> q_client;
// std::vector<std::atomic<bool>> is_executing(5);
// bool *num_cur_clients_init = false;
std::vector<bool> is_executing;

void Scheduler::profile_reset(int num_clients) {

	for (int i=0; i<num_clients; i++) {
		seen[i] = 0;
		streams[i] = -1;
		fidx[i] = 0;
	}
}

void Scheduler::profile_prep(queue<func_record>** qbuffers, int num_clients, bool reef) {

	register_functions();
	client_buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		client_buffers[i] = (queue<struct func_record>*)(qbuffers[i]);

	int num = num_clients;

	sched_streams = (cudaStream_t**)malloc((num)*sizeof(cudaStream_t*));
	sync_streams = (cudaStream_t**)malloc((num)*sizeof(cudaStream_t*));

	for (int i=0; i<num; i++){
		sched_streams[i] = NULL;
		sync_streams[i] = NULL;
	}

	events = (cudaEvent_t***)malloc((num)*sizeof(cudaEvent_t**));
	for (int i=0; i<num; i++)
		events[i] = NULL;

	create_streams(sched_streams, num, reef);
	create_streams(sync_streams, num, reef);
	create_events(events, num);

	

	seen = (int*)calloc(num,sizeof(int));
	event_ids = (int*)calloc(num, sizeof(int));
	localMask = (uint32_t*)calloc(num,sizeof(uint32_t));
	localMask_O= (uint32_t*)calloc(num,sizeof(uint32_t));
	streams = (int*)malloc(num_clients*sizeof(int));
	for (int i=0; i<num_clients; i++)
		streams[i] = -1;

	sched_stream = 0;

	status = -1;

}

void Scheduler::execute_kernel(int client_id, struct func_record frecord) {

	sync_barrier.wait();

    schedule_kernel(frecord, sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
    pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
	is_executing[client_id] = false;
}

void Scheduler::execute_kernel_profile(int client_id, struct func_record frecord, op_info op_info_cur, int tpc_usage, int cur_iter) {
	// Log before waiting at the barrier
	// std::cout << "Client " << client_id << " reached the barrier." << std::endl;

	// All threads block here until the required number of threads call wait()
	// sync_barrier.wait();

	// Log after passing the barrier
	// std::cout << "Client " << client_id << " passed the barrier and is executing." << std::endl;

	schedule_kernel_profile(frecord, sched_streams[client_id], client_id,
	events[client_id][event_ids[client_id]], seen, event_ids,
	client_id, op_info_cur, tpc_usage, cur_iter);
	pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
	is_executing[client_id] = false;
}



void Scheduler::schedule_reef(vector<func_record*> frecords, int num_clients, int depth, int hp_client) {

	// schedule based on REEF policy
    
	// if (num_clients==1) {
	// 	if (frecords[0] != NULL) {
	// 		schedule_kernel(*(frecords[0]), sched_streams[0], 0, events[0][event_ids[0]], seen, event_ids, 0);
	// 		pop_from_queue(client_buffers[0], client_mutexes[0], 0);
	// 	}
	// 	return;
	// }
    
	// check for malloc operations
	for (int i=0; i<num_clients; i++) {
		if (frecords[i] != NULL) {
            if (frecords[i]->type == MALLOC_RECORD ||
                frecords[i]->type == MEMCPY_RECORD || 
                frecords[i]->type == MEMSET_RECORD ||
                frecords[i]->type == FREE_RECORD){
				schedule_kernel(*(frecords[i]), sched_streams[i], i, events[i][event_ids[i]], seen, event_ids, i);
				pop_from_queue(client_buffers[i], client_mutexes[i], i);
				return;
			}
		}
	}

    bool canSchedule[num_clients];
    for (int i = 0; i < num_clients; ++i) {
        canSchedule[i] = true;
        if (event_ids[i] >= 1) {
            if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
                // printf("kernel %d finished\n", event_ids[i]);
                unsetmask_nomutex(i);
            }
            else{
                canSchedule[i] = false; 
            }
        }
    }

	// if hp is found, schedule
	if (frecords[hp_client] != NULL && canSchedule[hp_client]) {
		int hp_idx = seen[hp_client];
        op_info op_info_1 = op_info_vector[hp_client][hp_idx];
        int tpc_usage = op_info_1.sm_used / 2;
        tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 16 : tpc_usage;

        // int tpc_usage = op_info_1.knee_tpc;
        if(num_tpcs >= tpc_usage){
            setmask(client_mutexes[hp_client], tpc_usage, hp_client);
            schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
            pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
        }

		// check all kernels, and find suitable
		// for (int i=0; i<hp_client; i++) {
        for (int t = 1; t < num_clients; t++) {
            int i = (hp_client + t) % num_clients;
			if (frecords[i] != NULL && canSchedule[i]) {
				op_info op_info_0 = op_info_vector[i][seen[i]];
				if (op_info_0.duration <= op_info_1.duration && op_info_0.sm_used >= op_info_1.sm_used) {
					// colocate
					DEBUG_PRINT("SCHEDULE seen[0]=%d\n", seen[0]);
                    int tpc_usage = op_info_0.sm_used / 2;
                    tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 16 : tpc_usage;
                    // int tpc_usage = op_info_1.knee_tpc;
                    if(num_tpcs > 0){
                        setmask(client_mutexes[i], min(num_tpcs, tpc_usage), i);
                        schedule_kernel(*(frecords[i]), sched_streams[i], i, events[i][event_ids[i]], seen, event_ids, i);
                        pop_from_queue(client_buffers[i], client_mutexes[i], i);
                    }
					// if one is found, exit
					return;
				}
			}
		}
	}
	else {
		for (int i=0; i<hp_client; i++) {
			if (frecords[i] != NULL)
				penalty += 1;
		}
		if (penalty>=depth) {
			// schedule all
			for (int i=0; i<hp_client; i++) {
				if (frecords[i] != NULL && canSchedule[i]) {
                    op_info op_info_0 = op_info_vector[i][seen[i]];
                    int tpc_usage = op_info_0.sm_used / 2;
                    tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 16 : tpc_usage;
                    if(num_tpcs > 0){
                        setmask(client_mutexes[i], min(num_tpcs, tpc_usage), i);
                        schedule_kernel(*(frecords[i]), sched_streams[i], i, events[i][event_ids[i]], seen, event_ids, i);
                        pop_from_queue(client_buffers[i], client_mutexes[i], i);
                    }
				}
			}
			penalty = 0;
		}
	}
}

// Use Look ahead schedule v2
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {

// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients);
// 	std::vector<std::vector<Scheduler::ScheduleItem>> current_schedule;

// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}

// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		if(current_schedule.empty() && ready_client.size() == num_all_clients){
			
// 			// begin schedule for the current clients
// 			current_schedule = lookahead_schedule(ready_client);
// 			// current_schedule is now std::vector<std::vector<ScheduleItem>>
// 			for (size_t b = current_schedule.size(); b > 0; b--) {
// 				size_t index = b - 1; // since b goes from current_schedule.size() to 1
// 				// printf("Batch %zu:\n", index);
// 				// for (const auto &item : current_schedule[index]) {
// 				// 	printf("  id: %d, kernel_idx: %d, batch: %d, tpc: %d, is_critical: %d\n",
// 				// 		item.id, item.kernel_idx, item.batch, item.tpc, (int)item.is_critical);
// 				// }
// 			}
			
// 		}

// 		if (!current_schedule.empty()) {
//         // Use existing schedule
//         // Wait for all clients in the current batch to be ready
//         auto &current_batch_schedule = current_schedule.back();
//         bool all_ready = true;

//         // Check if all clients in the batch are ready
//         for (const auto &item : current_batch_schedule) {
//             int client_id = item.id;
//             if (std::find(ready_client.begin(), ready_client.end(), client_id) == ready_client.end()) {
//                 all_ready = false;
//                 break;
//             }
//         }

//         if (all_ready) {
//             bool all_scheduled = true; // Flag to track if all clients are scheduled successfully

//             // Iterate over each client in the current batch
//             for (auto it = current_batch_schedule.begin(); it != current_batch_schedule.end(); ) {
//                 int client_id = it->id;

//                 // Check if the client can be scheduled
//                 if (frecords[client_id] != nullptr && !is_executing[client_id] && num_tpcs >= it->tpc) {

//                     // Schedule the client
//                     op_info &op_info_cur = op_info_vector[client_id][it->kernel_idx];
//                     int tpc_usage = it->tpc;
//                     setmask(client_mutexes[client_id], tpc_usage, client_id);
//                     is_executing[client_id] = true;

//                     // Enqueue the task for execution
//                     pool.enqueue(&Scheduler::execute_kernel_profile, this, client_id, *(frecords[client_id]), op_info_cur, tpc_usage, num_client_cur_iters[client_id]);

//                     // Remove the client from the current batch since it's scheduled
//                     it = current_batch_schedule.erase(it);
//                 } else {
//                     // Cannot schedule this client now
//                     all_scheduled = false;
//                     ++it;
//                 }
//             }

//             // After attempting to schedule all clients
//             if (all_scheduled) {
//                 current_schedule.pop_back();
//             }
//         }
// 	}
    



// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {
			
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

// Use Look ahead schedule
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {

// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients+1);
// 	std::vector<std::vector<Scheduler::ScheduleItem>> current_schedule;

// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}

// 		if(can_schedule){
// 			// for (int j = 0; j < num_clients; ++j) {
// 			// 	if (frecords[j] != NULL) {
// 			// 			schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 			// 			pop_from_queue(client_buffers[j], client_mutexes[j], j);			
// 			// 	}
// 			// }
// 			for (int j = 0; j < num_clients; ++j) {
// 				if (frecords[j] != NULL) {
// 					if (frecords[j]->type != MALLOC_RECORD && 
// 						frecords[j]->type != MEMCPY_RECORD && 
// 						frecords[j]->type != MEMSET_RECORD && 
// 						frecords[j]->type != FREE_RECORD && is_executing[j] == false)  {
					
// 						int tpc_usage = knee_tpc; 
// 						op_info &op_info_cur = op_info_vector[j][seen[j]];
// 						setmask(client_mutexes[j], tpc_usage, j);
// 						is_executing[j] = true;
// 						// pool.enqueue(&Scheduler::execute_kernel, this, j, *(frecords[j]));
// 						pool.enqueue(&Scheduler::execute_kernel_profile, this, j, *(frecords[j]), op_info_cur, tpc_usage, num_client_cur_iters[j]);
// 					}
// 					else{
// 						schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 						pop_from_queue(client_buffers[j], client_mutexes[j], j);	
// 					}
// 				}
// 			}
		
// 		}
// 		else{
// 			for (int j = 0; j < num_clients; ++j) {
// 				if (frecords[j] != NULL) {
// 						schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 						pop_from_queue(client_buffers[j], client_mutexes[j], j);			
// 				}
// 			}
// 		}




// 		// if(can_schedule && seen[0] > 64){
// 		// 	// check if GPU work for this client has finished
// 		// 		// if (!locked[0]) {
// 		// 		// 	pthread_mutex_lock(client_mutexes[0]);
// 		// 		// 	locked[0] = true;
// 		// 		// 	DEBUG_PRINT("LOCK CLIENT %d\n", 0);
// 		// 		// }
// 		// 		bool ready = true;

// 		// 		if (event_ids[0] >= 1) {
// 		// 			if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 		// 				ready &= false;
// 		// 		}

// 		// 		if(ready){
// 		// 			break;
// 		// 		}
// 		// }

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		// float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
// 		auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
// 		// duration /= 1000.0;
// 		printf("Total loop took %ld nanoseconds\n", duration_ns);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }


// Use normal multi-stream
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {

// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
// 	auto start_total = std::chrono::high_resolution_clock::now();
   
//     auto start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start_total.time_since_epoch());
//     std::cout << "Start time: " << start_ns.count() << " ns" << std::endl;


// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients);
// 	std::vector<std::vector<Scheduler::ScheduleItem>> current_schedule;


// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}

// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}

//         // bool shift = true;
// 		// if(can_schedule && ready_client.size() == num_all_clients){
// 		// 	for (int j = 0; j < num_clients; ++j) {
// 		// 		if (frecords[j] != NULL  && seen[j] < 1336 && is_executing[j] == false) {
// 		// 			shift = false;
// 		// 			// schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 		// 			// pop_from_queue(client_buffers[j], client_mutexes[j], j);
//         //             is_executing[j] = true;
//         //             pool.enqueue(&Scheduler::execute_kernel, this, j, *(frecords[j]));
// 		// 		}
// 		// 	}
// 		// }

// 		if(can_schedule && ready_client.size() == num_all_clients){

// 			// auto co_exec_start = std::chrono::high_resolution_clock::now();
// 			// auto co_exec_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(co_exec_start.time_since_epoch());
// 			// std::cout << "Shcedule start time: " << co_exec_ns.count() << " ns !!!!" << std::endl;

// 			printf("begin co-executing !!!! \n");

// 			for (int client_id : ready_client) {
// 				if (frecords[client_id] != NULL && is_executing[client_id] == false && seen[client_id] < 1400) {
// 				// if (frecords[client_id] != NULL && seen[client_id] < 1400) {
// 					int tpc_usage = -1;
// 					op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
// 					is_executing[client_id] = true;
// 					// if(seen[client_id] < 1336){
// 						// schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 						// pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 						// pool.enqueue(&Scheduler::execute_kernel, this, client_id, *(frecords[client_id]));
// 					// }
// 					// else{

// 					// // tpc_usage = 12; 
// 					// // setmask(client_mutexes[client_id], tpc_usage, client_id);
// 					pool.enqueue(&Scheduler::execute_kernel_profile, this, client_id, *(frecords[client_id]), op_info_cur, tpc_usage, num_client_cur_iters[client_id]);
// 					// }
// 				}
// 			}
// 		}

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if(can_schedule && seen[i] > 1399){
// 				finished += 1;
// 			}
			
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		 // Convert to nanoseconds
// 		auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_total.time_since_epoch());

// 		// Print the start time in nanoseconds since epoch
// 		std::cout << "End time: " << end_ns.count() << " ns" << std::endl;

//     	auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();

// 		// duration /= 1000.0;
// 		printf("Total loop took %ld ns\n", duration_ns);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }




int getRandomTPC() {
    return 1 + rand() % 24;
}

std::vector<std::vector<int>> generateStaticTPCPlan(int num_clients, int num_kernels) {
    std::vector<std::vector<int>> plan(num_clients, std::vector<int>(num_kernels, 0));
    for (int client = 0; client < num_clients; ++client) {
        for (int kernel = 0; kernel < num_kernels; ++kernel) {
            plan[client][kernel] = getRandomTPC();
        }
    }
    return plan;
}

std::string planToString(const std::vector<std::vector<int>>& plan) {
    std::string hash;
    for (const auto& clientPlan : plan) {
        for (int tpc : clientPlan) {
            hash += std::to_string(tpc) + ",";
        }
        hash += "|";
    }
    return hash;
}

std::vector<std::vector<std::vector<int>>> generateMultipleStaticTPCPlans(int num_plans, int num_clients, int num_kernels) {
    std::vector<std::vector<std::vector<int>>> plans;
    std::unordered_set<std::string> planHashes;

    while (static_cast<int>(plans.size()) < num_plans) {
        auto plan = generateStaticTPCPlan(num_clients, num_kernels);
        std::string hash = planToString(plan);
        if (planHashes.find(hash) == planHashes.end()) {
            planHashes.insert(hash);
            plans.push_back(plan);
        }
    }
    return plans;
}

void recordPlanResultToFile(const vector<vector<int>>& plan, int planIndex, long long duration,
	const vector<Scheduler::ScheduleEntry>& scheduleOrder) {
	ofstream ofs("plan_results.txt", ios::app); // Open file in append mode.
	if (!ofs) {
	cerr << "Unable to open file for writing plan results." << endl;
	return;
	}

	ofs << "Plan #" << planIndex << ":\n";
	for (int client = 0; client < plan.size(); ++client) {
	ofs << "  Client " << client << ": ";
	for (int kernel = 0; kernel < plan[client].size(); ++kernel) {
	ofs << plan[client][kernel] << " ";
	}
	ofs << "\n";
	}

	ofs << "Scheduling Order:\n";
	for (const auto& entry : scheduleOrder) {
	ofs << "  (" << entry.client_id << ", " << entry.kernel_index << ", " << entry.tpc_usage << ", " << entry.concurrent_kernel << ")\n";
	}

	ofs << "Duration: " << duration << " microseconds\n";
	ofs << "-----------------------------\n";
	ofs.close();
}



// random generation
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {

// 	srand(static_cast<unsigned int>(time(NULL)));
// 	int numPlansNeeded = num_client_max_iters[0]; 
// 	int NUM_CLIENTS = num_clients;
//  	int NUM_KERNELS = 10;
// 	auto plans = generateMultipleStaticTPCPlans(numPlansNeeded, NUM_CLIENTS, NUM_KERNELS);

// 	// for (int i = 0; i < numPlansNeeded; ++i) {
//     //     std::cout << "Plan #" << i << ":\n";
//     //     for (int client = 0; client < NUM_CLIENTS; ++client) {
//     //         std::cout << "  Client " << client << ": ";
//     //         for (int kernel = 0; kernel < NUM_KERNELS; ++kernel) {
//     //             std::cout << plans[i][client][kernel] << " ";
//     //         }
//     //         std::cout << "\n";
//     //     }
//     //     std::cout << "-----------------------------\n";
//     // }
		
// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int window_start[2] = {10, 10};
// 	int window_end = 20;
// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;

// 	auto start_total = std::chrono::high_resolution_clock::now();
// 	auto start_micro = std::chrono::duration_cast<std::chrono::microseconds>(start_total.time_since_epoch());
// 	// std::cout << "Start time: " << start_micro.count() << " μs" << std::endl;
	
// 	std::chrono::time_point<std::chrono::system_clock> start_profile;

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients);
// 	std::vector<int> schedule_client(2, 0);
// 	bool finish_profile = false;
// 	bool set_time = true;
// 	std::unordered_map<int, std::set<int>> kernelset;
// 	std::vector<Scheduler::ScheduleEntry> scheduleOrder;

// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}

// 				// unsetmask_nomutex(i);
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}


// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}




// 		if(can_schedule){

// 			// window_start[0] = 100;
// 			// window_start[1] = 100;

// 			bool canSchedule[num_clients];
// 			for (int i = 0; i < num_clients; ++i) {
// 				canSchedule[i] = true;
// 				if (event_ids[i] >= 1) {
// 					if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 						// printf("kernel %d from client %d is finished\n", seen[i], i);
// 						unsetmask_nomutex(i);
// 					}
// 					else{
// 						// printf("kernel %d from client %d is not finished\n", seen[i], i);
// 						canSchedule[i] = false; 
// 					}
// 				}
// 			}

// 			if(finish_profile == false){

// 				if(num_client_cur_iters[0] == num_client_cur_iters[1]){
// 					if ((seen[0] < window_start[0] || seen[1] < window_start[1])) { 
// 						for (int client_id : ready_client) {
// 							if(seen[client_id] < window_start[client_id]){
// 								if (frecords[client_id] != NULL) {
// 									schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 									pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 								}
// 							}
// 						}
// 					} 
// 					else {

// 						bool begin_sche = true;
// 						if(seen[0] == window_start[0] && seen[1] == window_start[1]){
// 							if(cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess || cudaEventQuery(*(events[1][event_ids[1] - 1])) != cudaSuccess){
// 								begin_sche = false;
// 							}
// 						}
// 						if(begin_sche){

// 							if(set_time == true){
// 								// printf("reinitlize the start time\n");
// 								start_profile = std::chrono::high_resolution_clock::now();
// 								set_time = false;
// 							}
	
// 							for (int client_id : ready_client) {
// 								std::random_device rd;
// 								std::mt19937 gen(rd());
// 								std::bernoulli_distribution d(0.5);
// 								bool randomBool = d(gen);
	
// 								if (frecords[client_id] != NULL && seen[client_id] < window_end && canSchedule[client_id] == true) {
// 									int tpc_usage = plans[num_client_cur_iters[client_id] - 10][client_id][seen[client_id] - window_start[client_id]];
// 									if(num_tpcs >= tpc_usage && randomBool == true){
	
// 										// std::cout << "Scheduling kernel " << seen[client_id]+1
// 										// << " for client " << client_id 
// 										// << " with TPC usage: " << tpc_usage << std::endl;
	
// 										op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
// 										setmask(client_mutexes[client_id], tpc_usage, client_id);
// 										int concurrent_id;
// 										if(client_id == 0){
// 											concurrent_id = 1;
// 										}
// 										else{
// 											concurrent_id = 0;
// 										}

// 										if(cudaEventQuery(*(events[concurrent_id][event_ids[concurrent_id] - 1])) == cudaSuccess){
// 											scheduleOrder.push_back({client_id, seen[client_id] - window_start[client_id], tpc_usage, -1});
// 										} else {
// 											scheduleOrder.push_back({client_id, seen[client_id] - window_start[client_id], tpc_usage, seen[concurrent_id] - window_start[client_id] - 1});
// 										}
									
// 										schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 										// schedule_kernel_profile(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id, op_info_cur, tpc_usage, num_client_cur_iters[client_id]);
// 										pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 									}
// 								}
// 							}
// 						}
// 					}
// 				}
// 			}
// 			else{

// 				bool begin_sche = true;
// 				if(seen[0] == window_end  && seen[1] == window_end){
// 					if(cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess || cudaEventQuery(*(events[1][event_ids[1] - 1])) != cudaSuccess){
// 						begin_sche = false;
// 					}
// 				}
// 				if(begin_sche){
// 					for (int j = 0; j < num_clients; ++j) {
// 						if (frecords[j] != NULL) {
// 							if (frecords[j]->type != MALLOC_RECORD && 
// 								frecords[j]->type != MEMCPY_RECORD && 
// 								frecords[j]->type != MEMSET_RECORD && 
// 								frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 									// printf("start executing other kernels\n");
// 									schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 									pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}


// 		int finished = 0;
// 		int readytoprofile = 0;
// 		for (int i=0; i<num_clients; i++) {

//             // if(can_schedule && seen[i] >= 20){
//             //     if (cudaEventQuery(*(events[i][event_ids[i]])) == cudaSuccess){
//             //         finished += 1;
//             //     }
// 			// }

// 			if(can_schedule && seen[i] == window_end){
// 				if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 					readytoprofile += 1;
// 				}
// 			}

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// unsetmask_nomutex(i);
// 					set_time = true;
// 					kernelset[i].clear();
// 					finish_profile = false;
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if(readytoprofile == num_clients){
// 			finish_profile = true;
// 			auto end_10 = std::chrono::high_resolution_clock::now();
// 			auto duration_10_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(end_10 - start_profile).count();
// 			recordPlanResultToFile(plans[num_client_cur_iters[0] - 10], num_client_cur_iters[0] - 10, duration_10_nano, scheduleOrder);
// 			scheduleOrder.clear();
// 			printf(" kernels from %d to %d kernels took %ld nanoseconds\n",window_start[0], window_end, duration_10_nano);
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
//     	auto duration_nano= std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
// 		// duration /= 1000.0;
// 		printf("Total loop took %ld nanoseconds\n", duration_nano);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

vector<Scheduler::KernelData> readCSV(const string& filename, vector<string>& headers) {
    ifstream file(filename); 
    if (!file.is_open()) {
        cerr << "Could not open the file!" << endl;
        exit(1);  
    }
    string line;
    vector<Scheduler::KernelData> kernelData; 
    bool isHeader = true;
    
    while (getline(file, line)) {
        vector<string> row;
        stringstream ss;
        bool insideQuote = false;
        string cell;

        for (char c : line) {
            if (c == '"' && !insideQuote) {
                insideQuote = true;
            } else if (c == '"' && insideQuote) {
                insideQuote = false;
            } else if (c == ',' && !insideQuote) {
                row.push_back(cell);
                cell.clear();
            } else {
                cell.push_back(c);
            }
        }
        if (!cell.empty()) {
            row.push_back(cell);
        }

        if (isHeader) {
            headers = row;
            isHeader = false;  
        } else {
            Scheduler::KernelData kernel;
            kernel.Kernel_ID = stoi(row[0]); 
            kernel.Kernel_Name = row[1];
            kernel.Model = row[14];
			kernel.Duration = stoi(row[15]);
            kernel.Cluster = stoi(row[16]);    
            kernelData.push_back(kernel);
        }
    }

    file.close(); 
    return kernelData;  
}



vector<pair<Scheduler::KernelData, Scheduler::KernelData>> generateCoExecutionPlan(const vector<Scheduler::KernelData>& kernels) {
    vector<pair<Scheduler::KernelData, Scheduler::KernelData>> plan;

    // Organize kernels into groups based on the Cluster (Group)
    vector<vector<Scheduler::KernelData>> groups;
    for (const auto& kernel : kernels) {
        if (kernel.Cluster >= groups.size()) {
            groups.resize(kernel.Cluster + 1);
        }
        groups[kernel.Cluster].push_back(kernel);
    }

    // Pair kernels within the same group (group i with group i)
	for (int i = 0; i < groups.size(); ++i) {
        const auto& group = groups[i];
        if (group.size() >= 1) {
            int index1 = rand() % group.size();
            int index2 = rand() % group.size();
            plan.push_back(make_pair(group[index1], group[index2]));
        }
    }

    // Pair kernels between different groups (group i with group j, i != j)
    for (int i = 0; i < groups.size(); ++i) {
        for (int j = i + 1; j < groups.size(); ++j) {
            const auto& group_i = groups[i];
            const auto& group_j = groups[j];
            // Randomly pick one kernel from each group
            int index_i = rand() % group_i.size();
            int index_j = rand() % group_j.size();
            plan.push_back(make_pair(group_i[index_i], group_j[index_j]));
        }
    }

    return plan;
}

std::map<std::pair<int, int>, double> readContentionMatrix(const std::string& filename) {
    std::map<std::pair<int, int>, double> matrix;
    std::ifstream file(filename);
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string g1_str, g2_str, factor_str;
        std::getline(ss, g1_str, ',');
        std::getline(ss, g2_str, ',');
        std::getline(ss, factor_str, ',');

        int g1 = std::stoi(g1_str);
        int g2 = std::stoi(g2_str);
        double factor = std::stod(factor_str);

        matrix[{g1, g2}] = factor;

        // Debug print
        // std::cout << "Inserted: (" << g1 << ", " << g2 << ") => " << factor << std::endl;
    }

    return matrix;
}


void writeContentionMatrixToCSV(const vector<vector<float>>& contention_matrix, int num_groups, const string& filename) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Could not open the file for writing!" << endl;
        return;
    }

    // Write the header (group names and contention factor)
    file << "Group_1,Group_2,Contention_Factor" << endl;

    // Write the matrix values, including the group numbers and the diagonal
    for (int i = 0; i < num_groups; ++i) {
        for (int j = 0; j < num_groups; ++j) {  // Include i <= j, not just i < j
            if (contention_matrix[i][j] == 0 && contention_matrix[j][i] != 0) {
                // If contention_matrix[i][j] is 0, but contention_matrix[j][i] is not, fill with the reverse value
                file << i << "," << j << "," << contention_matrix[j][i] << endl;
            } else if (contention_matrix[i][j] == 0 && contention_matrix[j][i] == 0) {
                // If both contention_matrix[i][j] and contention_matrix[j][i] are 0, flag as not profiled
                file << i << "," << j << ",Not Profiled" << endl;
            } else {
                // Otherwise, write the actual value
                file << i << "," << j << "," << contention_matrix[i][j] << endl;
            }
        }
    }

    file.close();
    cout << "Contention matrix saved to " << filename << endl;
}



void fillTPCDataByKernelID(
    const std::string& tpc_file,
    std::vector<Scheduler::KernelData>& kernels
) {
    std::ifstream file(tpc_file);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << tpc_file << "\n";
        return;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);

    // Build a quick lookup: Kernel_ID → KernelData*
    std::unordered_map<int, Scheduler::KernelData*> id_map;
    id_map.reserve(kernels.size());
    for (auto& k : kernels) {
        id_map[k.Kernel_ID] = &k;
    }

    while (std::getline(file, line)) {
        if (line.empty()) 
            continue;

        // Split CSV row by commas
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        // Expect: Name, Index, 1..24, Knee TPC → 27 tokens
        if (tokens.size() < 27) 
            continue;

        // Parse kernel index from second column
        int kernel_id = std::stoi(tokens[1]);
        auto it = id_map.find(kernel_id);
        if (it == id_map.end()) 
            continue;

        Scheduler::KernelData* kd = it->second;

        // Read TPC performance for columns "1" through "24"
        kd->TPC_Performance.resize(24);
        for (int i = 0; i < 24; ++i) {
            kd->TPC_Performance[i] = std::stod(tokens[2 + i]);
        }

        // Read Knee_TPC from last column
        kd->Knee_TPC = std::stoi(tokens[26]);
    }
}

void printKernelData(const std::vector<Scheduler::KernelData>& kernels) {
    for (const auto& k : kernels) {
        std::cout << "=== Kernel ID: " << k.Kernel_ID << " ===" << std::endl;
        std::cout << "Name       : " << k.Kernel_Name << std::endl;
        std::cout << "Model      : " << k.Model << std::endl;
        std::cout << "Cluster    : " << k.Cluster << std::endl;
        std::cout << "Duration   : " << k.Duration << " ns" << std::endl;

        std::cout << "Block Size : " << k.Block_Size << std::endl;
        std::cout << "Grid Size  : " << k.Grid_Size << std::endl;
        std::cout << "Exclusive  : " << k.Exclusive_Run << " ns" << std::endl;
        std::cout << "Knee TPC   : " << k.Knee_TPC << std::endl;

        std::cout << "TPC Performance:" << std::endl;
        for (size_t i = 0; i < k.TPC_Performance.size(); ++i) {
            std::cout << "  TPC " << (i + 1) << ": " << k.TPC_Performance[i] << " ns" << std::endl;
        }
        std::cout << "-----------------------------" << std::endl;
    }
}

// accuracy profile
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {
// 	string filename = "/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_profile/kernel_groups/Dnet_32_groups.csv";
// 	std::string tpc_profile_file = "/home/zixi/orion_bu/artifact_evaluation/fig7/kernel_profiles/densenet201_bz32.csv";
// 	string filename_cm = "/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_profile/contention_matrix/contention_matrix_Dnet_32.csv";
//     map<pair<int, int>, double> contention_map = readContentionMatrix(filename_cm);

//     vector<string> headers;
//     vector<Scheduler::KernelData> kernels = readCSV(filename, headers);
// 	fillTPCDataByKernelID(tpc_profile_file, kernels);
// 	std::unordered_map<int, Scheduler::KernelData*> id_to_kernel;

// 	for (auto& kernel : kernels) {
// 		id_to_kernel[kernel.Kernel_ID] = &kernel;
// 	}
// 	// printKernelData(kernels);

// 	int num_groups = 0;
// 	for (const auto& kernel : kernels) {
// 		if (kernel.Cluster > num_groups) {
// 			num_groups = kernel.Cluster;  
// 		}
// 	}
// 	num_groups += 1;

// 	srand(time(0)); // Seed for random number generation
//     vector<pair<Scheduler::KernelData, Scheduler::KernelData>> profile_plan = generateCoExecutionPlan(kernels);
	
// 	if(!warmup){
// 		for (const auto& pair : profile_plan) {
// 			cout << "Pair: Kernel_ID_1 = " << pair.first.Kernel_ID << ", Kernel_ID_2 = " << pair.second.Kernel_ID
// 				 << ", Model_1 = " << pair.first.Model << ", Model_2 = " << pair.second.Model
// 				 << ", Cluster_1 = " << pair.first.Cluster << ", Cluster_2 = " << pair.second.Cluster << endl;
// 		}
// 		printf("Number of valid plans after filtering: %ld\n", profile_plan.size());
// 	}

// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	auto& pair = profile_plan[0];

// 	int window_start[2];
// 	int window_end[2];
// 	window_start[0] = pair.first.Kernel_ID;
// 	window_start[1] = pair.second.Kernel_ID; 
// 	// window_start[0] = 3867;
// 	// window_start[1] = 4006; 
// 	window_end[0] = window_start[0] + 1;
// 	window_end[1] = window_start[1] + 1;
	
// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;

// 	auto start_total = std::chrono::high_resolution_clock::now();
// 	auto start_micro = std::chrono::duration_cast<std::chrono::microseconds>(start_total.time_since_epoch());
// 	// std::cout << "Start time: " << start_micro.count() << " μs" << std::endl;
	
// 	std::chrono::time_point<std::chrono::system_clock> start_profile;

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients);
// 	std::vector<int> schedule_client(2, 0);
// 	bool finish_profile[2] = {false, false};
// 	bool set_time = true;
// 	std::unordered_map<int, std::set<int>> kernelset;
// 	std::vector<Scheduler::ScheduleEntry> scheduleOrder;
// 	int tpc_assignment[2];
// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}


// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}

// 		if(can_schedule){
// 			bool canSchedule[num_clients];
// 			for (int i = 0; i < num_clients; ++i) {
// 				canSchedule[i] = true;
// 				if (event_ids[i] >= 1) {
// 					if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 						unsetmask_nomutex(i);
// 					}
// 					else{
// 						canSchedule[i] = false; 
// 					}
// 				}
// 			}
// 			if(finish_profile[0] == false && finish_profile[1] == false && num_client_cur_iters[0] == num_client_cur_iters[1]){
// 					for (int client_id : ready_client) {
// 						if (seen[client_id] < window_start[client_id]) {
// 							if (frecords[client_id] != NULL) {

// 								schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, 
// 												 events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 								pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 							}
// 						}
// 					}
// 					if(seen[0] == window_start[0] && seen[1] == window_start[1]){
						
// 						bool begin_sche = true;
// 						for (int i = 0; i < 2; ++i) {
// 							if ((seen[i] != 0) && seen[i] == window_start[i]) {
// 								if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess) {
// 									begin_sche = false;
// 									break; 
// 								}
// 							}
// 						}
// 						if(begin_sche){

// 							if(ready_client.size() == num_clients && num_tpcs == 24){
// 								if(set_time == true){
// 									start_profile = std::chrono::high_resolution_clock::now();
// 									set_time = false;
// 								}
// 								// tpc_assignment[0] = rand() % 23 + 1;
// 								tpc_assignment[0] = 12;
// 								int remaining_tpc = 24 - tpc_assignment[0];
// 								tpc_assignment[1] = remaining_tpc;
// 								// printf("client 0 use %d, client 1 uses %d\n", tpc_assignment[0], tpc_assignment[1]);
// 								for (int client_id : ready_client) {
// 									if (frecords[client_id] != NULL && seen[client_id] < window_end[client_id]) {
										
// 										setmask(client_mutexes[client_id], tpc_assignment[client_id], client_id);
// 										op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
// 										schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 										// schedule_kernel_profile(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id, op_info_cur, 1, num_client_cur_iters[client_id]);
// 										pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
										
// 									}
// 								}
// 							}
// 						}
// 					}
// 			}
// 			else{
// 				bool begin_sche = true;
// 				if(seen[0] == window_end[0] && seen[1] == window_end[1]){
// 					if(cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess || cudaEventQuery(*(events[1][event_ids[1] - 1])) != cudaSuccess){
// 						begin_sche = false;
// 					}
// 				}
// 				if(begin_sche){
// 					for (int j = 0; j < num_clients; ++j) {
// 						if (frecords[j] != NULL && finish_profile[j] == true) {
// 							if (frecords[j]->type != MALLOC_RECORD && 
// 								frecords[j]->type != MEMCPY_RECORD && 
// 								frecords[j]->type != MEMSET_RECORD && 
// 								frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 									op_info &op_info_cur = op_info_vector[j][seen[j]];
// 									schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 									// schedule_kernel_profile(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, op_info_cur, 1, num_client_cur_iters[j]);
// 									pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}

// 		int readytoprofile = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if(can_schedule && seen[i] == window_end[i]){
// 				if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 					readytoprofile += 1;
// 				}
// 			}
// 		}

// 		if(readytoprofile == num_clients){
// 			finish_profile[0] = true;
// 			finish_profile[1] = true;
// 			auto end_10 = std::chrono::high_resolution_clock::now();
// 			auto duration_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(end_10 - start_profile).count();
// 			float contention_factor = (pair.first.Duration + pair.second.Duration) / static_cast<float>(duration_nano);
// 			// printf("the contention factor between cluster %d and cluster %d is %f\n",pair.first.Cluster, pair.second.Cluster, contention_factor);
// 			// printf("in profile client 0 use %d, client 1 uses %d\n", tpc_assignment[0], tpc_assignment[1]);
// 			long dur0 = pair.first.TPC_Performance[tpc_assignment[0]-1];
// 			long dur1 = pair.second.TPC_Performance[tpc_assignment[1]-1];

// 			// printf("pair.first.profile Duration %ld, pair.second.profile Duration: %ld\n", dur0, dur1);
// 			std::pair<int, int> key1 = {pair.first.Cluster, pair.second.Cluster};
// 			auto it = contention_map.find(key1);
// 			double contention;
// 			if (it != contention_map.end()) {
// 				contention = it->second;
// 				// printf("The contention factor between cluster %d and cluster %d is %f\n",
// 				// 	pair.first.Cluster, pair.second.Cluster, contention);
// 			} else {
// 				// printf("No contention factor found for cluster %d and cluster %d\n",
// 				// 	pair.first.Cluster, pair.second.Cluster);
// 			}
// 			long predict_l = (dur0 + dur1) / contention;

// 			// printf("The latency of co-location kernels %d and %d took %ld nanoseconds\n", window_start[0], window_start[1], duration_nano);
// 			printf("The predict latency of co-location kernels %d and %d is: %ld, the real duration took %ld nanoseconds\n", window_start[0], window_start[1],  predict_l, duration_nano);
// 			printf("using only single profile, the predict latency is: %ld\n", max(dur0,dur1));
// 			if (num_client_cur_iters[0] == num_client_cur_iters[1] && (num_client_cur_iters[0] > 9 && num_client_cur_iters[1] > 9) && ((num_client_cur_iters[0] - 10) < profile_plan.size() && (num_client_cur_iters[1] - 10)  < profile_plan.size())) {
// 				pair = profile_plan[(num_client_cur_iters[0]) - 10];
// 				window_start[0] = pair.first.Kernel_ID;
// 				window_start[1] = pair.second.Kernel_ID; 
// 				window_end[0] = window_start[0] + 1;
// 				window_end[1] = window_start[1] + 1;
// 				tpc_assignment[0] = 0;
// 				tpc_assignment[1] = 0;	
// 				printf("will colocation %d and %d\n", window_start[0], window_start[1]);
// 			}
// 		}

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if(can_schedule){
// 							if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess || finish_profile[i] == false){
// 								ready &= false;
// 							}
// 						}
// 						else{
// 							if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess){
// 								ready &= false;
// 							}
// 						}
// 					}
// 				}
// 				if (ready) {
// 					unsetmask_nomutex(i);
// 					set_time = true;
// 					kernelset[i].clear();
// 					finish_profile[i] = false;
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
//     	auto duration_nano= std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
// 		// duration /= 1000.0;
// 		printf("Total loop took %ld nanoseconds\n", duration_nano);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }



// For co-location profile
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {
// 	printf("11111\n");
// 	string filename = "/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_small_models/kernel_groups/Tnet_8_groups.csv";
// 	// std::string tpc_profile_file = "/home/zixi/orion_bu/artifact_evaluation/fig7/kernel_profiles/resnet101_bz8.csv";

//     vector<string> headers;
//     vector<Scheduler::KernelData> kernels = readCSV(filename, headers);
// 	printf("22222\n");
// 	// fillTPCDataByKernelID(tpc_profile_file, kernels);

//     // cout << "Headers: ";
//     // for (const auto& header : headers) {
//     //     cout << header << " ";
//     // }
//     // cout << endl;
//     // cout << "Kernel Data: " << endl;

//     // for (const auto& kernel : kernels) {
//     //     cout << "Kernel_ID: " << kernel.Kernel_ID
//     //          << ", Model: " << kernel.Model
//     //          << ", Cluster: " << kernel.Cluster 
// 	// 		 << ", Duration: " << kernel.Duration 
// 	// 		 << endl;
//     // }


// 	int num_groups = 0;
// 	for (const auto& kernel : kernels) {
// 		if (kernel.Cluster > num_groups) {
// 			num_groups = kernel.Cluster;  
// 		}
// 	}
// 	num_groups += 1;
// 	printf("num of groups %d\n", num_groups);
// 	vector<vector<float>> contention_matrix(num_groups, vector<float>(num_groups, 0.0f));

// 	srand(time(0));
// 	vector<pair<Scheduler::KernelData, Scheduler::KernelData>> profile_plan = generateCoExecutionPlan(kernels);
// 	if(!warmup){
// 		for (const auto& pair : profile_plan) {
// 			cout << "Pair: Kernel_ID_1 = " << pair.first.Kernel_ID << ", Kernel_ID_2 = " << pair.second.Kernel_ID
// 				 << ", Model_1 = " << pair.first.Model << ", Model_2 = " << pair.second.Model
// 				 << ", Cluster_1 = " << pair.first.Cluster << ", Cluster_2 = " << pair.second.Cluster << endl;
// 		}
// 		printf("Number of valid plans after filtering: %ld\n", profile_plan.size());
// 	}
	
// 	// Now filter out the pairs that belong to the same model
// 	// profile_plan.erase(
// 	// 	remove_if(profile_plan.begin(), profile_plan.end(),
// 	// 			  [](const pair<Scheduler::KernelData, Scheduler::KernelData>& p) {
// 	// 				  return p.first.Model == p.second.Model; // Remove if both kernels belong to the same model
// 	// 			  }),
// 	// 	profile_plan.end());

// 	// for (auto& pair : profile_plan) {
// 	// 	if (pair.first.Model != "Dnet_8" ) {
// 	// 		// If the first kernel is not from ResNet and the second is from ResNet, swap them
// 	// 		swap(pair.first, pair.second);
// 	// 	}
// 	// }
	

// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	auto& pair = profile_plan[0];

// 	int window_start[2];
// 	int window_end[2];
// 	window_start[0] = pair.first.Kernel_ID;
// 	window_start[1] = pair.second.Kernel_ID; 
// 	// window_start[0] = 3867;
// 	// window_start[1] = 4006; 
// 	window_end[0] = window_start[0] + 1;
// 	window_end[1] = window_start[1] + 1;
	
// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;

// 	auto start_total = std::chrono::high_resolution_clock::now();
// 	auto start_micro = std::chrono::duration_cast<std::chrono::microseconds>(start_total.time_since_epoch());
// 	// std::cout << "Start time: " << start_micro.count() << " μs" << std::endl;
	
// 	std::chrono::time_point<std::chrono::system_clock> start_profile;

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients);
// 	std::vector<int> schedule_client(2, 0);
// 	bool finish_profile[2] = {false, false};
// 	bool set_time = true;
// 	std::unordered_map<int, std::set<int>> kernelset;
// 	std::vector<Scheduler::ScheduleEntry> scheduleOrder;
	
// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}


// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}


// 		if(can_schedule){
// 			bool canSchedule[num_clients];
// 			for (int i = 0; i < num_clients; ++i) {
// 				canSchedule[i] = true;
// 				if (event_ids[i] >= 1) {
// 					if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 						unsetmask_nomutex(i);
// 					}
// 					else{
// 						canSchedule[i] = false; 
// 					}
// 				}
// 			}
// 			// printf("finish profile 0: %d, finish profile 1: %d\n", finish_profile[0] ,finish_profile[1] );
// 			if(finish_profile[0] == false && finish_profile[1] == false && num_client_cur_iters[0] == num_client_cur_iters[1]){
// 				// printf("num_client_cur_iters[0]: %d, num_client_cur_iters[1]: %d\n", num_client_cur_iters[0], num_client_cur_iters[1]);
// 					for (int client_id : ready_client) {
// 						if (seen[client_id] < window_start[client_id]) {
// 							if (frecords[client_id] != NULL) {

// 								schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, 
// 												 events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 								pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 							}
// 						}
// 					}
// 					// printf("seen[0]: %d and seen[1]: %d\n", seen[0], seen[1]);
// 					if(seen[0] == window_start[0] && seen[1] == window_start[1]){
						
// 						// printf("num_client_cur_iters[0]: %d, num_client_cur_iters[1]: %d\n", num_client_cur_iters[0], num_client_cur_iters[1]);
// 						bool begin_sche = true;
// 						// printf("seen[0]: %d and seen[1]: %d\n", seen[0], seen[1]);
// 						for (int i = 0; i < 2; ++i) {
// 							if ((seen[i] != 0) && seen[i] == window_start[i]) {
// 								if (cudaEventQuery(*(events[i][event_ids[i] - 1])) != cudaSuccess) {
// 									begin_sche = false;
// 									break; 
// 								}
// 							}
// 						}
// 						if(begin_sche){
// 							if(set_time == true){
// 								// printf("reinitlize the start time\n");
// 								start_profile = std::chrono::high_resolution_clock::now();
// 								set_time = false;
// 							}
// 							if(ready_client.size() == num_clients){
// 								for (int client_id : ready_client) {
// 									if (frecords[client_id] != NULL && seen[client_id] < window_end[client_id]) {
// 										op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
// 										setmask(client_mutexes[client_id], 12, client_id);
// 										schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 										// schedule_kernel_profile(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id, op_info_cur, 1, num_client_cur_iters[client_id]);
// 										pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
										
// 									}
// 								}
// 							}
// 						}
// 					}
// 			}
// 			else{
// 				bool begin_sche = true;
// 				if(seen[0] == window_end[0] && seen[1] == window_end[1]){
// 					if(cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess || cudaEventQuery(*(events[1][event_ids[1] - 1])) != cudaSuccess){
// 						begin_sche = false;
// 					}
// 				}
// 				if(begin_sche){
// 					for (int j = 0; j < num_clients; ++j) {
// 						if (frecords[j] != NULL && finish_profile[j] == true) {
// 							if (frecords[j]->type != MALLOC_RECORD && 
// 								frecords[j]->type != MEMCPY_RECORD && 
// 								frecords[j]->type != MEMSET_RECORD && 
// 								frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 									op_info &op_info_cur = op_info_vector[j][seen[j]];
// 									schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 									// schedule_kernel_profile(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, op_info_cur, 1, num_client_cur_iters[j]);
// 									pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}

// 		int readytoprofile = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if(can_schedule && seen[i] == window_end[i]){
// 				if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 					readytoprofile += 1;
// 				}
// 			}
// 		}

// 		if(readytoprofile == num_clients){
// 			finish_profile[0] = true;
// 			finish_profile[1] = true;
// 			long dur0 = pair.first.Duration;
// 			long dur1 = pair.second.Duration;
// 			auto end_10 = std::chrono::high_resolution_clock::now();
// 			auto duration_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(end_10 - start_profile).count();
// 			float contention_factor = max(dur0, dur1) / static_cast<float>(duration_nano);
// 			// printf("pair.first.Duration %ld, pair.second.Duration: %ld, duration_nano: %f", pair.first.Duration, pair.second.Duration, static_cast<float>(duration_nano));
// 			contention_matrix[pair.first.Cluster][pair.second.Cluster] = contention_factor;
// 			printf("the contention factor between cluster %d and cluster %d is %f\n",pair.first.Cluster, pair.second.Cluster, contention_matrix[pair.first.Cluster][pair.second.Cluster]);
// 			// printf("The latency of co-location kernels %d and %d took %ld nanoseconds\n", window_start[0], window_start[1], duration_nano);
// 			if (num_client_cur_iters[0] == num_client_cur_iters[1] && (num_client_cur_iters[0] > 9 && num_client_cur_iters[1] > 9) && ((num_client_cur_iters[0] - 10) < profile_plan.size() && (num_client_cur_iters[1] - 10)  < profile_plan.size())) {
// 				pair = profile_plan[(num_client_cur_iters[0]) - 10];
// 				window_start[0] = pair.first.Kernel_ID;
// 				window_start[1] = pair.second.Kernel_ID; 
// 				window_end[0] = window_start[0] + 1;
// 				window_end[1] = window_start[1] + 1;
// 				printf("will colocation %d and %d\n", window_start[0], window_start[1]);
// 			}
// 		}

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if(can_schedule){
// 							if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess || finish_profile[i] == false){
// 								ready &= false;
// 							}
// 						}
// 						else{
// 							if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess){
// 								ready &= false;
// 							}
// 						}
// 					}
// 				}
// 				if (ready) {
// 					unsetmask_nomutex(i);
// 					set_time = true;
// 					kernelset[i].clear();
// 					finish_profile[i] = false;
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {

// 		string output_filename = "contention_matrix.csv";
// 		// cout << "\nContention Matrix:" << endl;
// 		// cout << "Group_1, Group_2, Contention_Factor" << endl;
		
// 		// Iterate through the matrix and print values
// 		// for (int i = 0; i < num_groups; ++i) {
// 		// 	for (int j = 0; j < num_groups; ++j) {  // Print all pairs, including diagonal
// 		// 		cout << i << ", " << j << ", " << contention_matrix[i][j] << endl;
// 		// 	}
// 		// }
// 		writeContentionMatrixToCSV(contention_matrix, num_groups, output_filename);

// 		auto end_total = std::chrono::high_resolution_clock::now();
//     	auto duration_nano= std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
// 		// duration /= 1000.0;
// 		printf("Total loop took %ld nanoseconds\n", duration_nano);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }


void recordKRISP(int planIndex, long long duration,
	const vector<Scheduler::ScheduleEntry>& scheduleOrder) {
	ofstream ofs("plan_results_KRISP.txt", ios::app); // Open file in append mode.
	if (!ofs) {
	cerr << "Unable to open file for writing plan results." << endl;
	return;
	}

	ofs << "Plan #" << planIndex << ":\n";

	ofs << "Scheduling Order:\n";
	for (const auto& entry : scheduleOrder) {
	ofs << "  (" << entry.client_id << ", " << entry.kernel_index << ", " << entry.tpc_usage << ", " << entry.concurrent_kernel << ")\n";
	}

	ofs << "Duration: " << duration << " microseconds\n";
	ofs << "-----------------------------\n";
	ofs.close();
}

// test window schedule for KRISP
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {
		
// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int window_start[2] = {60, 60};
// 	int window_end = 70;
// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;

// 	auto start_total = std::chrono::high_resolution_clock::now();
// 	auto start_micro = std::chrono::duration_cast<std::chrono::microseconds>(start_total.time_since_epoch());
// 	// std::cout << "Start time: " << start_micro.count() << " μs" << std::endl;
	
// 	std::chrono::time_point<std::chrono::system_clock> start_profile;

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients);
// 	std::vector<int> schedule_client(2, 0);
// 	bool finish_profile = false;
// 	bool set_time = true;
// 	std::unordered_map<int, std::set<int>> kernelset;
// 	std::vector<Scheduler::ScheduleEntry> scheduleOrder;

// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}

// 				// unsetmask_nomutex(i);
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}


// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}




// 		if(can_schedule){

// 			bool canSchedule[num_clients];
// 			for (int i = 0; i < num_clients; ++i) {
// 				canSchedule[i] = true;
// 				if (event_ids[i] >= 1) {
// 					if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 						// printf("kernel %d from client %d is finished\n", seen[i], i);
// 						unsetmask_nomutex(i);
// 					}
// 					else{
// 						// printf("kernel %d from client %d is not finished\n", seen[i], i);
// 						canSchedule[i] = false; 
// 					}
// 				}
// 			}

// 			if(finish_profile == false){

// 				if(num_client_cur_iters[0] == num_client_cur_iters[1]){
// 					if ((seen[0] < window_start[0] || seen[1] < window_start[1])) { 
// 						for (int client_id : ready_client) {
// 							if(seen[client_id] < window_start[client_id]){
// 								if (frecords[client_id] != NULL) {
// 									schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 									pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 								}
// 							}
// 						}
// 					} 
// 					else {

// 						bool begin_sche = true;
// 						if(seen[0] == window_start[0] && seen[1] == window_start[1]){
// 							if(cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess || cudaEventQuery(*(events[1][event_ids[1] - 1])) != cudaSuccess){
// 								begin_sche = false;
// 							}
// 						}
// 						if(begin_sche){

// 							if(set_time == true){
// 								// printf("reinitlize the start time\n");
// 								start_profile = std::chrono::high_resolution_clock::now();
// 								set_time = false;
// 							}
	
// 							for (int client_id : ready_client) {

// 								if (frecords[client_id] != NULL && seen[client_id] < window_end && canSchedule[client_id] == true) {
// 									op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
// 									if(num_tpcs > 0){
// 										int tpc_usage = min(num_tpcs, op_info_cur.knee_tpc);
// 										setmask(client_mutexes[client_id], tpc_usage, client_id);

// 										int concurrent_id;
// 										if(client_id == 0){
// 											concurrent_id = 1;
// 										}
// 										else{
// 											concurrent_id = 0;
// 										}

// 										if(cudaEventQuery(*(events[concurrent_id][event_ids[concurrent_id] - 1])) == cudaSuccess){
// 											scheduleOrder.push_back({client_id, seen[client_id] - window_start[client_id], tpc_usage, -1});
// 										} else {
// 											scheduleOrder.push_back({client_id, seen[client_id] - window_start[client_id], tpc_usage, seen[concurrent_id] - window_start[client_id] - 1});
// 										}
									
// 										schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 										// schedule_kernel_profile(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id, op_info_cur, tpc_usage, num_client_cur_iters[client_id]);
// 										pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 									}
									
// 								}
// 							}
// 						}
// 					}
// 				}
// 			}
// 			else{

// 				bool begin_sche = true;
// 				if(seen[0] == window_end  && seen[1] == window_end){
// 					if(cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess || cudaEventQuery(*(events[1][event_ids[1] - 1])) != cudaSuccess){
// 						begin_sche = false;
// 					}
// 				}
// 				if(begin_sche){
// 					for (int j = 0; j < num_clients; ++j) {
// 						if (frecords[j] != NULL) {
// 							if (frecords[j]->type != MALLOC_RECORD && 
// 								frecords[j]->type != MEMCPY_RECORD && 
// 								frecords[j]->type != MEMSET_RECORD && 
// 								frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 									// printf("start executing other kernels\n");
// 									schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 									pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}


// 		int finished = 0;
// 		int readytoprofile = 0;
// 		for (int i=0; i<num_clients; i++) {

//             // if(can_schedule && seen[i] >= 20){
//             //     if (cudaEventQuery(*(events[i][event_ids[i]])) == cudaSuccess){
//             //         finished += 1;
//             //     }
// 			// }

// 			if(can_schedule && seen[i] == window_end){
// 				if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 					readytoprofile += 1;
// 				}
// 			}

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// unsetmask_nomutex(i);
// 					set_time = true;
// 					kernelset[i].clear();
// 					finish_profile = false;
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if(readytoprofile == num_clients){
// 			finish_profile = true;
// 			auto end_10 = std::chrono::high_resolution_clock::now();
// 			auto duration_10_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(end_10 - start_profile).count();
// 			recordKRISP(num_client_cur_iters[0] - 10, duration_10_nano, scheduleOrder);
// 			scheduleOrder.clear();
// 			printf(" kernels from %d to %d kernels took %ld nanoseconds\n",window_start[0], window_end, duration_10_nano);
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
//     	auto duration_nano= std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
// 		// duration /= 1000.0;
// 		printf("Total loop took %ld nanoseconds\n", duration_nano);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

// TEST SCHEDULE FOR oron
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {
		
// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int window_start[2] = {60, 60};
// 	int window_end = 70;
// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;

// 	auto start_total = std::chrono::high_resolution_clock::now();
// 	auto start_micro = std::chrono::duration_cast<std::chrono::microseconds>(start_total.time_since_epoch());
// 	// std::cout << "Start time: " << start_micro.count() << " μs" << std::endl;
	
// 	std::chrono::time_point<std::chrono::system_clock> start_profile;

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients);
// 	std::vector<int> schedule_client(2, 0);
// 	bool finish_profile = false;
// 	bool set_time = true;
// 	std::unordered_map<int, std::set<int>> kernelset;
// 	std::vector<Scheduler::ScheduleEntry> scheduleOrder;
// 	long int endT = 0;
// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}

// 				// unsetmask_nomutex(i);
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}


// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}




// 		if(can_schedule){

// 			bool canSchedule[num_clients];
// 			for (int i = 0; i < num_clients; ++i) {
// 				canSchedule[i] = true;
// 				if (event_ids[i] >= 1) {
// 					if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 						// printf("kernel %d from client %d is finished\n", seen[i], i);
// 						unsetmask_nomutex(i);
// 					}
// 					else{
// 						// printf("kernel %d from client %d is not finished\n", seen[i], i);
// 						canSchedule[i] = false; 
// 					}
// 				}
// 			}
// 			if(finish_profile == false){

// 				if(num_client_cur_iters[0] == num_client_cur_iters[1]){

// 					if ((seen[0] < window_start[0] || seen[1] < window_start[1])) { 
// 						for (int client_id : ready_client) {
// 							if(seen[client_id] < window_start[client_id]){
// 								if (frecords[client_id] != NULL) {
// 									schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 									pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 								}
// 							}
// 						}
// 					} 
// 					else {
// 						bool begin_sche = true;
// 						if(seen[0] == window_start[0] && seen[1] == window_start[1]){
// 							if(cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess || cudaEventQuery(*(events[1][event_ids[1] - 1])) != cudaSuccess){
// 								begin_sche = false;
// 							}
// 						}
// 						if(begin_sche){
// 							if(set_time == true){
// 								start_profile = std::chrono::high_resolution_clock::now();
// 								set_time = false;
// 							}
	

// 							bool schedule_BE = false;
// 							if (canSchedule[hp_client] && seen[hp_client] < window_end) {
// 								endT = 0;
// 								if (frecords[hp_client] != NULL) { 
// 									op_info op_info_1 = op_info_vector[hp_client][seen[hp_client]];
// 									int tpc_usage = op_info_1.sm_used / 2;
// 									tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;
// 									if (num_tpcs >= tpc_usage) {
// 										// printf("schedule hp client %d\n", hp_client);
// 										int concurrent_id;
// 										if(hp_client == 0){
// 											concurrent_id = 1;
// 										}
// 										else{
// 											concurrent_id = 0;
// 										}

// 										if(cudaEventQuery(*(events[concurrent_id][event_ids[concurrent_id] - 1])) == cudaSuccess){
// 											scheduleOrder.push_back({hp_client, seen[hp_client] - window_start[hp_client], tpc_usage, -1});
// 										} else {
// 											scheduleOrder.push_back({hp_client, seen[hp_client] - window_start[hp_client], tpc_usage, seen[concurrent_id] - window_start[hp_client] - 1});
// 										}
// 										setmask(client_mutexes[hp_client], tpc_usage, hp_client);
// 										schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
// 										pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
							
// 										int dur = op_info_1.duration;
// 										float threshold = 0.8;
// 										auto current_time = std::chrono::high_resolution_clock::now();
// 										auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time.time_since_epoch()).count();
// 										endT = static_cast<double>(dur) * threshold + static_cast<double>(duration_ns);
// 										schedule_BE = true;                            
// 									}
									
// 								}
// 								else {
// 									hp_client = (hp_client + 1) % num_clients;
// 								}
// 							}

// 							if(schedule_BE){ // low priority
// 								for (int t = 1; t < num_clients; t++) {
// 									int j = (hp_client + t) % num_clients;

// 									// // printf("schedule BE client %d\n", j);
// 									// // printf("------------------\n");
// 									if (frecords[j] != NULL && canSchedule[j] && seen[j] < window_end) {
// 										op_info op_info_0 = op_info_vector[j][seen[j]];
// 										int tpc_usage = op_info_0.sm_used / 2;
// 										tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;
// 										bool schedule = false;

// 										if ((num_clients==1) || (seen[hp_client]==0) || (frecords[j]->type == MALLOC_RECORD) || (frecords[j]->type == MEMCPY_RECORD) || (frecords[j]->type == MEMSET_RECORD) || (frecords[j]->type == FREE_RECORD)) {
// 											schedule = true;
// 										}
// 										else if (num_tpcs > 0 && seen[hp_client] > 0 && ((op_info_0.profile == -1 || profiles[hp_client] == -1 || (profiles[hp_client] != op_info_0.profile)))) {
// 											auto current_time = std::chrono::high_resolution_clock::now();
// 											auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time.time_since_epoch()).count();
// 											if(endT == 0){
// 												schedule = true;
// 											}
// 											else if (duration_ns + op_info_0.duration <= endT) {
// 												schedule = true;
// 												// printf("i can schedule\n");
// 											}
// 										}
// 										if (schedule) {
// 											int concurrent_id;
// 											if(j == 0){
// 												concurrent_id = 1;
// 											}
// 											else{
// 												concurrent_id = 0;
// 											}

// 											if(cudaEventQuery(*(events[concurrent_id][event_ids[concurrent_id] - 1])) == cudaSuccess){
// 												scheduleOrder.push_back({j, seen[j] - window_start[j], tpc_usage, -1});
// 											} else {
// 												scheduleOrder.push_back({j, seen[j] - window_start[j], tpc_usage, seen[j] - window_start[j] - 1});
// 											}
// 											setmask(client_mutexes[j], num_tpcs, j);
// 											schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 											pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 										}
// 									}
// 								}
// 								hp_client = (hp_client + 1) % num_clients;
// 							}
// 						}
// 					}
// 				}
// 			}
// 			else{
// 				bool begin_sche = true;
// 				if(seen[0] == window_end  && seen[1] == window_end){
// 					if(cudaEventQuery(*(events[0][event_ids[0] - 1])) != cudaSuccess || cudaEventQuery(*(events[1][event_ids[1] - 1])) != cudaSuccess){
// 						begin_sche = false;
// 					}
// 				}
// 				if(begin_sche){
// 					for (int j = 0; j < num_clients; ++j) {
// 						if (frecords[j] != NULL) {
// 							if (frecords[j]->type != MALLOC_RECORD && 
// 								frecords[j]->type != MEMCPY_RECORD && 
// 								frecords[j]->type != MEMSET_RECORD && 
// 								frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 									// printf("start executing other kernels\n");
// 									schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 									pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}


// 		int finished = 0;
// 		int readytoprofile = 0;
// 		for (int i=0; i < num_clients; i++) {

// 			if(can_schedule && seen[i] == window_end){
// 				if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 					readytoprofile += 1;
// 				}
// 			}

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					unsetmask_nomutex(i);
// 					set_time = true;
// 					kernelset[i].clear();
// 					finish_profile = false;
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if(readytoprofile == num_clients){
// 			finish_profile = true;
// 			auto end_10 = std::chrono::high_resolution_clock::now();
// 			auto duration_10_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(end_10 - start_profile).count();
// 			recordKRISP(num_client_cur_iters[0] - 10, duration_10_nano, scheduleOrder);
// 			scheduleOrder.clear();
// 			printf(" kernels from %d to %d kernels took %ld nanoseconds\n",window_start[0], window_end, duration_10_nano);
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
//     	auto duration_nano= std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
// 		// duration /= 1000.0;
// 		printf("Total loop took %ld nanoseconds\n", duration_nano);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

// shifting without profile
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {

// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

//     int window_start = 491;
// 	int window_end = 559;
//     int window_first_end_point = 523;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;

// 	auto start_total = std::chrono::high_resolution_clock::now();
// 	auto start_micro = std::chrono::duration_cast<std::chrono::microseconds>(start_total.time_since_epoch());
// 	std::cout << "Start time: " << start_micro.count() << " μs" << std::endl;

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients);
// 	std::vector<int> schedule_client(2, 0);

// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {
//             // if(is_executing[i]==true){
// 			// 	continue;
// 			// }
// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}

//         int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}

// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}


//         // for fused ops
// 		if(can_schedule){
// 			// printf("begin schedule !!!! \n");
//             // printf("seen 0 is %d\n",seen[0]);
//             // printf("seen 1 is %d\n",seen[1]);
//             if (seen[0] < window_start || seen[1] < window_start) {  // Ensure either client is under the threshold
// 				if(ready_client.size() == num_all_clients){
// 					for (int client_id : ready_client) {
// 						if(seen[client_id] < window_start){
// 							if (frecords[client_id] != NULL) {
// 								schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 								pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 							}
// 						}
// 					}
// 				}
// 			}else {
// 				if (seen[0] < window_first_end_point && seen[0] >= window_start) {
// 					if (frecords[0] != NULL) {
//                         schedule_kernel(*(frecords[0]), sched_streams[0], 0, events[0][event_ids[0]], seen, event_ids, 0);
//                         pop_from_queue(client_buffers[0], client_mutexes[0], 0);
						
// 					}
// 				} else {
//                     bool canSchedule[num_clients];
//                     for (int i = 0; i < num_clients; ++i) {
//                         canSchedule[i] = true;
//                         if (event_ids[i] >= 1) {
//                             if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
//                                 // printf("kernel %d finished\n", event_ids[i]);
//                                 unsetmask_nomutex(i);
//                             }
//                             else{
//                                 canSchedule[i] = false; 
//                             }
//                         }
//                     }
//                     for (int j = 0; j < num_clients; ++j) {
//                         if (frecords[j] != NULL && seen[j] <= window_end) {
//                             if(frecords[j]->type != MALLOC_RECORD 
//                             && frecords[j]->type != MEMCPY_RECORD 
//                             && frecords[j]->type != MEMSET_RECORD 
//                             && frecords[j]->type != FREE_RECORD 
//                             ){
//                                 if(canSchedule[j]){
//                                     int tpc_usage = -1;
//                                     op_info &op_info_cur = op_info_vector[j][seen[j]];
//                                     if(op_info_cur.is_critical == 1){
//                                         if(num_tpcs < 17){
//                                             continue;
//                                         }
//                                         tpc_usage = 17; 
//                                     }
//                                     else{
//                                         tpc_usage = 7; 
//                                     }
//                                     if(num_tpcs >= tpc_usage){
//                                         setmask(client_mutexes[j], tpc_usage, j);
//                                     }
//                                     else{
//                                         continue;
//                                     }
//                                     schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
//                                     pop_from_queue(client_buffers[j], client_mutexes[j], j);
                                    
//                                 }
//                             }
//                         }
//                     }
// 				}
// 			}

// 		}

//         //for consecutive ops
//         // if(can_schedule){
//         //     // printf("seen 0 is %d\n",seen[0]);
//         //     // printf("seen 1 is %d\n",seen[1]);
//         //     if (seen[0] < window_start || seen[1] < window_start) {  // Ensure either client is under the threshold
// 		// 		if(ready_client.size() == num_all_clients){
// 		// 			for (int client_id : ready_client) {
// 		// 				if(seen[client_id] < window_start){
// 		// 					if (frecords[client_id] != NULL) {
// 		// 						schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 		// 						pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 		// 					}
// 		// 				}
// 		// 			}
// 		// 		}
// 		// 	} else {
//         //             bool canSchedule[num_clients];
//         //             for (int i = 0; i < num_clients; ++i) {
//         //                 canSchedule[i] = true;
//         //                 if (event_ids[i] >= 1) {
//         //                     if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
//         //                         // printf("kernel %d finished\n", event_ids[i]);
//         //                         unsetmask_nomutex(i);
//         //                     }
//         //                     else{
//         //                         canSchedule[i] = false; 
//         //                     }
//         //                 }
//         //             }

//         //             // if(ready_client.size() == num_all_clients){
//         //                 for (int j = 0; j < num_clients; ++j) {
//         //                     if (frecords[j] != NULL && seen[j] <= window_end) {
//         //                         if(frecords[j]->type != MALLOC_RECORD 
//         //                         && frecords[j]->type != MEMCPY_RECORD 
//         //                         && frecords[j]->type != MEMSET_RECORD 
//         //                         && frecords[j]->type != FREE_RECORD 
//         //                         ){
//         //                             if(canSchedule[j]){
//         //                                 int tpc_usage = -1;
//         //                                 op_info &op_info_cur = op_info_vector[j][seen[j]];
//         //                                 if(op_info_cur.is_critical == 1){
//         //                                     if(num_tpcs < 17){
//         //                                         continue;
//         //                                     }
//         //                                     tpc_usage = 17; 
//         //                                 }
//         //                                 else{
//         //                                     tpc_usage = 7; 
//         //                                 }
//         //                                 if(num_tpcs >= tpc_usage){
//         //                                     setmask(client_mutexes[j], tpc_usage, j);
//         //                                 }
//         //                                 else{
//         //                                     continue;
//         //                                 }
//         //                                 schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
//         //                                 // schedule_kernel_profile(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, op_info_cur, tpc_usage, num_client_cur_iters[j]);
//         //                                 pop_from_queue(client_buffers[j], client_mutexes[j], j);
                                        
//         //                             }
//         //                         }
//         //                     }
//         //                 }
//         //             // }
// 		// 	}
//         // }

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

//             if(can_schedule && seen[i] > window_end){
//                 if (cudaEventQuery(*(events[i][event_ids[i]])) == cudaSuccess){
//                     finished += 1;
//                 }
// 			}
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
//                     unsetmask_nomutex(i);
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		// float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		auto end_micro = std::chrono::duration_cast<std::chrono::microseconds>(end_total.time_since_epoch());
// 		std::cout << "End time: " << end_micro.count() << " μs" << std::endl;
//     	auto duration_micro = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();

// 		// duration /= 1000.0;
// 		printf("Total loop took %ld microseconds\n", duration_micro);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }


// no any shedule
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {

// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

//     int window_start = 4224;
// 	int window_end = 4286;
//     int window_first_end_point = 241;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;

// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients);
// 	std::vector<int> schedule_client(2, 0);

// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}

// 		for(int j = 0; j < num_clients; j++){
// 			if(frecords[j] != NULL){
// 				schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 				pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 			}
// 		}
		

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess){
// 							ready &= false;

// 						}
// 					}
// 				}
// 				if (ready) {
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

//no any schedule with profile
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {

// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int window_start = 4224;
// 	int window_end = 4286;
//     int window_first_end_point = 523;
// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;

// 	auto start_total = std::chrono::high_resolution_clock::now();
// 	auto start_micro = std::chrono::duration_cast<std::chrono::microseconds>(start_total.time_since_epoch());
// 	std::cout << "Start time: " << start_micro.count() << " μs" << std::endl;

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
// 	ThreadPool pool(num_clients);
// 	std::vector<int> schedule_client(2, 0);

// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}

// 				unsetmask_nomutex(i);
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}

// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}
		
// 		// for fused ops
// 		// if(can_schedule){
// 		// 	// printf("begin schedule !!!! \n");
// 		// 	if (seen[0] < window_start || seen[1] < window_start) {  // Ensure either client is under the threshold
// 		// 		if(ready_client.size() == num_all_clients){
// 		// 			for (int client_id : ready_client) {
// 		// 				if(seen[client_id] < window_start){
// 		// 					if (frecords[client_id] != NULL && !is_executing[client_id]) {
// 		// 						is_executing[client_id] = true;
// 		// 						pool.enqueue(&Scheduler::execute_kernel, this, client_id, *(frecords[client_id]));
// 		// 					}
// 		// 				}
// 		// 			}
// 		// 		}
// 		// 	} else {
//         //         // Otherwise, enqueue for all clients
//         //         for (int client_id : ready_client) {
//         //             if (frecords[client_id] != NULL && !is_executing[client_id] && seen[client_id] <= window_end) {
//         //                 int tpc_usage = -1;
//         //                 op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
//         //                 is_executing[client_id] = true;
//         //                 pool.enqueue(&Scheduler::execute_kernel_profile, this, client_id, *(frecords[client_id]), op_info_cur, tpc_usage, num_client_cur_iters[client_id]);
//         //             }
//         //         }
// 		// 	}

// 		// }



// 		// for consecutive ops with kernel profile 
// 		if(can_schedule){
// 			// printf("begin schedule !!!! \n");
// 			if (seen[0] < window_start || seen[1] < window_start) {  // Ensure either client is under the threshold
// 				if(ready_client.size() == num_all_clients){
// 					for (int client_id : ready_client) {
// 						if(seen[client_id] < window_start){
// 							if (frecords[client_id] != NULL && !is_executing[client_id]) {
// 								int tpc_usage = -1;
// 								op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
// 								is_executing[client_id] = true;
// 								pool.enqueue(&Scheduler::execute_kernel, this, client_id, *(frecords[client_id]));
// 							}
// 						}
// 					}
// 				}
// 			} else {
// 					// Otherwise, enqueue for all clients
//                     for (int client_id : ready_client) {
//                         if (frecords[client_id] != NULL && !is_executing[client_id] && seen[client_id] <= window_end) {
//                             int tpc_usage = -1;
//                             op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
//                             is_executing[client_id] = true;
//                             pool.enqueue(&Scheduler::execute_kernel_profile, this, client_id, *(frecords[client_id]), op_info_cur, tpc_usage, num_client_cur_iters[client_id]);
//                         }
//                     }
// 			}
// 		}


// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

//             if(can_schedule && seen[i] > window_end){
//                 if (cudaEventQuery(*(events[i][event_ids[i]])) == cudaSuccess){
//                     finished += 1;
//                 }
// 			}

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		// float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		auto end_micro = std::chrono::duration_cast<std::chrono::microseconds>(end_total.time_since_epoch());
// 		std::cout << "End time: " << end_micro.count() << " μs" << std::endl;
//     	auto duration_micro = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();

// 		// duration /= 1000.0;
// 		printf("Total loop took %ld microseconds\n", duration_micro);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }


//original orion

// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {


// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	int hp_client = num_clients-1;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;
//     int coexe_step =0;
// 	// if hp is inference, use max_sms + also there is no update phase
// 	if (!is_train[hp_client]) {
// 		sm_threshold = max_sms;
// 		update_start = INT_MAX;
// 	}


//     int SM_hp_client = 0;
//     int total_SM = max_sms;
//     long int endT = 0;
// 	while(1) {
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {
// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 				//if (seen[i] == num_client_kernels[i]-1)
// 				//	continue;
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}

//         bool canSchedule[num_clients];
//         for (int i = 0; i < num_clients; ++i) {
//             canSchedule[i] = true;
//             if (event_ids[i] >= 1) {
//                 if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
//                     // printf("kernel %d finished\n", event_ids[i]);
//                     unsetmask_nomutex(i);
//                 }
//                 else{
//                     canSchedule[i] = false; 
//                 }
//             }
//         }

//         bool schedule_BE = false;

//         // hp_client = (hp_client + 1) % num_clients;

//         if (canSchedule[hp_client]) {
//             endT = 0;
//             if (frecords[hp_client] != NULL) { 
//                 op_info op_info_1 = op_info_vector[hp_client][seen[hp_client]];
// 				int tpc_usage = op_info_1.sm_used / 2;
// 				tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 20) ? 16 : tpc_usage;
//                 if (frecords[hp_client]->type != MALLOC_RECORD && 
// 					frecords[hp_client]->type != MEMCPY_RECORD && 
// 					frecords[hp_client]->type != MEMSET_RECORD && 
// 					frecords[hp_client]->type != FREE_RECORD) {
//                         if (num_tpcs >= tpc_usage) {
//                             // printf("schedule hp client %d\n", hp_client);
//                             setmask(client_mutexes[hp_client], tpc_usage, hp_client);
//                             schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
//                             pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
                
//                             int dur = op_info_1.duration;
//                             float threshold = 0.8;
//                             auto current_time = std::chrono::high_resolution_clock::now();
//                             auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time.time_since_epoch()).count();
//                             endT = static_cast<double>(dur) * threshold + static_cast<double>(duration_ns);
//                             schedule_BE = true;                            
//                         }
// 				}
//                 else{
//                     schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
//                     pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
//                 }
//             }
//             else {
//                 hp_client = (hp_client + 1) % num_clients;
//             }
//         }

//         if(schedule_BE){ // low priority
//             for (int t = 1; t < num_clients; t++) {
//                 int j = (hp_client + t) % num_clients;
//                 // printf("schedule BE client %d\n", j);
//                 // printf("------------------\n");
//                 if (frecords[j] != NULL && canSchedule[j]) {
//                     op_info op_info_0 = op_info_vector[j][seen[j]];
//                     int tpc_usage = op_info_0.sm_used / 2;
//                     tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 20) ? 24 : tpc_usage;
//                     bool schedule = false;

//                     if ((num_clients==1) || (seen[hp_client]==0) || (frecords[j]->type == MALLOC_RECORD) || (frecords[j]->type == MEMCPY_RECORD) || (frecords[j]->type == MEMSET_RECORD) || (frecords[j]->type == FREE_RECORD)) {
//                         schedule = true;
//                     }
//                     else if (num_tpcs > 0 && seen[hp_client] > 0 && ((op_info_0.profile == -1 || profiles[hp_client] == -1 || (profiles[hp_client] != op_info_0.profile)))) {
//                         auto current_time = std::chrono::high_resolution_clock::now();
//                         auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time.time_since_epoch()).count();
//                         if(endT == 0){
//                             schedule = true;
//                         }
//                         else if (duration_ns + op_info_0.duration <= endT) {
//                             schedule = true;
//                             // printf("i can schedule\n");
//                         }
//                     }
//                     if (schedule) {
//                         if (frecords[j]->type != MALLOC_RECORD && 
//                             frecords[j]->type != MEMCPY_RECORD && 
//                             frecords[j]->type != MEMSET_RECORD && 
//                             frecords[j]->type != FREE_RECORD) {
//                                 setmask(client_mutexes[j], min(num_tpcs, tpc_usage)  , j);
//                             }
//                         schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
//                         pop_from_queue(client_buffers[j], client_mutexes[j], j);
//                     }
//                 }
//             }
//             hp_client = (hp_client + 1) % num_clients;
//         }
//         else{
//             // schedule memory kernel for BE
//             // even we don't have hp client compute kernel in execution
//             for (int t = 1; t < num_clients; t++) {
//                 int j = (hp_client + t) % num_clients;
//                 if (frecords[j] != NULL) {
//                     if (frecords[j]->type == MALLOC_RECORD && 
//                         frecords[j]->type == MEMCPY_RECORD && 
//                         frecords[j]->type == MEMSET_RECORD && 
//                         frecords[j]->type == FREE_RECORD) {
//                             schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
//                             pop_from_queue(client_buffers[j], client_mutexes[j], j);
//                         }
//                 }
//             }
//         }
		

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				finished += 1;
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
//                     unsetmask_nomutex(i);
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;

// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 					// if (!reef && !seq && i==hp_client && is_train[hp_client]) {
// 					// 	printf("Client %d finished iteration %d, it took %f ms\n", i, num_client_cur_iters[i], duration);
// 					// 	hp_iter_duration += duration;
// 					// 	if ((num_client_cur_iters[i] % 10) == 0 && low_sms != sm_threshold) {
// 					// 		float hp_avg_duration = hp_iter_duration/10.0;
// 					// 		printf("--------------------- Average iter duration for client 1 is %f ms, limit is %f ms, sm_threshold is %d\n", hp_avg_duration, hp_limit_float, sm_threshold);
// 					// 		hp_iter_duration = 0;

// 					// 		// TODO: add better stopping conditions
// 					// 		if (hp_avg_duration > hp_limit_float) {
// 					// 			high_sms = sm_threshold;
// 					// 			sm_threshold = (low_sms+high_sms)/2;
// 					// 		}
// 					// 		else {
// 					// 			low_sms = sm_threshold;
// 					// 			sm_threshold = (low_sms+high_sms)/2;
// 					// 		}
// 					// 	}
// 					// }
// 					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
// 				}
// 				if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				) {
// 					finished += 1;
// 					if (!warmup) {
// 						auto end_total = std::chrono::high_resolution_clock::now();
// 						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
// 						duration /= 1000.0;
// 						printf("Client %d, Total loop took %f sec\n", i, duration);
// 						// if (i==num_clients-1) {
// 						// 	for (int k=0; k<num_clients-1; k++) {
// 						// 		printf("======= Client %d has done %d iterations\n", k, num_client_cur_iters[k]);
// 						// 		if (!locked[k])
// 						// 			pthread_mutex_lock(client_mutexes[k]);
// 						// 		stops[k] = true;
// 						// 		if (!locked[k])
// 						// 			pthread_mutex_unlock(client_mutexes[k]);
// 						// 	}
// 						// }
// 					}
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;

// 	}
// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
//         printf("Total co-exe step %d\n", coexe_step);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

// no any schedule
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {


// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;
// 	int hp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;

// 	// if hp is inference, use max_sms + also there is no update phase
// 	if (!is_train[hp_client]) {
// 		sm_threshold = max_sms;
// 		update_start = INT_MAX;
// 	}


//     int SM_hp_client = 0;
//     int total_SM = max_sms;
//     long int endT = 0;
// 	ThreadPool pool(num_clients);
// 	while(1) {
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {
// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 				//if (seen[i] == num_client_kernels[i]-1)
// 				//	continue;
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		bool canSchedule[num_clients];
//         for (int i = 0; i < num_clients; ++i) {
//             canSchedule[i] = true;
//             if (event_ids[i] >= 1) {
//                 if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
//                     // printf("kernel %d finished\n", event_ids[i]);
//                     // unsetmask_nomutex(i);
//                     unsetmask_O(i);
//                 }
//                 else{
//                     canSchedule[i] = false; 
//                 }
//             }
//         }

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}

// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}

// 		if(can_schedule){
			
// 			for (int i = 0; i < num_clients; ++i) {
// 				// int j = (hp_client + i) % num_clients;
// 				int j = i;
// 				if (frecords[j] != NULL) {
// 					if (frecords[j]->type != MALLOC_RECORD && 
// 						frecords[j]->type != MEMCPY_RECORD && 
// 						frecords[j]->type != MEMSET_RECORD && 
// 						frecords[j]->type != FREE_RECORD) {
// 							if(canSchedule[j]){
// 								op_info op_info_1 = op_info_vector[j][seen[j]];
// 								int tpc_usage = 0;


// 								// if(op_info_1.knee_tpc < 12){
// 								if(op_info_1.name == "void at::native::batch_norm_transform_input_kernel"){
// 									// printf("name of kernel %s\n", op_info_1.name.c_str());
// 									setmask_O(client_mutexes[j], op_info_1.knee_tpc, j);
// 									schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 									pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 									continue;
// 								}

// 								// if(num_tpcs > 0){
// 									// tpc_usage = min(op_info_1.knee_tpc, num_tpcs);
// 									// setmask(client_mutexes[j], tpc_usage, j);
// 									// schedule_kernel_profile(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, op_info_1, tpc_usage, num_client_cur_iters[j]);
// 									schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 									pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 								// }
// 							}
// 					}
// 					else{
// 						schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 						pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 					}
// 				}
// 			}
// 			// hp_client = (hp_client + 1) % num_clients;
// 		}
// 		else{
// 			for (int j = 0; j < num_clients; ++j) {
// 				if (frecords[j] != NULL && canSchedule[j]) {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}


// 		// if(can_schedule){
			
// 		// 	for (int i = 0; i < num_clients; ++i) {
// 		// 		// int j = (hp_client + i) % num_clients;
// 		// 		int j = i;
// 		// 		if (frecords[j] != NULL) {
// 		// 			if (frecords[j]->type != MALLOC_RECORD && 
// 		// 				frecords[j]->type != MEMCPY_RECORD && 
// 		// 				frecords[j]->type != MEMSET_RECORD && 
// 		// 				frecords[j]->type != FREE_RECORD) {
// 		// 					if(canSchedule[j]){
// 		// 						op_info op_info_1 = op_info_vector[j][seen[j]];
// 		// 						int tpc_usage = 24;
// 		// 						setmask(client_mutexes[j], tpc_usage, j);
// 		// 						schedule_kernel_profile(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, op_info_1, tpc_usage, num_client_cur_iters[j]);
// 		// 						pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 		// 					}
// 		// 			}
// 		// 			else{
// 		// 				schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 		// 				pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 		// 			}
// 		// 		}
// 		// 	}
// 		// 	// hp_client = (hp_client + 1) % num_clients;
// 		// }
// 		// else{
// 		// 	for (int j = 0; j < num_clients; ++j) {
// 		// 		if (frecords[j] != NULL && canSchedule[j]) {
// 		// 			schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 		// 			pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 		// 		}
// 		// 	}
// 		// }

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				finished += 1;
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// unsetmask_nomutex(i);
// 					unsetmask_O(i);
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;

// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 					// if (!reef && !seq && i==hp_client && is_train[hp_client]) {
// 					// 	printf("Client %d finished iteration %d, it took %f ms\n", i, num_client_cur_iters[i], duration);
// 					// 	hp_iter_duration += duration;
// 					// 	if ((num_client_cur_iters[i] % 10) == 0 && low_sms != sm_threshold) {
// 					// 		float hp_avg_duration = hp_iter_duration/10.0;
// 					// 		printf("--------------------- Average iter duration for client 1 is %f ms, limit is %f ms, sm_threshold is %d\n", hp_avg_duration, hp_limit_float, sm_threshold);
// 					// 		hp_iter_duration = 0;

// 					// 		// TODO: add better stopping conditions
// 					// 		if (hp_avg_duration > hp_limit_float) {
// 					// 			high_sms = sm_threshold;
// 					// 			sm_threshold = (low_sms+high_sms)/2;
// 					// 		}
// 					// 		else {
// 					// 			low_sms = sm_threshold;
// 					// 			sm_threshold = (low_sms+high_sms)/2;
// 					// 		}
// 					// 	}
// 					// }
// 					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
// 				}
// 				if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				) {
// 					finished += 1;
// 					if (!warmup) {
// 						auto end_total = std::chrono::high_resolution_clock::now();
// 						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
// 						duration /= 1000.0;
// 						printf("Client %d, Total loop took %f sec\n", i, duration);
// 						// if (i==num_clients-1) {
// 						// 	for (int k=0; k<num_clients-1; k++) {
// 						// 		printf("======= Client %d has done %d iterations\n", k, num_client_cur_iters[k]);
// 						// 		if (!locked[k])
// 						// 			pthread_mutex_lock(client_mutexes[k]);
// 						// 		stops[k] = true;
// 						// 		if (!locked[k])
// 						// 			pthread_mutex_unlock(client_mutexes[k]);
// 						// 	}
// 						// }
// 					}
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;

// 	}
// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

// KRISP
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {


// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	int hp_client = num_clients-1;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;

// 	// if hp is inference, use max_sms + also there is no update phase
// 	if (!is_train[hp_client]) {
// 		sm_threshold = max_sms;
// 		update_start = INT_MAX;
// 	}

// 	ThreadPool pool(num_clients);
//     int SM_hp_client = 0;
//     int total_SM = max_sms;
//     long int endT = 0;

// 	std::set<int> client_sets[num_clients];

// 	while(1) {
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {
// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 				//if (seen[i] == num_client_kernels[i]-1)
// 				//	continue;
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}

// 		bool canSchedule[num_clients];
		
// 		for (int i = 0; i < num_clients; ++i) {
// 			canSchedule[i] = true;
// 			if (event_ids[i] >= 1) {
// 				if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 					unsetmask_nomutex(i);
// 				}
// 				else {
// 					canSchedule[i] = false;
// 				}
// 			}
// 		}



// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}

// 		if(can_schedule){
// 			for (int j = 0; j < num_clients; ++j) {
// 				if (frecords[j] != NULL) {
// 					if (frecords[j]->type != MALLOC_RECORD && 
// 						frecords[j]->type != MEMCPY_RECORD && 
// 						frecords[j]->type != MEMSET_RECORD && 
// 						frecords[j]->type != FREE_RECORD) {
// 							if(canSchedule[j]){
// 								op_info op_info_1 = op_info_vector[j][seen[j]];
// 								// int tpc_usage = op_info_1.sm_used / 2;
// 								// tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;
// 								if(num_tpcs > 0){
// 									int tpc_usage = min(op_info_1.knee_tpc, num_tpcs);
// 									setmask(client_mutexes[j], tpc_usage, j);
// 									schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 								}
// 							}
// 						}
// 					else{
// 						schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 						pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 					}
// 				}
// 			}
// 		}
// 		else{
// 			for (int j = 0; j < num_clients; ++j) {
// 				if (frecords[j] != NULL && canSchedule[j]) {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		// KRISP-O
//         // for (int j = 0; j < num_clients; ++j) {
// 		// 	if (frecords[j] != NULL) {
//         //         if (frecords[j] != NULL) {
//         //             if (frecords[j]->type != MALLOC_RECORD && 
//         //                 frecords[j]->type != MEMCPY_RECORD && 
//         //                 frecords[j]->type != MEMSET_RECORD && 
//         //                 frecords[j]->type != FREE_RECORD) {
//         //                     if(canSchedule[j]){
//         //                         op_info op_info_1 = op_info_vector[j][seen[j]];
//         //                         // int tpc_usage = op_info_1.sm_used / 2;
//         //                         // tpc_usage = (tpc_usage < 1) ? 1 : (tpc_usage > 24) ? 24 : tpc_usage;
//         //                         int tpc_usage = op_info_1.knee_tpc;
//         //                         setmask_O(client_mutexes[j], tpc_usage, j);
//         //                         schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
//         //                         pop_from_queue(client_buffers[j], client_mutexes[j], j);
//         //                     }
//         //                 }
//         //             else{
//         //                 schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
//         //                 pop_from_queue(client_buffers[j], client_mutexes[j], j);
//         //             }
//         //         }
// 		// 	}
// 		// }
		

// 		int finished = 0;
// 		int stop_and_profile = 0;
// 		for (int i=0; i<num_clients; i++) {

//             if(num_client_cur_iters[i] > 9 && seen[i] >= 50){
//                 if (cudaEventQuery(*(events[i][event_ids[i]])) == cudaSuccess){
// 					stop_and_profile +=1;
//                 }
// 			}

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				finished += 1;
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
//                     unsetmask_nomutex(i);
//                     // unsetmask_O(i);
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;

// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 					// if (!reef && !seq && i==hp_client && is_train[hp_client]) {
// 					// 	printf("Client %d finished iteration %d, it took %f ms\n", i, num_client_cur_iters[i], duration);
// 					// 	hp_iter_duration += duration;
// 					// 	if ((num_client_cur_iters[i] % 10) == 0 && low_sms != sm_threshold) {
// 					// 		float hp_avg_duration = hp_iter_duration/10.0;
// 					// 		printf("--------------------- Average iter duration for client 1 is %f ms, limit is %f ms, sm_threshold is %d\n", hp_avg_duration, hp_limit_float, sm_threshold);
// 					// 		hp_iter_duration = 0;

// 					// 		// TODO: add better stopping conditions
// 					// 		if (hp_avg_duration > hp_limit_float) {
// 					// 			high_sms = sm_threshold;
// 					// 			sm_threshold = (low_sms+high_sms)/2;
// 					// 		}
// 					// 		else {
// 					// 			low_sms = sm_threshold;
// 					// 			sm_threshold = (low_sms+high_sms)/2;
// 					// 		}
// 					// 	}
// 					// }
// 					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
// 				}
// 				if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				) {
// 					finished += 1;
// 					if (!warmup) {
// 						auto end_total = std::chrono::high_resolution_clock::now();
// 						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
// 						duration /= 1000.0;
// 						printf("Client %d, Total loop took %f sec\n", i, duration);
// 						// if (i==num_clients-1) {
// 						// 	for (int k=0; k<num_clients-1; k++) {
// 						// 		printf("======= Client %d has done %d iterations\n", k, num_client_cur_iters[k]);
// 						// 		if (!locked[k])
// 						// 			pthread_mutex_lock(client_mutexes[k]);
// 						// 		stops[k] = true;
// 						// 		if (!locked[k])
// 						// 			pthread_mutex_unlock(client_mutexes[k]);
// 						// 	}
// 						// }
// 					}
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;

// 	}
// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
// 		// duration /= 1000.0;
// 		printf("Total loop took %f nanoseconds\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

// Reef
void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {


	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
	int start0 = 0;
	int start1 = 0;

	int prev_large = -1;
	int hp_running = -1;

	bool inf_finished = false;
	bool started = false;
 	std::chrono::time_point<std::chrono::system_clock> start_time;
	auto start_total = std::chrono::high_resolution_clock::now();

	vector<bool> total_client_set(num_clients, false);
	vector<int> profiles(num_clients, -1);
	vector<int> cur_sms(num_clients, -1);
	int hp_client = num_clients-1;

	bool large_found = false;
	long sum = 0; // sum of durations of ongoing BE kernels
	long size = 0; // sum of sizes of in-the-queues BE kernels
	int start = -1;

	// BS - works only for 2 clients for now
	// TODO: check this
	int low_sms = 0;
	int high_sms = max_sms_clients[0]; // 0 is the lp client
	int sm_threshold = max_sms_clients[0]/2;
	float hp_iter_duration = 0.0; // 1 is the hp client
	float hp_limit_float = (float)hp_limit;

	// if hp is inference, use max_sms + also there is no update phase
	if (!is_train[hp_client]) {
		sm_threshold = max_sms;
		update_start = INT_MAX;
	}

	while(1) {
		vector<func_record*> frecords(num_clients, NULL);
		size = 0;

		for (int i=0; i<num_clients; i++) {
			if (seen[i] == num_client_kernels[i])
				continue;

			pthread_mutex_lock(client_mutexes[i]);
			volatile int sz = client_buffers[i]->size();
			if (sz > 0) {
				frecords[i] = &(client_buffers[i]->front());
				int cur_iter = num_client_cur_iters[i];
				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					client_starts[i] = std::chrono::high_resolution_clock::now();
					client_starts_set[i][cur_iter] = true;
					if (!total_client_set[i]) {
						total_client_starts[i] = std::chrono::high_resolution_clock::now();
						total_client_set[i] = true;
					}
				}
				//if (seen[i] == num_client_kernels[i]-1)
				//	continue;
			}
			pthread_mutex_unlock(client_mutexes[i]);
		}

        hp_client = (hp_client + 1) % num_clients;
        schedule_reef(frecords, num_clients, depth, hp_client);
		
		int finished = 0;
		for (int i=0; i<num_clients; i++) {

			if (
				(num_client_cur_iters[i] == num_client_max_iters[i])
				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
				|| (stop_ack[i] == true)
			)
				finished += 1;
			else if (seen[i] == num_client_kernels[i]) {
				// check if GPU work for this client has finished
				if (!locked[i]) {
					pthread_mutex_lock(client_mutexes[i]);
					locked[i] = true;
					DEBUG_PRINT("LOCK CLIENT %d\n", i);
				}
				bool ready = true;
				if (seq) {
					if (event_ids[0] >= 1) {
						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
							ready &= false;
					}
				}
				else {
					if (event_ids[i] >= 1) {
						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
							ready &= false;
					}
				}
				if (ready) {
                    unsetmask_nomutex(i);
					// if yes, reset meta-structures for this client, and let it continue
					seen[i] = 0;
					if (seq)
						event_ids[0] = 0;
					event_ids[i] = 0;
					streams[i] = -1;
					fidx[i] = 0;
					request_status[i][num_client_cur_iters[i]] = true;
					//printf("UNLOCK CLIENT %d\n", i);
					pthread_mutex_unlock(client_mutexes[i]);
					num_client_cur_iters[i] += 1;
					locked[i] = false;

					auto end = std::chrono::high_resolution_clock::now();
					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
					duration /= 1000.0;
					client_durations[i].push_back(duration);
					
					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
				}
				if (
					(num_client_cur_iters[i] == num_client_max_iters[i])
					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
					|| (stop_ack[i] == true)
				) {
					finished += 1;
					if (!warmup) {
						auto end_total = std::chrono::high_resolution_clock::now();
						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
						duration /= 1000.0;
						printf("Client %d, Total loop took %f sec\n", i, duration);
					}
				}
			}
		}

		if (finished==num_clients)
			break;

	}
	if (!warmup) {
		auto end_total = std::chrono::high_resolution_clock::now();
		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
		duration /= 1000.0;
		printf("Total loop took %f sec\n", duration);
		//process_eval(client_durations);
	}

	return NULL;
}




// orion base
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {


// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	int hp_client = num_clients-1;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;

// 	// if hp is inference, use max_sms + also there is no update phase
// 	if (!is_train[hp_client]) {
// 		sm_threshold = max_sms;
// 		update_start = INT_MAX;
// 	}

// 	while(1) {
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {
// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 				//if (seen[i] == num_client_kernels[i]-1)
// 				//	continue;
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}

// 		if (frecords[hp_client] != NULL) { // high priority

// 			op_info op_info_1 = op_info_vector[hp_client][seen[hp_client]];
// 			schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
// 			streams[hp_client] = 1;
// 			profiles[hp_client] = op_info_1.profile;
// 			cur_sms[hp_client] = op_info_1.sm_used;

// 			status = 1;
// 			pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
// 		}
// 		//start = -1;
// 		int end = start + num_clients; // start+1+num_clients-1
// 		for (int t=start+1; t<end; t++) {
// 			// Do round-robin for the BE clients
// 			int j = t % (num_clients-1);
// 			if (frecords[j] != NULL) { // low priority
// 				op_info op_info_0 = op_info_vector[j][seen[j]];
// 				bool schedule = false;

// 				//printf("%d, %d, %d\n", low_sms, high_sms, sm_threshold);

// 				if ((num_clients==1) || (seen[hp_client]==0) || (frecords[j]->type == MALLOC_RECORD) || (frecords[j]->type == MEMCPY_RECORD) || (frecords[j]->type == MEMSET_RECORD) || (frecords[j]->type == FREE_RECORD))
// 					schedule = true;
// 				else if (num_client_cur_iters[j] <= 10 || num_client_cur_iters[hp_client] >= num_client_max_iters[hp_client]) {
// 					schedule = true;
// 				}
// 				else if (seen[hp_client] >= update_start && (op_info_0.sm_used <= sm_threshold && cudaEventQuery(*(events[hp_client][update_start-1])) == cudaSuccess)) // && (op_info_0.sm_used <= 10*sm_threshold))
// 					schedule = true;
// 				else if (seen[hp_client]>0 && (size + op_info_0.sm_used <= sm_threshold) &&  ((op_info_0.profile == -1 || profiles[hp_client]==-1 || (profiles[hp_client] != op_info_0.profile))))
// 					schedule = true;
// 				if (schedule && large_found) {
// 					bool do_schedule = true;
// 					for (int k=0; k<num_clients-1; k++) {
// 						if (event_ids[k]>=1) {
// 							cudaError_t status = cudaEventQuery(*(events[k][event_ids[k]-1]));
// 							if (status != cudaSuccess) {
// 								do_schedule = false;
// 								break;
// 							}
// 						}
// 					}
// 					if (do_schedule) {
// 						large_found = false;
// 						sum = 0;
// 					}
// 					else
// 						schedule = false;
// 				}
// 				if (schedule) {
// 					//if (op_info_0.duration > depth && num_client_cur_iters[1] < num_client_max_iters[1] && seen[1]==0) {
// 						//block = true;
// 					size += op_info_0.sm_used;
// 					if ((frecords[j]->type != MALLOC_RECORD) && (frecords[j]->type != MEMCPY_RECORD) && (frecords[j]->type != MEMSET_RECORD) && (frecords[j]->type != FREE_RECORD))
// 						sum += op_info_0.duration;
// 					if (sum > depth && num_client_cur_iters[hp_client] < num_client_max_iters[hp_client]) {
// 						large_found = true;
// 					}
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					status = 0;
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);

// 					streams[j] = 0;
// 					start = j;
// 				}
// 			}
// 		}
		

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				finished += 1;
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;

// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 					if (!reef && !seq && i==hp_client && is_train[hp_client]) {
// 						printf("Client %d finished iteration %d, it took %f ms\n", i, num_client_cur_iters[i], duration);
// 						hp_iter_duration += duration;
// 						if ((num_client_cur_iters[i] % 10) == 0 && low_sms != sm_threshold) {
// 							float hp_avg_duration = hp_iter_duration/10.0;
// 							printf("--------------------- Average iter duration for client 1 is %f ms, limit is %f ms, sm_threshold is %d\n", hp_avg_duration, hp_limit_float, sm_threshold);
// 							hp_iter_duration = 0;

// 							// TODO: add better stopping conditions
// 							if (hp_avg_duration > hp_limit_float) {
// 								high_sms = sm_threshold;
// 								sm_threshold = (low_sms+high_sms)/2;
// 							}
// 							else {
// 								low_sms = sm_threshold;
// 								sm_threshold = (low_sms+high_sms)/2;
// 							}
// 						}
// 					}
// 					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
// 				}
// 				if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				) {
// 					finished += 1;
// 					if (!warmup) {
// 						auto end_total = std::chrono::high_resolution_clock::now();
// 						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
// 						duration /= 1000.0;
// 						printf("Client %d, Total loop took %f sec\n", i, duration);
// 						if (i==num_clients-1) {
// 							for (int k=0; k<num_clients-1; k++) {
// 								printf("======= Client %d has done %d iterations\n", k, num_client_cur_iters[k]);
// 								if (!locked[k])
// 									pthread_mutex_lock(client_mutexes[k]);
// 								stops[k] = true;
// 								if (!locked[k])
// 									pthread_mutex_unlock(client_mutexes[k]);
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;

// 	}
// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }

// test schedule for my algorithm
// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {
		
// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	string filename = "/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_small_models/kernel_groups/R1net_8_groups.csv";
//     vector<string> headers;
//     vector<Scheduler::KernelData> kernels = readCSV(filename, headers);

// 	unordered_map<string, unordered_map<int, int>> model_to_cluster_map;

// 	for (const auto& kernel : kernels) {
// 		model_to_cluster_map[kernel.Model][kernel.Kernel_ID] = kernel.Cluster;
// 	}

// 	std::vector<std::string> model_names = {"R1net_8", "R1net_8", "R1net_8", "R1net_8"};
// 	// std::vector<std::string> model_names = {"R1net_32", "R1net_32"};
// 	unordered_map<int, unordered_map<int, int>> kernel_cluster_map;

//     for (int i = 0; i < model_names.size(); i++) {
//         kernel_cluster_map[i] = model_to_cluster_map[model_names[i]];
//     }

// 	int num_groups = 0;
// 	for (const auto& kernel : kernels) {
// 		if (kernel.Cluster > num_groups) {
// 			num_groups = kernel.Cluster;  
// 		}
// 	}
// 	num_groups += 1;

// 	string filename_cm = "/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_small_models/contention_matrix/contention_matrix_R1net_8.csv";
//     map<pair<int, int>, double> contention_map = readContentionMatrix(filename_cm);

// 	// for (const auto& entry : contention_map) {
//     //     // entry.first is the pair (group_1, group_2)
//     //     // entry.second is the contention factor
//     //     cout << "Group " << entry.first.first << " and Group " << entry.first.second
//     //          << " -> Contention Factor: " << entry.second << endl;
//     // }

// 	// abort();
// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;

// 	auto start_total = std::chrono::high_resolution_clock::now();
// 	auto start_micro = std::chrono::duration_cast<std::chrono::microseconds>(start_total.time_since_epoch());
// 	// std::cout << "Start time: " << start_micro.count() << " μs" << std::endl;
	
// 	std::chrono::time_point<std::chrono::system_clock> start_profile;

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;

// 	int hp_client = 1;
// 	int lp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;


//     int num_kernels[num_clients];

//     // Calculate the number of kernels for each client
//     for (int i = 0; i < num_clients; i++) {
//         num_kernels[i] = (num_client_max_iters[i] - 10) * num_client_kernels[i];
//         printf("Client %d: num of total kernels = %d\n", i, num_kernels[i]);
//     }


// 	while(1) {
		
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {

// 			if(is_executing[i]==true){
// 				continue;
// 			}

// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		int num_all_clients = num_clients;	
// 		vector<int> ready_client;

// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				)
// 				{
// 					num_all_clients-=1;
						
// 				}
// 		}

// 		for (int j = 0; j < num_clients; ++j) {
// 			if (frecords[j] != NULL) {
// 				if (frecords[j]->type != MALLOC_RECORD && 
// 					frecords[j]->type != MEMCPY_RECORD && 
// 					frecords[j]->type != MEMSET_RECORD && 
// 					frecords[j]->type != FREE_RECORD && num_client_cur_iters[j] > 9) {
// 						ready_client.push_back(j);
// 				}
// 				else {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}
		
// 		if(can_schedule){


// 			// std::sort(ready_client.begin(), ready_client.end(), [&num_kernels](int a, int b) {
// 			// 	return num_kernels[a] > num_kernels[b];  
// 			// });

// 			bool canSchedule[num_clients];
// 			for (int i = 0; i < num_clients; ++i) {
// 				canSchedule[i] = true;
// 				if (event_ids[i] >= 1) {
// 					if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 						// unsetmask_nomutex(i);
// 						unsetmask_m(i);
// 					}
// 					else{
// 						canSchedule[i] = false; 
// 					}
// 				}
// 			}

// 			// // printf("ready_client size:%ld\n", ready_client.size());
// 			std::set<int> critical_clients_set;
// 			for (int client_id : ready_client) {
// 				op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
// 				if (op_info_cur.is_critical == 1 ) {
// 					critical_clients_set.insert(client_id);
// 				}
// 			}

// 			if (critical_clients_set.size() > 1) {
// 				auto it = critical_clients_set.begin(); 
// 				int first_client = *it; 
// 				int second_client = -1;
// 				double max_contention = -1;
			
// 				// Compare every pair of clients in the critical set
// 				for (auto it1 = critical_clients_set.begin(); it1 != critical_clients_set.end(); ++it1) {
// 					int client1 = *it1;
// 					for (auto it2 = std::next(it1); it2 != critical_clients_set.end(); ++it2) {
// 						int client2 = *it2;
// 						int client_1_group = kernel_cluster_map[client1][seen[client1]];
// 						int client_2_group = kernel_cluster_map[client2][seen[client2]];
// 						pair<int, int> key = {client_1_group, client_2_group};
// 						double contention = contention_map[key];
// 						if (contention > max_contention) {
// 							max_contention = contention;
// 							first_client = client1;
// 							second_client = client2;
// 						}
// 					}
// 				}
			
// 				// If two clients with the highest contention are found, keep them in the set
// 				if (second_client != -1) {
// 					for (auto it = critical_clients_set.begin(); it != critical_clients_set.end(); ) {
// 						if (*it != first_client && *it != second_client) {
// 							it = critical_clients_set.erase(it);
// 						} else {
// 							++it;
// 						}
// 					}
// 				}
// 			}

			

// 			// if (critical_clients_set.size() > 1) {
// 			// 	auto it = critical_clients_set.begin(); 
// 			// 	int first_client = *it; 
// 			// 	int best_client = -1;
// 			// 	double min_contention = -1;
		
// 			// 	for (auto it2 = std::next(it); it2 != critical_clients_set.end(); ++it2) {
// 			// 		int other_client = *it2;
// 			// 		int client_1_group = kernel_cluster_map[first_client][seen[first_client]];
// 			// 		int client_2_group = kernel_cluster_map[other_client][seen[other_client]];
// 			// 		pair<int, int> key = {client_1_group, client_2_group};
// 			// 		double contention = contention_map[key];
// 			// 		if (contention > min_contention && contention > 1) {
// 			// 			min_contention = contention;
// 			// 			best_client = other_client;
// 			// 		}
// 			// 	}
			
// 			// 	if (best_client != -1) {
// 			// 		for (auto it = critical_clients_set.begin(); it != critical_clients_set.end(); ) {
// 			// 			if (*it != first_client && *it != best_client) {
// 			// 				it = critical_clients_set.erase(it);
// 			// 			} else {
// 			// 				++it;
// 			// 			}
// 			// 		}
// 			// 	}
// 			// }


// 			for (int client_id : ready_client) {

// 				if (frecords[client_id] != NULL) {

// 					op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];

// 					if(critical_clients_set.count(client_id) ){

// 						if(critical_clients_set.size() > 1){
// 							int tpc_usage = 24;
// 							if(num_tpcs >= tpc_usage){
// 								setmask_m(client_mutexes[client_id], tpc_usage, client_id, 2);
// 								schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 								pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 								num_kernels[client_id] --;
// 							}
// 						}
// 						else{
// 							int tpc_usage = op_info_cur.knee_tpc;
// 							if(num_tpcs >= tpc_usage && canSchedule[client_id] == true){
// 								if(num_all_clients == 1){
// 									tpc_usage = 24;
// 								}
// 								setmask_m(client_mutexes[client_id], tpc_usage, client_id, 1);
// 								schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 								pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 								num_kernels[client_id] --;
// 							}
// 						}
// 					}

// 					if(op_info_cur.is_critical == 0 && canSchedule[client_id] == true){
// 							int tpc_usage = op_info_cur.knee_tpc;
// 							if(num_all_clients == 1){
// 								tpc_usage = 24;
// 							}
// 							// if(num_tpcs >= tpc_usage){
// 								setmask_O(client_mutexes[client_id], tpc_usage, client_id);
// 								// setmask_m(client_mutexes[client_id], tpc_usage, client_id, 1);
// 								schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 								pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 								num_kernels[client_id] --;
// 							// }

							
// 					}
							

// 					// if(num_tpcs > 0 && canSchedule[client_id] == true){
// 					// 	int tpc_usage = op_info_cur.knee_tpc;
// 					// 	tpc_usage = min(num_tpcs, tpc_usage); 
// 					// 	setmask(client_mutexes[client_id], tpc_usage, client_id);
// 					// 	schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 					// 	pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 					// 	num_kernels[client_id] --;
// 					// }
					

// 					// if (canSchedule[client_id] == true) {
// 					// 	op_info &op_info_cur = op_info_vector[client_id][seen[client_id]];
// 					// 	int tpc_usage = num_client_cur_iters[client_id] - 9;
// 					// 	setmask(client_mutexes[client_id], tpc_usage, client_id);
// 					// 	schedule_kernel_profile(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id, op_info_cur, tpc_usage, num_client_cur_iters[client_id]);
// 					// 	pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);
// 					// 	num_kernels[client_id] --;
// 					// }

					

// 					// setmask(client_mutexes[client_id], 6, client_id);
// 					// schedule_kernel(*(frecords[client_id]), sched_streams[client_id], client_id, events[client_id][event_ids[client_id]], seen, event_ids, client_id);
// 					// pop_from_queue(client_buffers[client_id], client_mutexes[client_id], client_id);		
// 					// num_kernels[client_id] --;
							
// 				}
// 			}
// 		}


// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {

// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				{
// 					finished += 1;
// 				}
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					// printf("finish %d\n",i)
// 					unsetmask_m(i);
// 					// unsetmask_nomutex(i);
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;
// 					client_progress[i] = 0;
// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;
// 	}



// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
//     	auto duration= std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %ld nanoseconds\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }



// void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int depth, int hp_limit, int update_start) {


// 	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
// 	int start0 = 0;
// 	int start1 = 0;

// 	int prev_large = -1;
// 	int hp_running = -1;

// 	bool inf_finished = false;
// 	bool started = false;
//  	std::chrono::time_point<std::chrono::system_clock> start_time;
// 	auto start_total = std::chrono::high_resolution_clock::now();

// 	vector<bool> total_client_set(num_clients, false);
// 	vector<int> profiles(num_clients, -1);
// 	vector<int> cur_sms(num_clients, -1);
// 	// int hp_client = num_clients-1;
// 	int hp_client = 0;

// 	bool large_found = false;
// 	long sum = 0; // sum of durations of ongoing BE kernels
// 	long size = 0; // sum of sizes of in-the-queues BE kernels
// 	int start = -1;

// 	// BS - works only for 2 clients for now
// 	// TODO: check this
// 	int low_sms = 0;
// 	int high_sms = max_sms_clients[0]; // 0 is the lp client
// 	int sm_threshold = max_sms_clients[0]/2;
// 	float hp_iter_duration = 0.0; // 1 is the hp client
// 	float hp_limit_float = (float)hp_limit;

// 	// if hp is inference, use max_sms + also there is no update phase
// 	if (!is_train[hp_client]) {
// 		sm_threshold = max_sms;
// 		update_start = INT_MAX;
// 	}


//     int SM_hp_client = 0;
//     int total_SM = max_sms;
//     long int endT = 0;
// 	ThreadPool pool(num_clients);
// 	while(1) {
// 		vector<func_record*> frecords(num_clients, NULL);
// 		size = 0;

// 		for (int i=0; i<num_clients; i++) {
// 			if (seen[i] == num_client_kernels[i])
// 				continue;

// 			pthread_mutex_lock(client_mutexes[i]);
// 			volatile int sz = client_buffers[i]->size();
// 			if (sz > 0) {
// 				frecords[i] = &(client_buffers[i]->front());
// 				int cur_iter = num_client_cur_iters[i];
// 				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
// 					client_starts[i] = std::chrono::high_resolution_clock::now();
// 					client_starts_set[i][cur_iter] = true;
// 					if (!total_client_set[i]) {
// 						total_client_starts[i] = std::chrono::high_resolution_clock::now();
// 						total_client_set[i] = true;
// 					}
// 				}
// 				//if (seen[i] == num_client_kernels[i]-1)
// 				//	continue;
// 			}
// 			pthread_mutex_unlock(client_mutexes[i]);
// 		}


// 		bool can_schedule = true;
// 		for (int j = 0; j < num_clients; ++j) {
// 			if(num_client_cur_iters[j] < 10) {
// 				can_schedule = false;
// 			}
// 		}

// 		bool canSchedule[num_clients];
// 		for (int i = 0; i < num_clients; ++i) {
// 			canSchedule[i] = true;
// 			if (event_ids[i] >= 1) {
// 				if (cudaEventQuery(*(events[i][event_ids[i] - 1])) == cudaSuccess) {
// 					// printf("kernel %d finished\n", event_ids[i]);
// 					unsetmask_nomutex(i);
// 					// unsetmask_O(i);
// 				}
// 				else{
// 					canSchedule[i] = false; 
// 				}
// 			}
// 		}

// 		if(can_schedule){

// 			for (int i = 0; i < num_clients; ++i) {
// 				int j = i;
// 				if (frecords[j] != NULL) {
// 					if (frecords[j]->type != MALLOC_RECORD && 
// 						frecords[j]->type != MEMCPY_RECORD && 
// 						frecords[j]->type != MEMSET_RECORD && 
// 						frecords[j]->type != FREE_RECORD) {
// 							if(canSchedule[j]){
// 								op_info op_info_1 = op_info_vector[j][seen[j]];
// 								int tpc_usage = 0;

// 								if(op_info_1.knee_tpc < 12){
// 								// if(op_info_1.name == "void at::native::batch_norm_transform_input_kernel"){
// 									// printf("name of kernel %s\n", op_info_1.name.c_str());
// 									setmask_O(client_mutexes[j], op_info_1.knee_tpc, j);
// 									schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 									pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 									continue;
// 								}

// 								if(num_tpcs > 0){
// 									tpc_usage = min(op_info_1.knee_tpc, num_tpcs);
// 									setmask(client_mutexes[j], tpc_usage, j);
// 									// schedule_kernel_profile(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j, op_info_1, tpc_usage, num_client_cur_iters[j]);
// 									schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 									pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 								}
// 							}
// 					}
// 					else{
// 						schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 						pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 					}
// 				}
// 			}
// 			// hp_client = (hp_client + 1) % num_clients;
// 		}
// 		else{
// 			for (int j = 0; j < num_clients; ++j) {
// 				if (frecords[j] != NULL && canSchedule[j]) {
// 					schedule_kernel(*(frecords[j]), sched_streams[j], j, events[j][event_ids[j]], seen, event_ids, j);
// 					pop_from_queue(client_buffers[j], client_mutexes[j], j);
// 				}
// 			}
// 		}

// 		int finished = 0;
// 		for (int i=0; i<num_clients; i++) {
// 			if (
// 				(num_client_cur_iters[i] == num_client_max_iters[i])
// 				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 				|| (stop_ack[i] == true)
// 			)
// 				finished += 1;
// 			else if (seen[i] == num_client_kernels[i]) {
// 				// check if GPU work for this client has finished
// 				if (!locked[i]) {
// 					pthread_mutex_lock(client_mutexes[i]);
// 					locked[i] = true;
// 					DEBUG_PRINT("LOCK CLIENT %d\n", i);
// 				}
// 				bool ready = true;
// 				if (seq) {
// 					if (event_ids[0] >= 1) {
// 						if (cudaEventQuery(*(events[0][event_ids[0]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				else {
// 					if (event_ids[i] >= 1) {
// 						if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
// 							ready &= false;
// 					}
// 				}
// 				if (ready) {
// 					unsetmask_nomutex(i);
// 					// unsetmask_O(i);
// 					// if yes, reset meta-structures for this client, and let it continue
// 					seen[i] = 0;
// 					if (seq)
// 						event_ids[0] = 0;
// 					event_ids[i] = 0;
// 					streams[i] = -1;
// 					fidx[i] = 0;
// 					request_status[i][num_client_cur_iters[i]] = true;
// 					//printf("UNLOCK CLIENT %d\n", i);
// 					pthread_mutex_unlock(client_mutexes[i]);
// 					num_client_cur_iters[i] += 1;
// 					locked[i] = false;

// 					auto end = std::chrono::high_resolution_clock::now();
// 					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
// 					duration /= 1000.0;
// 					client_durations[i].push_back(duration);
// 					// if (!reef && !seq && i==hp_client && is_train[hp_client]) {
// 					// 	printf("Client %d finished iteration %d, it took %f ms\n", i, num_client_cur_iters[i], duration);
// 					// 	hp_iter_duration += duration;
// 					// 	if ((num_client_cur_iters[i] % 10) == 0 && low_sms != sm_threshold) {
// 					// 		float hp_avg_duration = hp_iter_duration/10.0;
// 					// 		printf("--------------------- Average iter duration for client 1 is %f ms, limit is %f ms, sm_threshold is %d\n", hp_avg_duration, hp_limit_float, sm_threshold);
// 					// 		hp_iter_duration = 0;

// 					// 		// TODO: add better stopping conditions
// 					// 		if (hp_avg_duration > hp_limit_float) {
// 					// 			high_sms = sm_threshold;
// 					// 			sm_threshold = (low_sms+high_sms)/2;
// 					// 		}
// 					// 		else {
// 					// 			low_sms = sm_threshold;
// 					// 			sm_threshold = (low_sms+high_sms)/2;
// 					// 		}
// 					// 	}
// 					// }
// 					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
// 				}
// 				if (
// 					(num_client_cur_iters[i] == num_client_max_iters[i])
// 					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
// 					|| (stop_ack[i] == true)
// 				) {
// 					finished += 1;
// 					if (!warmup) {
// 						auto end_total = std::chrono::high_resolution_clock::now();
// 						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
// 						duration /= 1000.0;
// 						printf("Client %d, Total loop took %f sec\n", i, duration);
// 						// if (i==num_clients-1) {
// 						// 	for (int k=0; k<num_clients-1; k++) {
// 						// 		printf("======= Client %d has done %d iterations\n", k, num_client_cur_iters[k]);
// 						// 		if (!locked[k])
// 						// 			pthread_mutex_lock(client_mutexes[k]);
// 						// 		stops[k] = true;
// 						// 		if (!locked[k])
// 						// 			pthread_mutex_unlock(client_mutexes[k]);
// 						// 	}
// 						// }
// 					}
// 				}
// 			}
// 		}

// 		if (finished==num_clients)
// 			break;

// 	}
// 	if (!warmup) {
// 		auto end_total = std::chrono::high_resolution_clock::now();
// 		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
// 		duration /= 1000.0;
// 		printf("Total loop took %f sec\n", duration);
// 		//process_eval(client_durations);
// 	}

// 	return NULL;
// }
extern "C" {

	Scheduler* sched_init() {

		// Scheduler* sched = new Scheduler();
		Scheduler* sched = new Scheduler(2);
		return sched;
	}


	void populate_kernel_info(char* kernel_info_file, vector<op_info> &ops) {

		// TODO: make this more generic, e.g. pass files/models w.r.t input
		printf("KERNEL_INFO_FILE IS %s\n", kernel_info_file);
		string line;
		std::ifstream infile(kernel_info_file);
		assert (infile.is_open());

		// ignore header
		std::getline(infile, line);

		while (std::getline(infile, line))
		{

			vector<string> v;
			stringstream sline = stringstream(line);
			while (sline.good()) {
        		string substr;
        		getline(sline, substr, ',');
        		v.push_back(substr);
    		}

			op_info info = {
				v[0],                               // Name
				stoi(v[1]),                         // Profile
				stoi(v[2]),                         // Memory footprint
				stoi(v[3]),                         // SM usage
				stof(v[4]),                         // Duration
				stoi(v[5]),                         // Grid
				stoi(v[6]),                         // Block
				stoi(v[7]),                     // Knee_TPC
				stoi(v[8]),                     // Is_Critical
				// vector<float>(), 				// Profile Data
        	};
			// for (size_t i = 8; i < v.size(); ++i) {
			// 	info.profile_data.push_back(stof(v[i])); 
			// }
			ops.push_back(info);
		}

		infile.close();

	}

	void setup_change(Scheduler* scheduler, int client_id, char* file, int num_kernels) {

		// needed for backward

		op_info_vector[client_id].clear();
		populate_kernel_info(file, op_info_vector[client_id]);
		int max_sm_used = 0;
		for (auto info: op_info_vector[client_id])
			max_sm_used = max(max_sm_used, info.sm_used);
		max_sms_clients[client_id] = max_sm_used;
		num_client_kernels[client_id] = num_kernels;

	}

	void setup(
		Scheduler* scheduler,
		int num_clients,
		int* tids,
		char** models,
		char** files,
		int* num_kernels,
		int* num_iters,
		bool* train,
		bool reef
	) {

		struct passwd *pw = getpwuid(getuid());
		char *homedir = pw->pw_dir;
		// const char* lib_path = "/orion_bu/src/cuda_capture/libinttemp.so";
		const char* lib_path = "/home/zixi/orion_bu/src/cuda_capture/libinttemp.so";

		klib = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);

		if (!klib) {
			fprintf(stderr, "Error: %s\n", dlerror());
			return;
		}

#ifdef SYS_gettid
		pid_t mytid = syscall(SYS_gettid);
#else
#error "SYS_gettid unavailable on this system"
#endif

		// 1. thread structures
		pid_t** thread_ids_all = (pid_t**)dlsym(klib, "thread_ids");
		*thread_ids_all = (pid_t*)malloc((2*num_clients+1)*sizeof(pid_t)); // 2*N threads + scheduler

		for (int i=0; i<num_clients; i++)
			(*thread_ids_all)[i] = tids[i];
		(*thread_ids_all)[num_clients] = mytid;
		for (int i=num_clients+1; i<2*num_clients+1; i++)
			(*thread_ids_all)[i] = 0;
		//printf("address is %p, %p\n", thread_ids_all, *thread_ids_all);

		int** num_total_clients = (int**)dlsym(klib, "num_total_clients");
		*num_total_clients = (int*)malloc(sizeof(int));
		**num_total_clients = num_clients;

		num_cur_clients.resize(num_clients);
		client_block.resize(num_clients);
		client_mask.resize(num_clients);
		is_executing.resize(num_clients);
		for (int i = 0; i < num_clients; ++i) {
			num_cur_clients[i] = i;
			client_block[i] = 0;
			client_mask[i] = 0;
			is_executing[i] = false;
		}
		client_finished = new bool[num_clients](); // Initialize all elements to false

		for (int i=0; i<=num_clients; i++) {
			DEBUG_PRINT("Scheduler setup the thread id at %d to be %d\n", i, (*thread_ids_all)[i]);
		}

		// 2. metadata structures
		for (int i=0; i<num_clients; i++) {
			op_info_vector.push_back({});
			client_durations.push_back({});
			populate_kernel_info(files[i], op_info_vector[i]);
			int max_sm_used = 0;
			for (auto info: op_info_vector[i])
				max_sm_used = max(max_sm_used, info.sm_used);
			max_sms_clients.push_back(max_sm_used);
			printf("----------- SIZE: %ld\n", op_info_vector[i].size());
			is_train.push_back(train[i]);
			client_progress.push_back(0);
			func_progress.push_back(-1);
		}

		// 3. indexes
		int** fidx_ptr = (int**)dlsym(klib, "func_indexes");
		*fidx_ptr = (int*)calloc(num_clients, sizeof(int));
		fidx = *fidx_ptr;

		num_client_kernels = num_kernels;
		num_client_max_iters = num_iters;

		num_client_cur_iters = (int*)calloc(num_clients, sizeof(int));
		locked = (bool*)calloc(num_clients, sizeof(bool));

		// to get measurements
		client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
		total_client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
		client_starts_set = (bool**)malloc(num_clients*sizeof(bool*));
		for (int i=0; i<num_clients; i++) {
			client_starts_set[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
		}

		// 4. communication queues + locks
		queue<func_record>*** buffers_ptr = (queue<func_record>***)dlsym(klib, "kqueues");
		*buffers_ptr = (queue<func_record>**)malloc(num_clients*sizeof(queue<func_record>*));
		queue<func_record>** buffers = *buffers_ptr;
		for (int i=0; i<num_clients; i++) {
			buffers[i] = new queue<func_record>();
			printf("buffer size is %ld\n", buffers[i]->size());
		}

		pthread_mutex_t*** client_mutexes_ptr = (pthread_mutex_t***)dlsym(klib, "mutexes");
		*client_mutexes_ptr = (pthread_mutex_t**)malloc(num_clients*sizeof(pthread_mutex_t*));
		client_mutexes = *client_mutexes_ptr;
		for (int i=0; i<num_clients; i++) {
			client_mutexes[i] = new pthread_mutex_t(); //(pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		}
		scheduler->profile_prep(buffers, num_clients, reef);

		// 5. runtime control
		bool*** request_status_ptr = (bool***)dlsym(klib, "client_request_status");
		*request_status_ptr = (bool**)malloc(num_clients*sizeof(bool*));
		request_status = *request_status_ptr;

		// check!
		bool** stops_ptr = (bool**)dlsym(klib, "client_stop");
		*stops_ptr = (bool*)calloc(num_clients, sizeof(bool));
		stops = *stops_ptr;

		bool** stop_ack_ptr = (bool**)dlsym(klib, "client_stop_ack");
		*stop_ack_ptr = (bool*)calloc(num_clients, sizeof(bool));
		stop_ack = *stop_ack_ptr;

		bool** affinity_set_ptr = (bool**)dlsym(klib, "affinity_set");
		(*affinity_set_ptr) = (bool*)calloc(num_clients+1, sizeof(bool));

		for (int i=0; i<num_clients; i++) {
			request_status[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
		}
	}


	void* schedule(Scheduler* scheduler, int num_clients, bool profile_mode, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int reef_depth, int hp_limit, int update_start) {
	// void* schedule(Scheduler* scheduler, int num_clients, bool profile_mode, int iter, bool warmup, int warmup_iters, bool reef, bool seq, int reef_depth, int hp_limit, int update_start, int mymask) {

		printf("entered sched func!\n");
		if (profile_mode)
			scheduler->busy_wait_profile(num_clients, iter, warmup, warmup_iters, true, seq, reef_depth, hp_limit, update_start);
			// scheduler->busy_wait_profile(num_clients, iter, warmup, warmup_iters, true, seq, reef_depth, hp_limit, update_start, mymask);

		printf("exited sched func!\n");
		return NULL;
	}

	void* reset(Scheduler* scheduler, int num_clients) {
		scheduler->profile_reset(num_clients);
		return NULL;
	}
}