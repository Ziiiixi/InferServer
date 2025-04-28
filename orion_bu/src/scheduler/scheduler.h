#include <stdio.h>
#include <dlfcn.h>
#include <queue>
#include <vector>
#include <pthread.h>
#include <syscall.h>
#include <pwd.h>
#include <iostream>
#include <string.h>
#include <tuple>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <boost/thread/barrier.hpp>

#include "utils_sched.h"

//void* sched_func(void* args);
class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;

private:
    // Workers
    std::vector<std::thread> workers;
    // Task queue
    std::queue<std::function<void()>> tasks;

    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
        worker.join();
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Don't allow enqueueing after stopping the pool
        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

class Scheduler {

	public:

        boost::barrier sync_barrier;
            Scheduler(int num_clients)
            : sync_barrier(num_clients)
        {}
	    struct ScheduleItem {
			int id;
			int kernel_idx;
			int batch;
			int tpc;
			bool is_critical;
    	};
        struct ScheduleEntry {
            int client_id;
            int kernel_index;
            int tpc_usage;
            int concurrent_kernel;
        };
        
        struct KernelData {
            int Kernel_ID;
            std::string Kernel_Name;
            std::string Model;
            int Cluster;
            long Duration;
        
            // New fields for TPC performance
            int Block_Size;
            int Grid_Size;
            double Exclusive_Run;
            std::vector<double> TPC_Performance;  // index 0 for 1 TPC, index 23 for 24 TPCs
            int Knee_TPC;
        };
        
		void profile_prep(queue<func_record>** qbuffers, int num_clients, bool reef);
		void generate_partitions(int n, int start, std::vector<std::vector<int>>& partition, std::vector<std::vector<std::vector<int>>>& partitions) ;
		void profile_reset(int num_clients);
		std::pair<int, std::vector<std::vector<std::pair<int, int>>>>  computeTailLatencyForPartition(const std::vector<std::vector<int>>& partition, const std::vector<std::vector<float>>& latencyTable, int totalTPCs);
		std::pair<float, std::vector<std::pair<int, int>>> computeTailLatencyDP(const std::vector<int>& clients, const std::vector<std::vector<float>>& latencyTable, int tpc);
		void* busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq,  int depth, int hp_limit, int update_start);
		void execute_kernel_profile(int client_id, struct func_record frecord, op_info op_info_cur, int tpc_usage, int cur_iter);
        void execute_kernel(int client_id, struct func_record frecord);
		std::vector<std::vector<Scheduler::ScheduleItem>> lookahead_schedule(vector<int> &ready_client);
        std::vector<std::vector<Scheduler::ScheduleItem>> shifting_schedule(vector<int> &ready_client);
		// void* busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, bool seq,  int depth, int hp_limit, int update_start, int mymask);
		// void schedule_spacial(vector<func_record*> frecords, int num_clients, int depth);
		void schedule_reef(vector<func_record*> frecords, int num_clients, int depth, int hp_client);
		// void schedule_KRISP(vector<func_record*> frecords, int num_clients, int depth);
		// void schedule_KRISP_I(vector<func_record*> frecords, int num_clients, int depth);
		// void schedule_KRISP_O(vector<func_record*> frecords, int num_clients, int depth);
		// int schedule_sequential(vector<func_record*> frecords, int num_clients, int start);

};

//void* sched_func(void* sched);
//Scheduler* sched_init();
