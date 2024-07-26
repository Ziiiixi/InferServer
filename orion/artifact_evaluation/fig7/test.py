import threading
from threading import Barrier
import time

# Number of worker threads
num_workers = 3

# Initialize barriers
barriers = [Barrier(num_workers + 1), Barrier(num_workers + 1)]

def worker(worker_id):
    print(f"Worker {worker_id} starting")
    time.sleep(1)  # Simulate work
    print(f"Worker {worker_id} reached first barrier")
    barriers[0].wait()  # Wait at the first barrier
    print(f"Worker {worker_id} passed first barrier")
    time.sleep(1)  # Simulate more work
    print(f"Worker {worker_id} reached second barrier")
    barriers[1].wait()  # Wait at the second barrier
    print(f"Worker {worker_id} passed second barrier")

def scheduler():
    print("Scheduler waiting for workers to reach first barrier")
    barriers[0].wait()  # Wait for all worker threads and scheduler thread to reach the first barrier
    print("Scheduler scheduling tasks")
    time.sleep(2)  # Simulate scheduling tasks
    print("Scheduler finished scheduling")
    barriers[1].wait()  # Optionally, wait for all threads to reach the second barrier

# Create and start worker threads
worker_threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_workers)]
for t in worker_threads:
    t.start()

# Create and start the scheduler thread
scheduler_thread = threading.Thread(target=scheduler)
scheduler_thread.start()

# Wait for all worker threads to finish
for t in worker_threads:
    t.join()

# Wait for the scheduler thread to finish
scheduler_thread.join()

print("All threads have finished")
