import os
from platform import node
import sched
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import models, datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
from datetime import timedelta
import random
import numpy as np
import time
import os
import argparse
import threading
import json
from ctypes import *
from pathlib import Path

def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.data = torch.rand([self.batchsize, 3, 224, 224], pin_memory=True)
        # self.data = torch.ones([self.batchsize, 3, 224, 224], pin_memory=True)
        self.target = torch.ones([self.batchsize], pin_memory=True, dtype=torch.long)

    def __iter__(self):
        return self

    def __next__(self):
        return self.data, self.target

class RealDataLoader():
    def __init__(self, batchsize):
        train_transform =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]
        )
        train_dataset = \
                datasets.ImageFolder("/mnt/data/home/fot/imagenet/imagenet-raw-euwest4",transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batchsize, num_workers=8)

    def __iter__(self):
        print("Inside iter")
        return iter(self.train_loader)


def block(backend_lib, it):
    # block client until request served
    backend_lib.block(it)

def check_stop(backend_lib):
    return backend_lib.stop()


from pathlib import Path
import random
from typing import List, Optional

def pick_trace_sleep_times(trace_dir: str,
                           num_iters: int,
                           id_: Optional[int] = None,
                           loop_if_short: bool = True,
                           shuffle: bool = False) -> List[float]:
    """
    Return *num_iters* inter-arrival gaps (seconds) taken from the id_-th trace
    file in *trace_dir* (alphabetical order).

    Parameters
    ----------
    trace_dir : str
        Directory containing trace text files (one float per line).
    num_iters : int
        Number of sleep times to return.
    id_ : int | None
        0-based index into sorted(trace_files).
        • id_ = 0 → first file, id_ = 1 → second file, …
        • If None, choose a random file (back-compatibility).
    loop_if_short : bool, default True
        Repeat the trace if it is shorter than *num_iters*.
    shuffle : bool, default False
        Shuffle the resulting list after looping.
    """
    trace_dir = Path(trace_dir)
    trace_files = sorted(p for p in trace_dir.iterdir() if p.is_file())
    if not trace_files:
        raise FileNotFoundError(f"No trace files found in {trace_dir}")

    # ----- file selection ----------------------------------------------------
    if id_ is None:
        trace_file = random.choice(trace_files)
    else:
        if not (0 <= id_ < len(trace_files)):
            raise IndexError(f"id_={id_} out of range (0‒{len(trace_files)-1})")
        trace_file = trace_files[id_]

    # ----- read and convert to gaps -----------------------------------------
    with trace_file.open() as f:
        vals = [float(line.strip()) for line in f if line.strip()]

    if len(vals) < 2:
        raise ValueError(f"Trace file {trace_file} is too short")

    # If monotone non-decreasing ⇒ absolute timestamps → convert to gaps
    if all(b >= a for a, b in zip(vals, vals[1:])):
        gaps = [vals[0]] + [b - a for a, b in zip(vals, vals[1:])]
    else:
        gaps = vals

    # ----- extend / shuffle --------------------------------------------------
    if len(gaps) < num_iters:
        if not loop_if_short:
            raise ValueError(f"Trace length {len(gaps)} < num_iters={num_iters}")
        reps = -(-num_iters // len(gaps))          # ceil division
        gaps = (gaps * reps)[:num_iters]

    if shuffle:
        random.shuffle(gaps)

    return gaps[:num_iters]


def imagenet_loop(
    model_name,
    batchsize,
    train,
    num_iters,
    rps,
    uniform,
    dummy_data,
    local_rank,
    barriers,
    client_barrier,
    tid,
    input_file=''
):

    seed_everything(42)
    print(model_name, batchsize, local_rank, barriers, tid)
    # backend_lib = cdll.LoadLibrary(os.path.expanduser('~') + "/orion_bu/src/cuda_capture/libinttemp.so")
    backend_lib = cdll.LoadLibrary("/home/zixi/orion_bu/src/cuda_capture/libinttemp.so")
    if rps > 0 and input_file=='':
        # if uniform:
            # sleep_times = [1/rps]*num_iters
        # else:
        # sleep_times = pick_trace_sleep_times(
        #     # trace_dir="/home/zixi/orion_bu/benchmarking/benchmark_suite/TW_may25_1hr_traces/",
        #     trace_dir="/home/zixi/orion_bu/benchmarking/benchmark_suite/Real_world_trace/trace_txt",
        #     num_iters=num_iters,
        #     id_=tid
        # )
        sleep_times = np.random.exponential(scale=1/20, size=num_iters)

    elif input_file != '':
        with open(input_file) as f:
                sleep_times = json.load(f)
    else:
        sleep_times = [0]*num_iters


    print(f"SIZE is {len(sleep_times)}")
    print(sleep_times)
    barriers[0].wait()

    print("-------------- thread id:  ", threading.get_native_id())

    # if (train and tid==1):
        # time.sleep(5)

    #data = torch.rand([batchsize, 3, 224, 224]).contiguous()
    #target = torch.ones([batchsize]).to(torch.long)
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(0)

    if train:
        model.train()
        optimizer =  torch.optim.SGD(model.parameters(), lr=0.1)
        criterion =  torch.nn.CrossEntropyLoss().to(local_rank)
    else:
        model.eval()

    if dummy_data:
        train_loader = DummyDataLoader(batchsize)
    else:
        train_loader = RealDataLoader(batchsize)

    train_iter = enumerate(train_loader)
    batch_idx, batch = next(train_iter)
    gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
    print("Enter loop!")

    #  open loop
    next_startup = time.time()
    open_loop = True
    overall_start = time.time()
    if True:
        timings=[]
        for i in range(1):
            print("Start epoch: ", i)

            while batch_idx < num_iters:
                # time.sleep(1)
                print(f"Client {tid}, submit!, batch_idx is {batch_idx}, num iter is {num_iters}")
                start_iter = time.time()
                #torch.cuda.profiler.cudart().cudaProfilerStart()
                if train:
                    #client_barrier.wait()
                    print(f"Client {tid}, submit!, batch_idx is {batch_idx}")
                    # if tid==0 and batch_idx==20:
                    #     torch.cuda.profiler.cudart().cudaProfilerStart()
                    gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
                    optimizer.zero_grad()
                    output = model(gpu_data)
                    loss = criterion(output, gpu_target)
                    loss.backward()
                    optimizer.step()
                    block(backend_lib, batch_idx)
                    iter_time = time.time()-start_iter
                    timings.append(iter_time)
                    #print(f"Client {tid} finished! Wait! It took {timings[batch_idx]}")
                    batch_idx, batch = next(train_iter)
                    if (batch_idx == 1): # for backward
                        barriers[0].wait()
                    if batch_idx == 10: # for warmup
                        barriers[0].wait()
                        start = time.time()
                    if check_stop(backend_lib):
                        print("---- STOP!")
                        break
                    # if batch_idx==20:
                    #     torch.cuda.profiler.cudart().cudaProfilerStart()
                else:
                    with torch.no_grad():
                        cur_time = time.time()
                        #### OPEN LOOP ####
                        if open_loop:
                            # print("cur time:", cur_time, "next startup time", next_startup)
                            if (cur_time >= next_startup):
                                print(f"Client {tid}, submit!, batch_idx is {batch_idx}")
                                if batch_idx==100:
                                    torch.cuda.profiler.cudart().cudaProfilerStart()
                                # gpu_data = batch[0].to(local_rank)
                                output = model(gpu_data)
                                block(backend_lib, batch_idx)
                                # print(f"time {time.time()}, next_startup time is {next_startup}, diff = {time.time() - next_startup}")
                                # print(f"time {time.time()}, cur_time time is {cur_time}, diff = {time.time() - cur_time}")
                                # print(f"time {time.time()}, next_startup time is {next_startup}")
                                req_time = time.time()-next_startup
                                # req_time = time.time()-cur_time
                                timings.append(req_time)
                                print(f"Client {tid} finished! Wait! It took {req_time}")
                                # decomment it #####
                                if batch_idx>=10:
                                    next_startup += sleep_times[batch_idx]
                                else:
                                    next_startup = time.time()
                                ###########
                                batch_idx,batch = next(train_iter)
                                if (batch_idx == 1 or (batch_idx == 10)):
                                    barriers[0].wait()
                                    # hp starts after
                                    if (batch_idx==10):
                                        next_startup = time.time()
                                        start = time.time()
                                dur = next_startup-time.time()
                                # print(f"Client sleep {dur}")
                                if (dur>0):
                                    time.sleep(dur)
                                if check_stop(backend_lib):
                                    print(f"Client {tid} ---- STOP!")
                                    break
                        else:
                            #### CLOSED LOOP ####
                            print(f"Client {tid}, submit!, batch_idx is {batch_idx}")
                            gpu_data = batch[0].to(local_rank)
                            output = model(gpu_data)
                            block(backend_lib, batch_idx)
                            print(f"Client {tid} finished! Wait!")
                            batch_idx,batch = next(train_iter)
                            if ((batch_idx == 1) or (batch_idx == 10)):
                                barriers[0].wait()

        print(f"Client {tid} at barrier! ")
        overall_end = time.time()
        overall_duration = overall_end - overall_start
        print(f"Client {tid} Total execution time: {overall_duration} seconds")
        barriers[0].wait()
        total_time = time.time() - start
        timings = timings[10:]
        # timings = sorted(timings)
        print(f"Client {tid} finshed {len(timings)} iterations")
        # print(f"Client {tid} timings {timings}")
        if not train and len(timings)>0:
            
            p50 = np.percentile(timings, 50)
            p95 = np.percentile(timings, 95)
            p99 = np.percentile(timings, 99)
            average = np.mean(timings)
            print(f"Client {tid} finished! p50: {p50} sec, p95: {p95} sec, p99: {p99} sec, average: {average}")
            data = {
                'p50_latency': p50*1000,
                'p95_latency': p95*1000,
                'p99_latency': p99*1000,
                'throughput': (batch_idx-10)/total_time
            }
        else:
            data = {
                'throughput': (batch_idx-10)/total_time
            }
        print()
        with open(f'client_{tid}.json', 'w') as f:
            json.dump(data, f)

        print("Finished! Ready to join!")
