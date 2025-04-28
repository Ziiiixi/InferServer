import argparse
import json
import threading
import time
from ctypes import *
import os
import sys
from torchvision import models
import torch
import pynvml  # Import pynvml for energy measurement

home_directory = os.path.expanduser('~')
sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch")
sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils")
from benchmark_suite.transformer_trainer import transformer_loop
sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/BERT")
from benchmark_suite.bert_trainer_mock import bert_loop

from benchmark_suite.train_imagenet import imagenet_loop
from benchmark_suite.toy_models.bnorm_trainer import bnorm_loop
from benchmark_suite.toy_models.conv_bn_trainer import conv_bn_loop

from src.scheduler_frontend import PyScheduler

function_dict = {
    "alexnet": imagenet_loop,# mmy
    "resnet50": imagenet_loop,
    "resnet152": imagenet_loop,
    "resnet101": imagenet_loop,
    "mobilenet_v2": imagenet_loop,
    "densenet201": imagenet_loop, #mmy
    "resnext50_32x4d": imagenet_loop, #mmy
    "shufflenet_v2_x1_0": imagenet_loop, #mmy
    "vgg19": imagenet_loop, #mmy
    "squeezenet1_0": imagenet_loop, #mmy
    "bnorm": bnorm_loop,
    "conv_bnorm": conv_bn_loop,
    "bert": bert_loop,
    "transformer": transformer_loop,
    "vit_b_16": imagenet_loop,
}

def seed_everything(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_gpu_power_usage(handle):
    """Retrieve current GPU power usage in watts."""
    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # Power usage in milliwatts
    return power_mw / 1000.0  # Convert to watts

def energy_monitor(stop_event, power_list, sample_interval=1.0):
    """Monitor GPU power usage until stop_event is set."""
    while not stop_event.is_set():
        power = get_gpu_power_usage(gpu_handle)
        power_list.append(power)
        time.sleep(sample_interval)

def launch_jobs(config_dict_list, input_args, run_eval):
    print("in launch_jobs")
    seed_everything(42)
    
    print(config_dict_list)
    num_clients = len(config_dict_list)
    print(num_clients)

    s = torch.cuda.Stream()

    # Initialize barriers
    num_barriers = num_clients + 1
    barriers = [threading.Barrier(num_barriers) for _ in range(num_clients)]
    client_barrier = threading.Barrier(num_clients)

    # Load scheduler library
    if run_eval:
        # sched_lib = cdll.LoadLibrary(home_directory + "/orion_bu/src/scheduler/scheduler_eval.so")
        sched_lib = cdll.LoadLibrary("/home/zixi/orion_bu/src/scheduler/scheduler_eval.so")
    else:
        # sched_lib = cdll.LoadLibrary(home_directory + "/orion_bu/src/scheduler/scheduler.so")
        sched_lib = cdll.LoadLibrary("/home/zixi/orion_bu/src/scheduler/scheduler_eval.so")
    py_scheduler = PyScheduler(sched_lib, num_clients)

    print(torch.__version__)

    model_names = [config_dict['arch'] for config_dict in config_dict_list]
    model_files = [config_dict['kernel_file'] for config_dict in config_dict_list]

    additional_model_files = [config_dict['additional_kernel_file'] if 'additional_kernel_file' in config_dict else None for config_dict in config_dict_list]
    num_kernels = [config_dict['num_kernels'] for config_dict in config_dict_list]
    num_iters = [config_dict['num_iters'] for config_dict in config_dict_list]
    train_list = [config_dict['args']['train'] for config_dict in config_dict_list]
    additional_num_kernels = [config_dict['additional_num_kernels'] if 'additional_num_kernels' in config_dict else None for config_dict in config_dict_list]

    tids = []
    threads = []
    for i, config_dict in enumerate(config_dict_list):
        func = function_dict[config_dict['arch']]
        model_args = config_dict['args']
        model_args.update({
            "num_iters": num_iters[i],
            "local_rank": 0,
            "barriers": barriers,
            "client_barrier": client_barrier,
            "tid": i
        })

        thread = threading.Thread(target=func, kwargs=model_args)
        thread.start()
        tids.append(thread.native_id)
        threads.append(thread)

    print(tids)

    sched_thread = threading.Thread(
        target=py_scheduler.run_scheduler,
        args=(
            barriers,
            tids,
            model_names,
            model_files,
            additional_model_files,
            num_kernels,
            additional_num_kernels,
            num_iters,
            True,
            run_eval,
            input_args.algo == 'reef',
            input_args.algo == 'sequential',
            input_args.reef_depth if input_args.algo == 'reef' else input_args.orion_max_be_duration,
            input_args.orion_hp_limit,
            input_args.orion_start_update,
            train_list
            # input_args.mymask
        )
    )
    print("before start !!!!!!!!!!!!!!")
    # print(input_args.mymask)
    sched_thread.start()

    # Start energy monitoring
    stop_event = threading.Event()
    power_list = []
    monitor_thread = threading.Thread(target=energy_monitor, args=(stop_event, power_list))
    monitor_thread.start()

    for thread in threads:
        thread.join()

    print("Train threads joined!")

    sched_thread.join()
    print("Scheduler thread joined!")

    # Stop energy monitoring
    stop_event.set()
    monitor_thread.join()

    print("--------- All threads joined!")

    # Calculate total energy consumed
    total_energy_joules = sum(power_list) * 1.0  # power_list sampled every 1 second
    total_energy_kj = total_energy_joules / 1000.0  # Convert to kJ
    print(f"Total energy consumed: {total_energy_kj:.2f} kJ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True,
                        help='Choose one of orion | reef | sequential')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the experiment configuration file')
    parser.add_argument('--reef_depth', type=int, default=1,
                        help='If reef is used, this stands for the queue depth')
    parser.add_argument('--orion_max_be_duration', type=int, default=1,
                        help='If orion is used, the maximum aggregate duration of on-the-fly best-effort kernels')
    parser.add_argument('--orion_start_update', type=int, default=1,
                        help='If orion is used, and the high priority job is training, this is the kernel id after which the update phase starts')
    parser.add_argument('--orion_hp_limit', type=int, default=1,
                        help='If orion is used, and the high priority job is training, this shows the maximum tolerated training iteration time')
    # parser.add_argument('--mymask', type=int, default=1,
    #                     help='mymask added')

    args = parser.parse_args()

    # Initialize NVML
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0

    torch.cuda.set_device(0)
    # affinity_mask = {0,1,2,3}
    # os.sched_setaffinity(0, affinity_mask)
    profile = True
    with open(args.config_file) as f:
        config_dict = json.load(f)
    launch_jobs(config_dict, args, True)

    # Shutdown NVML
    pynvml.nvmlShutdown()
