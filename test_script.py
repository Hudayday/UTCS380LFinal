import re
import gc
import os
import sys
import csv
import time
import torch
import psutil
import signal
import subprocess
import numpy as np
from pynvml import *
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Dict, Union, List, Tuple
from transformers.utils.hub import convert_file_size_to_int
from accelerate import infer_auto_device_map, init_empty_weights, disk_offload
from transformers import pipeline, BitsAndBytesConfig, GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXTokenizerFast


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_precision_from_type(float_precision):
    # need improvement
    # Extract numbers using list comprehension and join them
    return int(''.join([char for char in float_precision if char.isdigit()]))


def calc_memeory_req(model, float_precision):
    size_bytes = count_parameters(model) * get_precision_from_type(str(float_precision)) / 8
    return bytes_to_giga_bytes(size_bytes)


def calc_GPU_info(dev):
    if torch.cuda.is_available():
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(dev)
        info = nvmlDeviceGetMemoryInfo(h)
        return info.total, info.free, info.used #return Bytes
    else:
      return 0, 0, 0
    

def flush():
      gc.collect()
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()


def init_experiment(MODEL):
    if MODEL == "GPT-Neo" : return "EleutherAI/gpt-neo-2.7B", GPTNeoForCausalLM, GPT2Tokenizer
    elif MODEL == "GPT-J" : return "EleutherAI/gpt-j-6B", AutoModelForCausalLM, AutoTokenizer
    elif MODEL == "GPT-NeoX" : return "EleutherAI/gpt-neox-20b", GPTNeoXForCausalLM, GPTNeoXTokenizerFast


# Rewrite from accelerate for pure GPU usage
def get_max_memory(number_of_GPU_in_use = -1, max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None):
    """
    Get the maximum memory available if nothing is passed, converts string to int otherwise.
    number_of_GPU_in_use will ignore extra GPUs, -1 means no limitation
    """
    import psutil

    if max_memory is None:
        if not (torch.cuda.is_available()):
            max_memory = {}

        else:
            if number_of_GPU_in_use == -1 or torch.cuda.device_count() < number_of_GPU_in_use:
                num_GPU = torch.cuda.device_count()
            else:
                num_GPU = number_of_GPU_in_use
            # Make sure CUDA is initialized on each GPU to have the right memory info.
            for i in range(num_GPU):
                _ = torch.tensor([0], device=i)
            max_memory = {i: 0.9*torch.cuda.mem_get_info(i)[0] for i in range(num_GPU)}

        max_memory["cpu"] = psutil.virtual_memory().available
        return max_memory

    for key in max_memory:
        if isinstance(max_memory[key], str):
            max_memory[key] = convert_file_size_to_int(max_memory[key])

    # Need to sort the device by type to make sure that we allocate the gpu first.
    # As gpu/xpu are represented by int, we need to sort them first.
    gpu_devices = [k for k in max_memory.keys() if isinstance(k, int)]
    gpu_devices.sort()
    # check if gpu devices are available and if not, throw a warning
    num_devices = torch.cuda.device_count()
    for device in gpu_devices:
        if device >= num_devices or device < 0:
            print(f"Device {device} is not available, available devices are {list(range(num_devices))}")
    # Add the other devices in the preset order if they are available
    all_devices = gpu_devices + [k for k in ["mps", "cpu", "disk"] if k in max_memory.keys()]
    # Raise an error if a device is not recognized
    for k in max_memory.keys():
        if k not in all_devices:
            raise ValueError(
                f"Device {k} is not recognized, available devices are integers(for GPU/XPU), 'mps', 'cpu' and 'disk'"
            )
    max_memory = {k: max_memory[k] for k in all_devices}

    return max_memory


def model_loader(model_disk, model_name, selection : int = 0, model_type = GPTNeoXForCausalLM, float_precision = torch.bfloat16):
    '''
    Huday v 0.0.1
    Para selection represents the way of loading the model,
    which some of them are limited to the number of graphic card
    0: out-of-the-box "auto" loadiing ("auto" or "balanced": Accelerate will split the weights so that each GPU is used equally)
    1: out-of-the-box "sequential" loading ("sequential": Accelerate will fill the GPUs in order (so the last ones might not be used at all))
    # 2: custom balanced loading
    # 3: custom sequential loading
    2: Quantization, force load on 1 GPU (manual now)
    3: vanilla transformer force load on one GPU
    '''
    if selection == 0:
        model = model_type.from_pretrained(model_name, torch_dtype=float_precision, device_map='auto')
    elif selection == 1:
        model = model_type.from_pretrained(model_name, torch_dtype=float_precision, device_map='sequential')
    # elif selection == 2:
    #     df = custom_balanced_loading(model_disk, float_precision)
    #     model = model_type.from_pretrained(model_name, torch_dtype=float_precision, device_map=df)
    # elif selection == 3:
    #     df = custom_sequential_loading(model_disk, float_precision)
    #     model = model_type.from_pretrained(model_name, torch_dtype=float_precision, device_map=df)
    elif selection == 2: #INT8
        model = model_type.from_pretrained(model_name, torch_dtype=float_precision, device_map=infer_auto_device_map(model_disk,get_max_memory(1)),\
                                           quantization_config=BitsAndBytesConfig(load_in_8bit=True,
                                                                                  bnb_8bit_compute_dtype=torch.bfloat16,
                                                                                  llm_int8_enable_fp32_cpu_offload=True # we need to add this foor cpu offloading
                                                                                  ))
    elif selection == 3:
        model = model_type.from_pretrained(model_name, torch_dtype=float_precision, device_map=infer_auto_device_map(model_disk,get_max_memory(1)))
    elif selection == 4:
        model = model_type.from_pretrained(model_name, torch_dtype=float_precision)
    elif selection == 5:
        model = model_type.from_pretrained(model_name)

    print(model.hf_device_map)
    print("Total Memeory Requirement of {} is {:.2f} GB".format(model_name, calc_memeory_req(model, float_precision)))
    print("Maximum GPU Memory Usage of {} is {:.2f} GB".format(model_name, bytes_to_giga_bytes(torch.cuda.max_memory_allocated())))
    total, free, used = calc_GPU_info(0)
    print("GPU-1 Info: Total - {:.2f} GB, Free - {:.2f} GB, Used - {:.2f} GB".format(bytes_to_giga_bytes(total), bytes_to_giga_bytes(free), bytes_to_giga_bytes(used)))
    if torch.cuda.device_count() > 1:
        total, free, used = calc_GPU_info(1)
        print("GPU-2 Info: Total - {:.2f} GB, Free - {:.2f} GB, Used - {:.2f} GB".format(bytes_to_giga_bytes(total), bytes_to_giga_bytes(free), bytes_to_giga_bytes(used)))

    return model

def evaluate(pipe, prompt):
    in_bytes = sys.getsizeof(prompt) # size of input (bytes)
    start = time.time() # start measutring time (sec)
    conv = pipe(prompt) # call pipeline
    end = time.time() # stop measurinng time (sec)
    latency = end - start # (sec)
    throughput = in_bytes/latency # (bytes/sec)
    return conv, latency, throughput


def multi_evalute(output_filename, MODEL, SELECTION, model_disk, model_name, model_type, tranformer_type):
    fields = ['Timestamp', 'Load Type', 'Prompt', 'output','Latency ave (sec)', 'Latency std (sec)', 'Throughput ave (bytes/sec)', 'Throughput std (bytes/sec)']
    out_list = []
    
    i = SELECTION

    command_record = [
    "nvidia-smi",
    "--query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
    "--format=csv",
    "-l", "1",
    "-f", f"./GPU-record-{MODEL}-method_{i}.csv"
    ]

    process = subprocess.Popen(command_record) 
    pid = process.pid
    
    print(f"======================START=={MODEL}_{i}=========================")
    model = model_loader(model_disk, model_name, selection = i, model_type = model_type, float_precision = torch.bfloat16)
    tokenizer = tranformer_type.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    filename = "input_prompt.txt"
    with open(filename) as file:
        for prompt in file:
            conv_list = []
            lat_list = []
            thrput_list = []
            for j in range(3):   #Take average and std
                conv, lat, thrput = evaluate(pipe, prompt)
                conv_list.append(conv)
                lat_list.append(lat)
                thrput_list.append(thrput)
                
            timestamp = time.time()

            # Convert the timestamp to a datetime object
            dt_object = datetime.fromtimestamp(timestamp)

            # Format the datetime object
            formatted_time = dt_object.strftime('%Y/%m/%d %H:%M:%S.%f')
            
            out_list.append([formatted_time,i,prompt,conv_list,np.mean(lat_list),np.std(lat_list),np.mean(thrput_list),np.std(thrput_list)])
            
        del model
        flush()
        os.kill(pid, signal.SIGINT)
        print(f"======================END=={MODEL}_{i}=========================")

    with open(output_filename, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields)   
        csvwriter.writerows(out_list)    


def multi_evalute_multi_model(output_filename, MODEL, SELECTION, model_disk, model_name, model_type, tranformer_type):
    fields = ['Timestamp', 'Load Type', 'Prompt', 'output','Latency ave (sec)', 'Latency std (sec)', 'Throughput ave (bytes/sec)', 'Throughput std (bytes/sec)']
    out_list = []
    
    i = SELECTION

    command_record = [
    "nvidia-smi",
    "--query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
    "--format=csv",
    "-l", "1",
    "-f", f"./GPU-record-{MODEL}-method_{i}.csv"
    ]

    process = subprocess.Popen(command_record) 
    pid = process.pid
    
    print(f"======================START=={MODEL}_{i}=========================")
    model = model_loader(model_disk, model_name, selection = i, model_type = model_type, float_precision = torch.bfloat16)
    model2 = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", torch_dtype=torch.bfloat16)

    tokenizer = tranformer_type.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    tokenizer2 = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    pipe2 = pipeline("text-generation", model=model2, tokenizer=tokenizer2)
    
    filename = "input_prompt.txt"
    with open(filename) as file:
        for prompt in file:
            conv_list = []
            lat_list = []
            thrput_list = []

            conv_list2 = []
            lat_list2 = []
            thrput_list2 = []
            for j in range(3):   #Take average and std
                conv, lat, thrput = evaluate(pipe, prompt)
                conv_list.append(conv)
                lat_list.append(lat)
                thrput_list.append(thrput)

                conv, lat, thrput = evaluate(pipe2, prompt)
                conv_list2.append(conv)
                lat_list2.append(lat)
                thrput_list2.append(thrput)
                
            timestamp = time.time()

            # Convert the timestamp to a datetime object
            dt_object = datetime.fromtimestamp(timestamp)

            # Format the datetime object
            formatted_time = dt_object.strftime('%Y/%m/%d %H:%M:%S.%f')
            
            out_list.append([formatted_time,i,prompt,conv_list,np.mean(lat_list),np.std(lat_list),np.mean(thrput_list),np.std(thrput_list),conv_list2,np.mean(lat_list2),np.std(lat_list2),np.mean(thrput_list2),np.std(thrput_list2)])
            
        del model
        flush()
        os.kill(pid, signal.SIGINT)
        print(f"======================END=={MODEL}_{i}=========================")

    with open(output_filename, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields)   
        csvwriter.writerows(out_list)    


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <MODEL> <SELECTION>')
        print(f'<MODEL> : <GPT-Neo> (2.7b), <GPT-J> (6b), <GPT-NeoX> (20b)')
        print(f'<SELECTION>: <0> (auto/balanced-loading), <1> (sequential-loading), <2> (quantized-loading), <3> (single_GPU-loading)')
        return -1

    print("======================START==OF==EXPERIMENT=========================")

    MODEL = sys.argv[1] # model for the experiment
    SELECTION = int(sys.argv[2]) # selction for mapping
    print(f'Model selected for experiment: {MODEL}')
    print(f'Device Mapinng selected for experiment: {SELECTION}')

    # Initialization 
    print(f'Is Cuda Availabe? : {torch.cuda.is_available()}')
    print(f'Cuda Device Count : {torch.cuda.device_count()}')
    model_name, model_type, tranformer_type = init_experiment(MODEL) #initialize model and transformer wrappers
    print(f'Full Name of Model selected for experiment: {model_name}')

    # Intialise model on memory
    model2 = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", torch_dtype=torch.bfloat16)
    model_disk = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={'': 'cpu'})
    model_disk.to('cpu')
    print(f'\nModel Overview:\n{model_disk}\n')
    print(f'\nModel Device Initial Mapping:\n{infer_auto_device_map(model_disk)}\n')

    # Evaluation
    output_filename = f"result_{MODEL}_{SELECTION}.csv"
    print("======================START==OF==EVALUATION=========================")
    multi_evalute(output_filename, MODEL, SELECTION, model_disk, model_name, model_type, tranformer_type)
    print("======================END==OF==EVALUATION=========================")

    print("======================END==OF==EXPERIMENT=========================")

    del model_disk
    flush()
    
    return 0 


if __name__ == "__main__":
  main()