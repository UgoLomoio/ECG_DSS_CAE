"""
*************************************************************************
*
* Developed by Salvatore Petrolo @t0re199
* _______________________________________
*
*  Copyright C 2022 Salvatore Petrolo
*  All Rights Reserved.
*
* NOTICE:  All information contained herein is, and remains
* the property of Salvatore Petrolo.
* The intellectual and technical concepts contained
* herein are proprietary to Salvatore Petrolo.
* Dissemination of this information or reproduction of this material
* is strictly forbidden.
*************************************************************************
"""


import dill
import torch
from EcgStuffs.src.dpnet import conf


def load(path=conf.DPNET_PATH):
    with open(path, "rb") as fd:
        return dill.load(fd)


def load_cpu(path=conf.DPNET_CPU_PATH):
    
    cpu = torch.device("cpu")
    with open(path, "rb") as fd:
        return torch.load(fd, map_location = cpu, pickle_module=dill)

def create_cpu_model(path = conf.DPNET_PATH, cpu_path = conf.DPNET_CPU_PATH):

    #load gpu
    gpu_model = load(path)
    #save cpu
    cpu_model = gpu_model.to(torch.device("cpu"))
    torch.save(cpu_model, cpu_path, pickle_module=dill)
    #load cpu
    #cpu_model_loaded = load_cpu(cpu_path)
    #print(cpu_model_loaded)