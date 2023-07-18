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
from EcgStuffs.src.dpnet import conf

def load(path=conf.DPNET_PATH):
    with open(path, "rb") as fd:
        return dill.load(fd)
