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


import math


NORMAL_LABEL = 0x0
AF_LABEL = 0x1


def sliding_window(signal, size, stride=0x1):
    windows = []
    sig_len = signal.shape[0x1]
    win_num = math.ceil((sig_len - size) / stride)
    add_last = sig_len % win_num != 0x0
    for i in range(win_num):
        offset = i * stride
        windows.append((signal[:, offset : offset + size]))
    if add_last:
        windows.append(signal[:, sig_len - size :])
    return windows


def generate_windows_labels(win_num, time_wise_labels, width_in_sec, stride_in_sec=0x1):
    labels = [NORMAL_LABEL] * win_num
    for i in range(win_num):
        w_begin = round(i * stride_in_sec)
        w_end = round(i * stride_in_sec + width_in_sec)

        if time_wise_labels[w_begin : w_end].sum() > 0x0:
            labels[i] = AF_LABEL

    return labels
