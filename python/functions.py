import os, psutil
import numpy as np


def MemoryMb():
    process = psutil.Process(os.getpid()) # Keep track of memory usage
    memoryMb = process.memory_info().rss / 10 ** 6
    print(f'\nMemory used = {memoryMb} Mb\n', flush=True) # Display MB of memory usage
    del memoryMb, process


def ConvertLabelToInt(mapping, str_label):
    for intkey, string in mapping.items():
        if str_label == string:
            return intkey

    return "The label has not been found :("


#     #https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/11146645#11146645
def CartesianProduct(*arrays): 
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)