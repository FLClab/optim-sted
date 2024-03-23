
import h5py
import shutil
import os
import time
import numpy

# BlockingIOError
# OSError

path = "../../data/debug/tmp.hdf5"

with h5py.File(path, mode="a") as file:
    print(file.keys())

    with h5py.File(path, mode="r+") as file2:
        print(file2.keys())

        if "tmp" in file2:
            del file2["tmp"]
        file2.create_dataset("tmp", data=numpy.random.rand(128, 128))
        print(file.keys())
