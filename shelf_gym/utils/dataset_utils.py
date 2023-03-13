import h5py
import numpy as np
from tqdm import tqdm

def np_to_h5(h5_save_path, np_data):
    h5_file = h5py.File(h5_save_path, "w")
    for i, d in tqdm(enumerate(np_data)):
        h5_file.create_dataset("data_" + str(i), data=d)
    h5_file.close()
    return


def h5_to_np(h5f_data, size_1, size_2):
    if size_2 <0:
        np_data = np.zeros((len(h5f_data), size_1), dtype=np.float16)
    else:
        np_data = np.zeros((len(h5f_data), size_1, size_2), dtype=np.float16)
    for i in tqdm(range(len(h5f_data))):
        dset = h5f_data["data_" + str(i)]
        np_data[i] = np.asarray(dset[:])
    return np_data
