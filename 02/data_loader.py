import glob
import numpy as np


DATASET_PATH = "./pointclouds-500"
LABELS = [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100

def load_one_data(dataset_folder: str = DATASET_PATH, id: int = 0):
    label = LABELS[id]

    file_name = "{:03d}.xyz".format(id)
    xyz_data = np.loadtxt(f"{dataset_folder}/{file_name}")

    return xyz_data, label

def load_all_datasets(dataset_folder: str = DATASET_PATH):
    # Create an array of 500 elements
    labels = LABELS
    all_data = []

    for file_num in range(500):
        file_name = "{:03d}.xyz".format(file_num)
        xyz_data = np.loadtxt(f"{dataset_folder}/{file_name}")
        all_data.append(xyz_data)
    
    dataset = np.array(all_data, dtype=object)
    print("dataset shape: ", dataset.shape)
    # print("dataset[0]:\n", type(dataset[0]))

    return dataset, labels


if __name__ == "__main__":
    
    load_all_datasets(DATASET_PATH)
    