import glob
import numpy as np

DATASET_PATH = "./pointclouds-500/"










if __name__ == "__main__":
    # Load XYZ file (assuming it's space-separated)
    xyz_files = glob.glob(DATASET_PATH + "*.xyz")

    if not xyz_files:
        print("No .xyz_files files found in the directory.")
        # return None
    
    xyz_data = np.loadtxt(xyz_files[1])

    # Access X, Y, Z columns separately
    x, y, z = xyz_data[:, 0], xyz_data[:, 1], xyz_data[:, 2]

    print(x[:5], y[:5], z[:5])  # Print first 5 values