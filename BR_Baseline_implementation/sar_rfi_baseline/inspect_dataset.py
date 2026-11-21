
from scipy.io import loadmat
import glob
import os
import numpy as np

data_path = r"C:\Users\hp\Pictures\BR Baseline implementation\Data"

files = glob.glob(os.path.join(data_path, "**", "*.mat"), recursive=True)

print("Total MATLAB files:", len(files))

for file in sorted(files):
    print("\n" + "=" * 60)
    print("FILE:", os.path.relpath(file, data_path))

    data = loadmat(file)

    for key, value in data.items():
        if key.startswith("__"):
            continue

        print("Variable:", key)
        print("Shape:", value.shape)
        print("Data type:", value.dtype)
        print("Complex:", np.iscomplexobj(value))

        if value.size:
            print("Min:", np.nanmin(value))
            print("Max:", np.nanmax(value))
