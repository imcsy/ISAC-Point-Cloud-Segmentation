

#%%
import os
import random
import numpy as np

# Path to your folder
folder_path = "G:\我的云端硬盘\THESIS_dataset\mmw\MyModelNet_cls\car"

#%%
all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
sample_size = min(100, len(all_files))
sampled_files = random.sample(all_files, sample_size)

#%%
all_data = []
for file_name in sampled_files:
    file_path = os.path.join(folder_path, file_name)
    try:
        # Load data, assuming it's comma-separated
        data = np.loadtxt(file_path, delimiter=',')
        
        # If your files contain multiple lines, this adds all points
        if data.ndim == 1:
            all_data.append(data)
        else:
            all_data.extend(data)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# Convert to a single numpy array
all_data_array = np.array(all_data) # Shape will be (TotalPoints, 4)

#%%
stds = np.std(all_data_array, axis=0)

print(f"Standard Deviations (x, y, z, v):")
print(f"X std: {stds[0]:.6f}")
print(f"Y std: {stds[1]:.6f}")
print(f"Z std: {stds[2]:.6f}")
print(f"V std: {stds[3]:.6f}")