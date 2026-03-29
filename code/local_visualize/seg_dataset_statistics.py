#%%
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

#%%
# path
MYS3DIS_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyS3DIS_seg" # output

#%%
file_list = [os.path.join(MYS3DIS_RADAR_PATH, f) for f in os.listdir(MYS3DIS_RADAR_PATH) if f.endswith('.npy')]

class_count = np.zeros(4, dtype=int)
class_v_dist = [[] for _ in range(4)]
for f in tqdm(file_list):
    data = np.load(f)

    velocities = data[:, 3]
    labels = data[:, 4].astype(int)
    
    for class_idx in range(4):
        mask = (labels == class_idx)
        
        # Update point count
        class_count[class_idx] += np.sum(mask)
        
        # Append velocity values for this class
        class_v_dist[class_idx].extend(velocities[mask].tolist())

#%%
# Summary Output
classes = ["Car", "Building", "Pole", "Clutter"]
print("--- Dataset Statistics ---")
for i, name in enumerate(classes):
    avg_v = np.mean(class_v_dist[i]) if class_v_dist[i] else 0
    std_v = np.std(class_v_dist[i]) if class_v_dist[i] else 0
    print(f"{name}: {class_count[i]} points | Avg Velocity: {avg_v:.2f} | Std Dev: {std_v:.2f}")

#%%
class_names = ['car', 'building', 'pole', 'clutter']
class_colors = ["#e6793e", '#0080ff', '#ff66ff', '#c0c0c0']

#%%
plt.figure()
plt.pie(
    class_count,
    labels=class_names,
    autopct='%1.1f%%',
    startangle=90,
    colors=class_colors
)
plt.title('Class Distribution')
plt.axis('equal')  # Makes it a circle (pancake shape)

plt.show()

#%%
plt.figure()

bins = 100

for i in range(4):
    v = np.array(class_v_dist[i])
    
    # Compute histogram manually
    hist, bin_edges = np.histogram(v, bins=bins)
    
    # Normalize to percentage
    hist = hist / hist.sum() * 100
    
    # Compute bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.plot(
        bin_centers,
        hist,
        label=class_names[i],
        color=class_colors[i]
    )

plt.xlabel('Velocity')
plt.ylabel('Percentage (%)')
plt.title('Class-Specific Velocity Profiles')
plt.legend()

plt.show()