#%%
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

#%%
# path
MYS3DIS_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyS3DIS_seg" # output
MYS3DIS_STATISTICS_PATH = r"G:\我的云端硬盘\THESIS\code\local_visualize\data\seg_statistics"
file_list = [os.path.join(MYS3DIS_RADAR_PATH, f) for f in os.listdir(MYS3DIS_RADAR_PATH) if f.endswith('.npy')]

#%%
# point cloud size distribution
num_ps_vec = []
# class distribution
class_count = np.zeros(4, dtype=int)
# class-specific velocity profile
class_v_dist = [[] for _ in range(4)]

for f in tqdm(file_list):
    data = np.load(f)
    velocities = data[:, 3]             
    labels = data[:, 4].astype(int)

    num_ps_vec.append(data.shape[0])    # point cloud size distribution
    
    for class_idx in range(4):
        mask = (labels == class_idx)
        
        # Update point count
        class_count[class_idx] += np.sum(mask)
        # Append velocity values for this class
        class_v_dist[class_idx].extend(velocities[mask].tolist())

#%%
#  Visualization 
# ==================================================
# point cloud size distribution
num_ps_vec = np.array(num_ps_vec)
unique_values, counts = np.unique(num_ps_vec, return_counts=True)
plt.figure()
plt.hist(num_ps_vec, bins=30)
plt.xlabel("Number of points per frame")
plt.ylabel("Frequency")
plt.title("Point Cloud Size Distribution (Frame)")
plt.show()

print("mean:", num_ps_vec.mean())   
print("min:", num_ps_vec.min())    
print("max:", num_ps_vec.max())    
'''
mean: 1648.9441666666667
min: 1591
max: 1703
'''

#%%
# statistics
classes = ["Car", "Building", "Pole", "Clutter"]
print("--- Dataset Statistics ---")
for i, name in enumerate(classes):
    avg_v = np.mean(class_v_dist[i]) if class_v_dist[i] else 0
    std_v = np.std(class_v_dist[i]) if class_v_dist[i] else 0
    print(f"{name}: {class_count[i]} points | Avg Velocity: {avg_v:.2f} | Std Dev: {std_v:.2f}")

class_names = ['car', 'building', 'pole', 'clutter']
class_colors = ["#e6793e", '#0080ff', '#ff66ff', '#c0c0c0']

'''
--- Dataset Statistics ---
Car: 57239 points | Avg Velocity: -0.84 | Std Dev: 4.02
Building: 1197390 points | Avg Velocity: 0.00 | Std Dev: 0.10
Pole: 80014 points | Avg Velocity: 0.00 | Std Dev: 0.00
Clutter: 644090 points | Avg Velocity: 0.04 | Std Dev: 0.47
'''

#%%
# class distribution
plt.figure()
plt.pie(
    class_count,
    labels=class_names,
    autopct='%1.1f%%',
    startangle=90,
    colors=class_colors
)
plt.title('Class Distribution')
plt.axis('equal')  # Makes it a circle 
plt.show()

#%%
# Class-Specific Velocity Profiles
plt.figure()
bins = 50
for i in range(4):
    v = np.array(class_v_dist[i])
    print(f"Mean velocity of {class_names[i]} is {v.mean()}")
    
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

#%%
# Velocity Composition
def plot_velocity_class_composition(class_v_dist, class_names, bin_size=2, max_v=8):
    bins = np.arange(-max_v + bin_size, max_v + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2
    
    hist_data = []
    for v_list in class_v_dist:
        counts, _ = np.histogram(v_list, bins=bins)
        hist_data.append(counts)
    
    hist_data = np.array(hist_data)  # Shape: (4, num_bins)
    
    bin_totals = hist_data.sum(axis=0)
    
    # bin_totals[bin_totals == 0] = 1 

    percentages = (hist_data / bin_totals) * 100
    
    plt.figure(figsize=(10, 6))
    bottom_val = np.zeros(len(bin_centers))
    colors = ["#e6793e", '#0080ff', '#ff66ff', '#c0c0c0'] # Standard distinct colors

    for i in range(len(class_names)):
        plt.bar(bin_centers, percentages[i], width=bin_size*0.8, 
                bottom=bottom_val, label=class_names[i], color=colors[i])
        bottom_val += percentages[i]

    # plt.title('Velocity Composition', fontsize=14)
    plt.xlabel('Velocity (m/s)', fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=16)
    plt.xticks(bins[::2]) # Show every second bin label for clarity
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Usage ---
# Replace with your actual class names
plot_velocity_class_composition(class_v_dist, class_names)

#%%
#   save the result
# ==================================================
path = os.path.join(MYS3DIS_STATISTICS_PATH, "num_ps_vec.npy")
np.save(path, num_ps_vec)
path = os.path.join(MYS3DIS_STATISTICS_PATH, "class_count.npy")
np.save(path, class_count)
path = os.path.join(MYS3DIS_STATISTICS_PATH, "num_ps_vec.npz")
np.savez_compressed(path, *[np.array(v) for v in class_v_dist])


# Load npz
# data = np.load('class_v_dist.npz')
# # Access them like a dictionary: data['arr_0'], data['arr_1'], etc.
# loaded_data = [data[key] for key in data.files]