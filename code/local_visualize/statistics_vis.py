#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
#   classification
# ==================================================
# Path to your statistics folder
STAT_PATH = r"G:\我的云端硬盘\THESIS\code\local_visualize\data\cls_statistics"

def plot_comparison(car_data, clutter_data, title, xlabel, filename, num_bins=70):
    plt.figure(figsize=(10, 6))

    global_min = min(np.nanmin(car_data), np.nanmin(clutter_data))
    global_max = max(np.nanmax(car_data), np.nanmax(clutter_data))
    bounds = np.array([global_min, global_max])
    car_data = np.concatenate([car_data, bounds])
    clutter_data = np.concatenate([clutter_data, bounds])
    bins = np.linspace(global_min, global_max, num_bins)
    
    # Using 'stat="density"' is crucial to compare two datasets of different sizes
    sns.histplot(clutter_data, color="orange", label="Clutter", 
                 stat="count", kde=True, alpha=0.4, bins=40)
    sns.histplot(car_data, color="dodgerblue", label="Car", 
                 stat="count", kde=True, alpha=0.4, bins=40)
    
    # plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel("Frequency", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 22})
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

#%%
num_ps_car = np.load(os.path.join(STAT_PATH, "num_ps_vec_car.npy"))
num_ps_clutter = np.load(os.path.join(STAT_PATH, "num_ps_vec_clutter.npy"))

vel_car = np.load(os.path.join(STAT_PATH, "ave_vel_vec_car.npy"))
vel_clutter = np.load(os.path.join(STAT_PATH, "ave_vel_vec_clutter.npy"))

spread_car = np.load(os.path.join(STAT_PATH, "spread_vec_car.npy"))
spread_clutter = np.load(os.path.join(STAT_PATH, "spread_vec_clutter.npy"))

#%%
# num_ps_car = num_ps_car[num_ps_car < 50]
# num_ps_clutter = num_ps_clutter[num_ps_clutter < 50]

plot_comparison(num_ps_car, num_ps_clutter, 
                "Comparison: Point Cloud Size Distribution", 
                "Number of points per sample", "Fig1_Point_Count.png",  
                num_bins=70)

plot_comparison(vel_car, vel_clutter, 
                "Comparison: Average Velocity Distribution", 
                "Average Velocity (m/s)", "Fig2_Velocity.png")

plot_comparison(spread_car, spread_clutter, 
                "Comparison: Spatial Dispersion Distribution", 
                "Average Spread Distance", "Fig3_Dispersion.png")


#%%
#   segmentation
# ==================================================
STAT_PATH = r"G:\我的云端硬盘\THESIS\code\local_visualize\data\seg_statistics"

class_count = np.load(os.path.join(STAT_PATH, "class_count.npy"))

labels = ["Car", "Building", "Pole", "Clutter"]
colors = [
    (1.0, 0.502, 0.0),   # orange
    (0.0, 0.502, 1.0),   # blue
    (1.0, 0.4, 1.0),     # pink
    (0.753, 0.753, 0.753), # gray
    (1.0, 0.0, 0.0)      # red
]

plt.figure(figsize=(6,6))
plt.pie(class_count, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 16})

# plt.title("Class Distribution")
plt.axis('equal')  # Makes the pie chart circular

plt.show()

#%%
num_ps_vec = np.load(os.path.join(STAT_PATH, "num_ps_vec.npy"))
plt.figure(figsize=(10, 6))

# Using 'stat="density"' is crucial to compare two datasets of different sizes
sns.histplot(num_ps_vec, color="orange", label="Scene", 
                stat="count", kde=True, alpha=0.4, bins=40)

# plt.title(title, fontsize=15, fontweight='bold')
plt.xlabel("Number of points per sample", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.legend(prop={'size': 16})
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

#%%
vel_data = np.load(os.path.join(STAT_PATH, "num_ps_vec.npz"))
