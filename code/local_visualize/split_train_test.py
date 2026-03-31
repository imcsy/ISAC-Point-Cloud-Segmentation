#%%
import random
import os

#%%
MYMODELNET_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyModelNet_cls"

#%%
N = 7116 # 6268
name = "clutter" # car
training_percent = 0.7

indices = list(range(1, N+1))

#%%
random.shuffle(indices)
train_size = int(training_percent * N)

train_idx = indices[:train_size]
test_idx = indices[train_size:]

#%%
training_file_ls = []
for i in train_idx:
    training_file_ls.append(f"{name}_{i:05d}")

path = os.path.join(MYMODELNET_PATH, "modelnet2_train.txt")
with open(path, "a") as f:
    f.write("\n".join(training_file_ls))

#%%
test_file_ls = []
for i in test_idx:
    test_file_ls.append(f"{name}_{i:05d}")

path = os.path.join(MYMODELNET_PATH, "modelnet2_test.txt")
with open(path, "a") as f:
    f.write("\n".join(test_file_ls))