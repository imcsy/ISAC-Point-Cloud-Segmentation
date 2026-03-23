import os
import sys
from indoor3d_util import DATA_PATH, collect_point_label

RAW_S3DIS_DIR = '/content/drive/MyDrive/THESIS_dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version'
OUTPUT_DIR = '/content/drive/MyDrive/THESIS_dataset/S3DIS/stanford_indoor3d'
sys.path.append(RAW_S3DIS_DIR)

anno_paths = [line.rstrip() for line in open(os.path.join(RAW_S3DIS_DIR, 'meta/anno_paths.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
        collect_point_label(anno_path, os.path.join(OUTPUT_DIR, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!!')
