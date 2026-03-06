## GOAL

1. segmentation on ISAC-generated point cloud
2. attacks like injecting points or eliminating points
3. improve robustness under attacks

## Environment

Colab T4 GPU

## Dataset

Multimodel-Wireless: https://le-liang.github.io/mmw/index.html

## Pipeline

1. segmentation on ISAC-generated point cloud
   - use pretrained model ```randlanet_semantickitti from open3d.ml.torch``` to label LiDAR dataset (car, building)
   - move labels to radar point cloud
   - train model to segment radar point cloud