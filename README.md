## GOAL

1. segmentation on ISAC-generated point cloud, car detection
2. attacks like injecting points or eliminating points
3. improve robustness under attacks

## Environment

Colab T4 GPU

## Dataset

Multimodel-Wireless: https://le-liang.github.io/mmw/index.html

## Pipeline

1. segmentation on ISAC-generated point cloud
   - use pretrained model ```randlanet_semantickitti from open3d.ml.torch```  and range filter to label LiDAR dataset (building, road)
   - use bounding box to label cars
   - turn json file of radar point cloud into pcd file
   - calibrate all coordinates (lidar, radar, boxes)
   - move labels to radar point cloud
   - train model to segment radar point cloud