import open3d as o3d
import numpy as np

def create_3d_bbox(xmin, xmax, ymin, ymax, zmin, zmax, color=[1, 0, 0]):
    """
    Creates a 3D LineSet box (cuboid) from min/max bounds.
    color: [R, G, B] normalized to 0-1.
    """
    # 1. Define the 8 corners of the cuboid
    points = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin], # Bottom 4
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]  # Top 4
    ])

    # 2. Define the 12 lines (edges) connecting the points
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0], # Bottom face lines
        [4, 5], [5, 6], [6, 7], [7, 4], # Top face lines
        [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical pillars
    ]

    # 3. Create the LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # 4. Apply the color
    colors = [color for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set