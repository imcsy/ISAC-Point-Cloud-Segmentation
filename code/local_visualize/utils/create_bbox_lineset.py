import open3d as o3d

def create_bbox_lineset(car_corners_list):

    lines = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]
    colors = [[1,0,0] for _ in lines]

    bbox_list = []
    for car_corners in car_corners_list:
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(car_corners), 
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbox_list.append(line_set)

    return bbox_list