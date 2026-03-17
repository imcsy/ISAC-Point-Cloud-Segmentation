import numpy as np

def is_car(point, car_corners_list):

    for corners in car_corners_list:

        # box center
        center = corners.mean(axis=0)

        # box axes
        axis_x = corners[1] - corners[0]
        axis_y = corners[3] - corners[0]
        axis_z = corners[4] - corners[0]

        lx = np.linalg.norm(axis_x)
        ly = np.linalg.norm(axis_y)
        lz = np.linalg.norm(axis_z)

        axis_x /= lx
        axis_y /= ly
        axis_z /= lz

        rel = point - center

        proj_x = rel @ axis_x
        proj_y = rel @ axis_y
        proj_z = rel @ axis_z

        if (
            abs(proj_x) <= lx/2 and
            abs(proj_y) <= ly/2 and
            abs(proj_z) <= lz/2
        ):
            return True

    return False