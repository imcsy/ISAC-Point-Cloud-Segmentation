import numpy as np

def is_pole(point, pole_box_list):

    for box in pole_box_list:
        xmin, xmax, ymin, ymax, zmin, zmax = box
        x, y, z = point
        if xmin < x < xmax and ymin < y < ymax and zmin < z < zmax:
            return True

    return False