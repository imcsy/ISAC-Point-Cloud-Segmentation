import numpy as np

def get_rotation_matrix(rot):
    roll, pitch, yaw = rot

    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    Rx = np.array([
        [1,0,0],
        [0,np.cos(roll),-np.sin(roll)],
        [0,np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch),0,np.sin(pitch)],
        [0,1,0],
        [-np.sin(pitch),0,np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw),-np.sin(yaw),0],
        [np.sin(yaw), np.cos(yaw),0],
        [0,0,1]
    ])

    return Rz @ Ry @ Rx