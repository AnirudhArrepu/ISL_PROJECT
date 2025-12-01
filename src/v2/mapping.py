import numpy as np

def map_theta_to_joint_angles(theta):
    """
    theta order:
        [0] right_elbow
        [1] left_elbow
        [2] right_knee
        [3] left_knee
        [4] right_ankle
        [5] left_ankle
        [6] right_shoulder
        [7] left_shoulder
    """
    re, le, rk, lk, ra, la, rs, ls = theta

    # Mirror signs for right side because image coordinates differ from robot coordinates
    return np.array([
        0.0,        # pelvis_to_torso (fixed)
        0.0,        # torso_to_head (fixed)
        ls,         # left shoulder
        le,         # left elbow
        -rs,        # right shoulder
        -re,        # right elbow
        lk,         # left hip (0 for now)
        lk,         # left knee
        la,         # left ankle
        -rk,        # right hip (0 for now)
        -rk,        # right knee
        -ra         # right ankle
    ], dtype=np.float32)
