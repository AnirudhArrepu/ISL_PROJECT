import cv2
import mediapipe as mp
import numpy as np

def compute_joint_angle(a, b, c):
    """
    Angle ABC in 2D (returns flexion angle for hinge joints).
    a,b,c are (x,y,conf) OR pixel coords.
    """
    a, b, c = np.array(a)[:2], np.array(b)[:2], np.array(c)[:2]
    ab = a - b
    cb = c - b

    dot = np.dot(ab, cb)
    denom = np.linalg.norm(ab) * np.linalg.norm(cb)
    if denom == 0:
        return 0.0

    internal = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))
    flex = 180.0 - internal
    return flex


def mediapipe_to_openpose25(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark

    def L(idx): return np.array([lm[idx].x, lm[idx].y, lm[idx].visibility], dtype=np.float32)
    def mid(a, b):
        return np.array([(lm[a].x + lm[b].x)/2,
                         (lm[a].y + lm[b].y)/2,
                         (lm[a].visibility + lm[b].visibility)/2])

    BODY25 = [
        L(0), mid(11, 12), L(12), L(14), L(16), L(11), L(13), L(15),
        mid(23, 24), L(24), L(26), L(28), L(23), L(25), L(27),
        L(2), L(5), L(7), L(8), L(29), L(31), L(27), L(30), L(32), L(28)
    ]

    return np.stack(BODY25)


def extract_theta_init_from_image(image_path):
    kp = mediapipe_to_openpose25(image_path)
    if kp is None:
        return None

    # BODY25 index shortcuts
    RSH, REL, RWR = 2, 3, 4
    LSH, LEL, LWR = 5, 6, 7
    RHIP, RKNEE, RANK = 9, 10, 11
    LHIP, LKNEE, LANK = 12, 13, 14

    angles_deg = [
        compute_joint_angle(kp[RSH], kp[REL], kp[RWR]),   # right elbow flex
        compute_joint_angle(kp[LSH], kp[LEL], kp[LWR]),   # left elbow flex
        compute_joint_angle(kp[RHIP], kp[RKNEE], kp[RANK]),  # right knee flex
        compute_joint_angle(kp[LHIP], kp[LKNEE], kp[LANK]),  # left knee flex
        compute_joint_angle(kp[RKNEE], kp[RANK], kp[22]) - 90.0,  # right ankle
        compute_joint_angle(kp[LKNEE], kp[LANK], kp[19]) - 90.0,  # left ankle
        compute_joint_angle(kp[1], kp[RSH], kp[REL]),    # right shoulder
        compute_joint_angle(kp[1], kp[LSH], kp[LEL]),    # left shoulder
    ]

    angles = np.radians(np.array(angles_deg, dtype=np.float32))
    angles = np.clip(angles, np.radians(-90), np.radians(90))
    return angles
