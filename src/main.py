import cv2
import mediapipe as mp
import numpy as np
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import time
import os

cv2.startWindowThread()  

#relative imports
from humanoidenv import HumanoidWalkEnv
from agent import DQNAgent

def mediapipe_to_openpose25(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    if not results.pose_landmarks:
        return None
        raise ValueError("No person detected in image.", image_path)
    
    lm = results.pose_landmarks.landmark

    # Function to get (x, y, confidence)
    def get_landmark(idx):
        l = lm[idx]
        return np.array([l.x, l.y, l.visibility], dtype=np.float32)

    # Compute midpoint helper
    def midpoint(a, b):
        la, lb = lm[a], lm[b]
        return np.array([(la.x + lb.x) / 2, (la.y + lb.y) / 2, (la.visibility + lb.visibility) / 2], dtype=np.float32)

    # BODY_25 mapped from MediaPipe
    mapping = [
        get_landmark(0),                # Nose
        midpoint(11, 12),               # Neck
        get_landmark(12),               # RShoulder
        get_landmark(14),               # RElbow
        get_landmark(16),               # RWrist
        get_landmark(11),               # LShoulder
        get_landmark(13),               # LElbow
        get_landmark(15),               # LWrist
        midpoint(23, 24),               # MidHip
        get_landmark(24),               # RHip
        get_landmark(26),               # RKnee
        get_landmark(28),               # RAnkle
        get_landmark(23),               # LHip
        get_landmark(25),               # LKnee
        get_landmark(27),               # LAnkle
        get_landmark(2),                # REye
        get_landmark(5),                # LEye
        get_landmark(7),                # REar
        get_landmark(8),                # LEar
        get_landmark(29),               # 19: LBigToe
        get_landmark(31),               # 20: LSmallToe
        get_landmark(27),               # 21: LHeel
        get_landmark(30),               # 22: RBigToe
        get_landmark(32),               # 23: RSmallToe
        get_landmark(28),               # 24: RHeel
    ]

    keypoints_25 = np.stack(mapping)
    return keypoints_25

def compute_joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    dot_product = np.dot(ab, cb)
    norm_product = np.linalg.norm(ab) * np.linalg.norm(cb)
    if norm_product == 0:
        return None
    angle = np.degrees(np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0)))
    return angle

def skeleton_to_theta_init(keypoints_25):
    """
    keypoints_25: np.array (25, 3) of (x, y, conf)
    returns: np.array (n_angles,) initial pose vector θ_init
    """

    # Indices (BODY_25 reference)
    NOSE, NECK = 0, 1
    R_SHOULDER, R_ELBOW, R_WRIST = 2, 3, 4
    L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7
    MID_HIP, R_HIP, R_KNEE, R_ANKLE = 8, 9, 10, 11
    L_HIP, L_KNEE, L_ANKLE = 12, 13, 14

    kp = keypoints_25

    # Helper for missing joints
    def safe_angle(a, b, c):
        try:
            return compute_joint_angle(kp[a], kp[b], kp[c])
        except Exception:
            return 0.0

    # Angles
    angles = {
        "neck_tilt": safe_angle(MID_HIP, NECK, NOSE),
        "r_shoulder": safe_angle(NECK, R_SHOULDER, R_ELBOW),
        "r_elbow": safe_angle(R_SHOULDER, R_ELBOW, R_WRIST),
        "l_shoulder": safe_angle(NECK, L_SHOULDER, L_ELBOW),
        "l_elbow": safe_angle(L_SHOULDER, L_ELBOW, L_WRIST),
        "r_hip": safe_angle(MID_HIP, R_HIP, R_KNEE),
        "r_knee": safe_angle(R_HIP, R_KNEE, R_ANKLE),
        "l_hip": safe_angle(MID_HIP, L_HIP, L_KNEE),
        "l_knee": safe_angle(L_HIP, L_KNEE, L_ANKLE),
        "r_ankle": safe_angle(R_KNEE, R_ANKLE, 22),  # RBigToe
        "l_ankle": safe_angle(L_KNEE, L_ANKLE, 19),  # LBigToe
    }
    # apply sign conventions explicitly
    angles["r_shoulder"] *= -1.0
    angles["r_elbow"]   *= -1.0
    angles["r_hip"]     *= -1.0
    angles["r_knee"]    *= -1.0
    angles["r_ankle"]   *= -1.0

    theta_init = np.array(list(angles.values()), dtype=np.float32)
    return theta_init

def visualize_pose_on_image(image_path, keypoints_25, save_path="overlay_pose.jpg"):
    """
    Draws the reconstructed 2D skeleton on top of the input image.
    keypoints_25: np.array (25, 3) of (x, y, visibility)
    """
    # BODY_25 connections (OpenPose style)
    BODY_25_PAIRS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Right arm
        (1, 5), (5, 6), (6, 7),              # Left arm
        (1, 8),                              # Spine
        (8, 9), (9, 10), (10, 11),           # Right leg
        (8, 12), (12, 13), (13, 14),         # Left leg
        (0, 15), (0, 16),                    # Eyes
        (15, 17), (16, 18),                  # Ears
        (11, 22), (22, 23), (11, 24),        # Right foot (Ankle->BigToe, BigToe->SmallToe, Ankle->Heel)
        (14, 19), (19, 20), (14, 21)         # Left foot
    ]

    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Convert normalized coordinates to pixel coordinates
    points = []
    for i, (x, y, conf) in enumerate(keypoints_25):
        cx, cy = int(x * w), int(y * h)
        points.append((cx, cy))

        # Draw joints
        color = (0, 255, 0) if conf > 0.5 else (0, 100, 255)
        cv2.circle(image, (cx, cy), 4, color, -1)

    # Draw connections
    for (a, b) in BODY_25_PAIRS:
        if a < len(points) and b < len(points):
            pt1, pt2 = points[a], points[b]
            cv2.line(image, pt1, pt2, (255, 255, 255), 2)

    cv2.imwrite(save_path, image)
    # cv2.imshow("Pose Overlay", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Pose visualization saved to {save_path}")

def extract_theta_init_from_image(image_path):
    """
    Extracts joint angles (theta_init) using the 2D pose pipeline,
    which is better suited for PyBullet's joint angle definitions.
    """
    try:
        keypoints_25_2d = mediapipe_to_openpose25(image_path)
        if keypoints_25_2d is None:
            return 
    except ValueError as e:
        print(f"Error processing image: {e}")
        print("Falling back to a zero pose.")
        return np.zeros(11, dtype=np.float32) 
    
    theta_init = skeleton_to_theta_init(keypoints_25_2d)
    
    print("2D-based θ_init (degrees):", theta_init, flush=True)

    return np.radians(theta_init)

# import numpy as np

def map_theta_to_joint_angles(theta_init):
    """
    Maps the 11-element theta_init (neck, shoulders, elbows, hips, knees, ankles)
    to your URDF joint order (12 joints total).
    Handles radians, sign corrections, and clipping for stability.
    """
    if theta_init is None or len(theta_init) < 11:
        print("[WARN] Invalid theta_init; returning neutral pose.")
        return np.zeros(12, dtype=np.float32)

    # Convert degrees → radians if necessary
    theta = np.asarray(theta_init, dtype=np.float32)
    if np.max(np.abs(theta)) > 2 * np.pi:
        theta = np.radians(theta)

    # Clip extreme angles for physical realism
    theta = np.clip(theta, -np.pi / 2, np.pi / 2)

    # ---- Human joint angles order ----
    # [neck, r_shoulder, r_elbow, l_shoulder, l_elbow,
    #  r_hip, r_knee, l_hip, l_knee, r_ankle, l_ankle]

    # ---- Map to URDF order ----
    # (tuned so positive angles bend naturally)
    mapping = {
        "torso": 0.1 * (theta[5] + theta[7]),  # small spine lean avg of hips
        "neck": theta[0] * 0.5,
        "l_shoulder": theta[3],
        "l_elbow": theta[4],
        "r_shoulder": -theta[1],
        "r_elbow": -theta[2],
        "l_hip": theta[7],
        "l_knee": theta[8],
        "l_ankle": theta[10],
        "r_hip": -theta[5],
        "r_knee": -theta[6],
        "r_ankle": -theta[9],
    }

    # Final order according to your environment
    ordered = [
        mapping["torso"],       # 0 pelvis_to_torso
        mapping["neck"],        # 1 torso_to_head
        mapping["l_shoulder"],  # 2 torso_to_left_shoulder
        mapping["l_elbow"],     # 3 left_elbow
        mapping["r_shoulder"],  # 4 torso_to_right_shoulder
        mapping["r_elbow"],     # 5 right_elbow
        mapping["l_hip"],       # 6 pelvis_to_left_hip
        mapping["l_knee"],      # 7 left_knee
        mapping["l_ankle"],     # 8 left_ankle
        mapping["r_hip"],       # 9 pelvis_to_right_hip
        mapping["r_knee"],      # 10 right_knee
        mapping["r_ankle"],     # 11 right_ankle
    ]

    return np.array(ordered, dtype=np.float32)

def run_images_skeleton_extraction(image_dir):
    print("here")
    output_dir = os.path.join("../outputs", "skeletons")
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            
            keypoints_25 = mediapipe_to_openpose25(image_path)
            if keypoints_25 is None:
                continue
            
            output_path = os.path.join(output_dir, filename)
            visualize_pose_on_image(image_path, keypoints_25, save_path=output_path)  

def run_pipeline():
    theta_init_2d = extract_theta_init_from_image("../poses/1_the-departed-00105821.jpg")

    env = HumanoidWalkEnv(render=True)
    state_dim = env.observation_space.shape[0]
    n_joints = len(env.joint_indices)

    theta_init_3d = map_theta_to_joint_angles(theta_init_2d)

    # Clamp angles to prevent self-folding
    theta_init_3d = np.clip(theta_init_3d, np.radians(-90), np.radians(90))

    # Initialize DQN agent
    agent = DQNAgent(state_dim, n_joints, n_bins=5, device='cpu')

    agent.epsilon=1
    print(agent.epsilon)

    # Training loop 
    num_episodes = 1000
    max_steps = 1500  # safety limit to prevent infinite loops

    for ep in range(num_episodes):
        # state, _ = env.reset(initial_pose=theta_init_3d)  
        state, _ = env.reset(initial_pose=np.zeros(n_joints))  # can replace with pose-based init
        
        pos, _ = p.getBasePositionAndOrientation(env.humanoid)
        prev_com = np.array(pos)
        done = False
        total_reward = 0.0
        step_count = 0

        while not done and step_count < max_steps:
            # Select discrete joint actions
            action_indices = agent.q_net.act(state, epsilon=agent.epsilon)

            # Convert to torques and scale moderately
            torques = agent.indices_to_torques(action_indices)*100

            # Apply torques through the environment
            next_state, _, env_done, _, _ = env.step(torques)

            # Compute custom reward using actual COM displacement
            reward, fallen, new_com = agent.compute_reward(env, prev_com, torques, step_count)

            # Determine done condition (either environment or fall)
            done = env_done or fallen

            # Store transition
            agent.remember(state, action_indices, reward, next_state, done)
            agent.update()

            # Advance
            state = next_state
            prev_com = new_com
            total_reward += reward
            step_count += 1

        print(f"Episode {ep+1:03d} | Steps: {step_count:4d} | Total Reward: {total_reward:8.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    run_pipeline()
    # env = HumanoidWalkEnv(render=True)
    # theta_init_2d = extract_theta_init_from_image("../poses/1_the-departed-00105821.jpg")
    # theta_init_3d = map_theta_to_joint_angles(theta_init_2d)
    # env.reset(initial_pose=theta_init_3d)
    # input("Press Enter to exit...")
    # run_images_skeleton_extraction("../poses")

