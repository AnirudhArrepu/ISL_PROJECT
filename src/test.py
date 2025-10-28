import cv2
import mediapipe as mp
import numpy as np
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import time

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
        raise ValueError("No person detected in image.")
    
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
        get_landmark(31),               # LBigToe
        get_landmark(32),               # LSmallToe
        get_landmark(29),               # LHeel
        get_landmark(28),               # RBigToe
        get_landmark(30),               # RSmallToe
        get_landmark(27),               # RHeel
    ]

    keypoints_25 = np.stack(mapping)
    return keypoints_25

# Example usage
# pose_25 = mediapipe_to_openpose25("person.jpg")
# print(pose_25.shape)   # (25, 3)
# print(pose_25)


def compute_joint_angle(a, b, c):
    """Returns the angle at point b formed by (a-b-c) in degrees."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    angle = np.degrees(
        np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    )
    return angle + 360 if angle < 0 else angle

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

    theta_init = np.array(list(angles.values()), dtype=np.float32)
    return theta_init


def mediapipe_to_openpose25_3d(image_path):
    """Extracts 25 OpenPose-style 3D keypoints from a single RGB image using MediaPipe world landmarks."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    if not results.pose_world_landmarks:
        raise ValueError("No 3D landmarks detected. Try a clearer full-body image.")

    lm = results.pose_world_landmarks.landmark

    def get_landmark(idx):
        l = lm[idx]
        return np.array([l.x, l.y, l.z], dtype=np.float32)

    def midpoint(a, b):
        la, lb = lm[a], lm[b]
        return np.array([(la.x + lb.x)/2, (la.y + lb.y)/2, (la.z + lb.z)/2], dtype=np.float32)

    # BODY_25 mapping in 3D
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
        get_landmark(31),               # LBigToe
        get_landmark(32),               # LSmallToe
        get_landmark(29),               # LHeel
        get_landmark(28),               # RBigToe
        get_landmark(30),               # RSmallToe
        get_landmark(27),               # RHeel
    ]

    keypoints_25 = np.stack(mapping)
    return keypoints_25  # shape (25, 3)


def compute_3d_angle(a, b, c):
    """Compute the 3D joint angle at point b formed by (a-b-c) in degrees."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def skeleton_to_theta_init_3d(keypoints_25):
    """
    Converts 25 3D keypoints to approximate humanoid joint angles θ_init.
    """
    # Indices from BODY_25
    NOSE, NECK = 0, 1
    R_SHOULDER, R_ELBOW, R_WRIST = 2, 3, 4
    L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7
    MID_HIP, R_HIP, R_KNEE, R_ANKLE = 8, 9, 10, 11
    L_HIP, L_KNEE, L_ANKLE = 12, 13, 14
    R_BIGTOE, L_BIGTOE = 22, 19

    kp = keypoints_25

    def safe_angle(a, b, c):
        try:
            return compute_3d_angle(kp[a], kp[b], kp[c])
        except Exception:
            return 0.0

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
        "r_ankle": safe_angle(R_KNEE, R_ANKLE, R_BIGTOE),
        "l_ankle": safe_angle(L_KNEE, L_ANKLE, L_BIGTOE),
    }

    theta_init = np.array(list(angles.values()), dtype=np.float32)
    return theta_init

def extract_theta_init_from_image(image_path):
    keypoints_25 = mediapipe_to_openpose25_3d(image_path)
    theta_init = skeleton_to_theta_init_3d(keypoints_25)
    print("3D θ_init (degrees):", theta_init)
    return np.radians(theta_init)  # convert to radians for humanoid

# # Example usage
# theta_init = skeleton_to_theta_init(pose_25)
# print("θ_init =", theta_init)

# theta_init = np.radians(theta_init)  # convert to radians if needed


def run_pipeline():
    theta_init = extract_theta_init_from_image("../person.jpg")

    env = HumanoidWalkEnv(render=True)
    state_dim = env.observation_space.shape[0]
    n_joints = len(env.joint_indices)

    # Initialize DQN agent
    agent = DQNAgent(state_dim, n_joints, n_bins=5, device='cpu')

    # Training loop 
    num_episodes = 100
    max_steps = 1500  # safety limit to prevent infinite loops

    for ep in range(num_episodes):
        # state, _ = env.reset(initial_pose=theta_init)  
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
            torques = agent.indices_to_torques(action_indices)*50

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

    num_test_episodes = 5
    max_steps = 1500

    for ep in range(num_test_episodes):
        state, _ = env.reset(initial_pose=np.zeros(n_joints))
        pos, _ = p.getBasePositionAndOrientation(env.humanoid)
        prev_com = np.array(pos)
        total_reward = 0.0

        for step in range(max_steps):
            # Get action from policy
            action_indices = agent.q_net.act(state, epsilon=0.0)
            torques = agent.indices_to_torques(action_indices) * 50

            # Apply to environment
            next_state, _, done, _, _ = env.step(torques)

            # Compute reward (optional for analysis)
            reward, fallen, new_com = agent.compute_reward(env, prev_com, torques, step)
            total_reward += reward

            state = next_state
            prev_com = new_com

            # Real-time visualization if you’re using PyBullet GUI
            time.sleep(1/60)

            if done or fallen:
                break

        print(f"[TEST] Episode {ep+1}: Total Reward = {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    run_pipeline()

