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
    cv2.imshow("Pose Overlay", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Pose visualization saved to {save_path}")

def extract_theta_init_from_image(image_path):
    """
    Extracts joint angles (theta_init) using the 2D pose pipeline,
    which is better suited for PyBullet's joint angle definitions.
    """
    # Get 2D keypoints for visualization AND angle calculation
    try:
        keypoints_25_2d = mediapipe_to_openpose25(image_path)
    except ValueError as e:
        print(f"Error processing image: {e}")
        print("Falling back to a zero pose.")
        return np.zeros(11, dtype=np.float32) # Assuming 11 joints from your skeleton_to_theta_init

    # Use the 2D angle calculation function (skeleton_to_theta_init).
    # This function is better suited as it calculates relative angles
    # (e.g., ~0 degrees for a straight arm) which PyBullet expects,
    # unlike the 3D function which calculates internal angles (e.g., ~180 degrees).
    theta_init = skeleton_to_theta_init(keypoints_25_2d)
    
    print("2D-based θ_init (degrees):", theta_init, flush=True)

    # Visualize the 2D pose that is being used for initialization
    # visualize_pose_on_image(image_path, keypoints_25_2d, save_path="pose_overlay.jpg")

    return np.radians(theta_init)

def run_pipeline():
    theta_init = extract_theta_init_from_image("../image.png")

    env = HumanoidWalkEnv(render=True)
    state_dim = env.observation_space.shape[0]
    n_joints = len(env.joint_indices)

    # Initialize DQN agent
    agent = DQNAgent(state_dim, n_joints, n_bins=5, device='cpu')

    # Training loop 
    num_episodes = 1000
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

