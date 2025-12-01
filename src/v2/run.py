import os
import numpy as np
import torch
import pybullet as p

from pose_extraction import extract_theta_init_from_image
from mapping import map_theta_to_joint_angles
from humanoidenv import HumanoidWalkEnv
from agent import DQNAgent

# -------------------------------
# Checkpoint utilities
# -------------------------------
CHECKPOINT_DIR = "../../checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(agent, episode):
    path = os.path.join(CHECKPOINT_DIR, f"dqn_ep{episode}.pth")
    torch.save({
        "q_net": agent.q_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "steps_done": agent.steps_done,
    }, path)
    print(f"[CHECKPOINT SAVED] {path}")


def load_checkpoint(agent, path):
    ckpt = torch.load(path, map_location=agent.device)
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.target_net.load_state_dict(ckpt["target_net"])
    agent.optimizer.load_state_dict(ckpt["optimizer"])
    agent.epsilon = ckpt["epsilon"]
    agent.steps_done = ckpt["steps_done"]
    print(f"[CHECKPOINT LOADED] {path}")


# ------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------
def train(
    num_episodes=500,
    max_steps=1500,
    image_path="../poses/pose.jpg",
    checkpoint_every=50,
    resume_path=None
):

    # 1. Extract pose from image
    theta = extract_theta_init_from_image(image_path)
    if theta is None:
        print("Pose not detected in image.")
        return

    init_pose = map_theta_to_joint_angles(theta)

    # 2. Create environment
    env = HumanoidWalkEnv(render=True)
    state_dim = env.observation_space.shape[0]
    n_joints = len(env.joint_indices)

    # 3. Agent
    agent = DQNAgent(state_dim, n_joints, n_bins=5, device="cpu")

    # Resume from checkpoint if available
    if resume_path:
        load_checkpoint(agent, resume_path)

    print("Training begins...")

    # -------------------------
    # Main training loop
    # -------------------------
    for ep in range(1, num_episodes + 1):

        # Reset with initial pose
        state, _ = env.reset(initial_pose=init_pose)

        # Get COM for velocity reward
        prev_pos, _ = p.getBasePositionAndOrientation(env.humanoid)
        prev_com = np.array(prev_pos)

        total_reward = 0
        done = False

        # Step loop
        for step in range(max_steps):

            # Select discrete action per joint
            action_bins = agent.q_net.act(state, epsilon=agent.epsilon)

            # Convert bins -> torques
            torques = agent.indices_to_torques(action_bins) #* 4.0

            # Step environment
            next_state, _, env_done, _, _ = env.step(torques)

            # Compute custom reward from your agent class
            reward, fallen, new_com = agent.compute_reward(
                env, prev_com, torques, step
            )

            done = env_done or fallen

            # Store transition
            agent.remember(state, action_bins, reward, next_state, done)
            agent.update()

            # Move to next step
            state = next_state
            prev_com = new_com
            total_reward += reward

            if done:
                break

        print(f"Episode {ep:04d} | Steps: {step+1:03d} | Reward: {total_reward:8.3f} | Epsilon: {agent.epsilon:.3f}")

        # Save checkpoint periodically
        if ep % checkpoint_every == 0:
            save_checkpoint(agent, ep)

    env.close()
    print("Training finished!")


# ------------------------------------------------------------
# INFERENCE / EVALUATION
# ------------------------------------------------------------
def inference(checkpoint_path, image_path="../poses/pose.jpg"):

    theta = extract_theta_init_from_image(image_path)
    if theta is None:
        print("No pose detected.")
        return

    init_pose = map_theta_to_joint_angles(theta)

    # Environment with GUI
    env = HumanoidWalkEnv(render=True)

    state_dim = env.observation_space.shape[0]
    n_joints = len(env.joint_indices)
    agent = DQNAgent(state_dim, n_joints, n_bins=5, device="cpu")

    load_checkpoint(agent, checkpoint_path)
    agent.epsilon = 0.0  # Greedy inference

    print("Inference running...")

    while True:
        state, _ = env.reset(initial_pose=init_pose)

        while True:
            action_idx = agent.q_net.act(state, epsilon=0.0)
            torques = agent.indices_to_torques(action_idx) * 40.0

            next_state, _, done, _, _ = env.step(torques)
            state = next_state

            # if done:
            #     print("Humanoid fell. Restarting...")
            #     break


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    # ---- Run Training ----
    train(
        num_episodes=2000,
        max_steps=1500,
        image_path="../../poses/1_the-departed-00105821.jpg",
        checkpoint_every=50,
        resume_path= "../../checkpoints/dqn_ep2000.pth"  # or "./checkpoints/dqn_ep200.pth"
    )

    # ---- Or Run Inference ----
    # inference("../../checkpoints/dqn_ep2000.pth", image_path="../../poses/1_the-departed-00105821.jpg")
