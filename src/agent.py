import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from collections import deque

# ----------------------
# Q-Network
# ----------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_joints, n_bins=7, hidden_sizes=[256, 128]):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.n_joints = n_joints
        self.n_bins = n_bins
        self.output_dim = n_joints * n_bins

        layers = []
        input_size = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h

        layers.append(nn.Linear(input_size, self.output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def act(self, state, epsilon=0.1):
        """
        Epsilon-greedy action selection.
        Returns discrete indices per joint, shape [n_joints]
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n_bins, size=self.n_joints)
        with torch.no_grad():
            q_values = self.forward(torch.FloatTensor(state).unsqueeze(0))
            q_values = q_values.view(self.n_joints, self.n_bins)
            return torch.argmax(q_values, dim=1).cpu().numpy()


# ----------------------
# Replay Buffer
# ----------------------
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


# ----------------------
# DQN Agent (Modified)
# ----------------------
class DQNAgent:
    def __init__(self, state_dim, n_joints, n_bins=5, lr=1e-3, gamma=0.99,
                 epsilon_start=1, epsilon_end=0.05, epsilon_decay=0.9999,
                 buffer_size=100000, batch_size=64, target_update=500, device='cpu'):

        self.n_joints = n_joints
        self.n_bins = n_bins
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        self.target_update = target_update

        self.q_net = QNetwork(state_dim, n_joints, n_bins).to(device)
        self.target_net = QNetwork(state_dim, n_joints, n_bins).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0

        # Discretized torque bins
        self.torque_bins = np.linspace(-1.0, 1.0, n_bins)

    def indices_to_torques(self, action_indices):
        """Convert discrete action indices to continuous torque array."""
        return np.array([self.torque_bins[a] for a in action_indices], dtype=np.float32)

    def compute_reward(self, env, prev_com, torques, step_count):
        pos, orn = p.getBasePositionAndOrientation(env.humanoid)
        z = pos[2]
        com = np.array(pos)

        # Forward velocity (x direction)
        forward_vel = (com[0] - prev_com[0]) / env.timestep

        # -----------------------
        # Reward Components
        # -----------------------

        # 1. Alive reward (survival)
        alive = 1.0 if z > 0.6 else -10.0

        # 2. Forward progress reward
        r_forward = 2.0 * forward_vel

        # 3. Upright reward (penalize tilting)
        # dot product of torso up vector with world up
        _, torso_orn = p.getLinkState(env.humanoid, 1)[4:6]
        torso_up = p.getMatrixFromQuaternion(torso_orn)[6:9]
        upright = np.dot(torso_up, np.array([0, 0, 1]))   # 1 = perfectly upright

        r_upright = 2.0 * upright

        # 4. Energy / torque penalty
        r_energy = -0.001 * np.sum(np.square(torques))

        # 5. Smooth motion (joint velocity penalty)
        joint_states = p.getJointStates(env.humanoid, env.joint_indices)
        joint_vel = np.array([s[1] for s in joint_states])
        r_smooth = -0.05 * np.sum(np.square(joint_vel))

        # 6. Lateral stability (stay balanced in y axis)
        r_balance = -1.0 * abs(com[1])

        # -----------------------
        # Total reward
        # -----------------------
        reward = r_forward + alive + r_upright + r_energy + r_smooth + r_balance

        # Episode termination: fallen
        done = z < 0.3 or upright < 0.2

        return reward, done, com


    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)  # shape [batch, n_joints]

        # Current Q-values
        q_values = self.q_net(states).view(self.batch_size, self.n_joints, self.n_bins)
        q_selected = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1).mean(dim=1, keepdim=True)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).view(self.batch_size, self.n_joints, self.n_bins)
            max_next_q = next_q_values.max(dim=2)[0].mean(dim=1, keepdim=True)
            target = rewards + self.gamma * (1 - dones) * max_next_q

        # Compute loss
        loss = nn.SmoothL1Loss()(q_selected, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
        # print("new epsilon, ", self.epsilon)

        # Target network update
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.steps_done += 1
