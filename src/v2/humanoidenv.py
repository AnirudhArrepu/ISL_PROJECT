import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

class HumanoidWalkEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, urdf_path="../../full_humanoid.urdf", render=False):
        super().__init__()
        self.render_mode = render

        p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")

        self.urdf_path = urdf_path
        self.humanoid = p.loadURDF(self.urdf_path, [0, 0, 1.0])

        self.joint_indices = [i for i in range(p.getNumJoints(self.humanoid))
                              if p.getJointInfo(self.humanoid, i)[2] != p.JOINT_FIXED]

        n = len(self.joint_indices)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,))
        self.observation_space = spaces.Box(-10, 10, shape=(2*n,))

        self.timestep = 1/240.0

    def reset(self, initial_pose=None, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.humanoid = p.loadURDF(self.urdf_path, [0, 0, 1.0])

        if initial_pose is not None:
            for j, angle in zip(self.joint_indices, initial_pose):
                p.resetJointState(self.humanoid, j, angle)

        return self._get_obs(), {}

    def step(self, action):
        # Apply action as torques
        for idx, j in enumerate(self.joint_indices):
            # Get allowable joint effort from URDF (max torque)
            jinfo = p.getJointInfo(self.humanoid, j)
            effort_limit = jinfo[10]   # PyBullet stores "effort" as field 10

            # Apply torque using normalized action [-1,1]
            applied_torque = float(action[idx] * effort_limit)

            p.setJointMotorControl2(
                bodyIndex=self.humanoid,
                jointIndex=j,
                controlMode=p.TORQUE_CONTROL,
                force=applied_torque*3
            )

        p.stepSimulation()

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_termination()
        return obs, reward, done, False, {}

    def _get_obs(self):
        js = p.getJointStates(self.humanoid, self.joint_indices)
        pos = [s[0] for s in js]
        vel = [s[1] for s in js]
        return np.array(pos + vel, dtype=np.float32)
    
    def _get_observation(self):
        joint_states = p.getJointStates(self.humanoid, self.joint_indices)
        positions = [s[0] for s in joint_states]
        velocities = [s[1] for s in joint_states]
        return np.array(positions + velocities, dtype=np.float32)

    def _compute_reward(self):
        # Reward: forward progress of the torso
        pos, _ = p.getBasePositionAndOrientation(self.humanoid)
        return pos[0]

    def _check_termination(self):
        # Terminate if humanoid falls
        pos, _ = p.getBasePositionAndOrientation(self.humanoid)
        return pos[2] < 0.3

    def close(self):
        p.disconnect()