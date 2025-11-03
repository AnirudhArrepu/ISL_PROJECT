import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

class HumanoidWalkEnv(gym.Env):
    """
    Custom Humanoid Environment.
    Initializes a humanoid URDF and allows resetting to a specific initial pose vector.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, urdf_path="../full_humanoid.urdf", render=False):
        super().__init__()
        self.render_mode = render

        # Connect to PyBullet
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load plane
        self.plane = p.loadURDF("plane.urdf")
        self.timestep = 1.0 / 240.0

        # Load humanoid
        self.urdf_path = urdf_path
        self.humanoid = p.loadURDF(self.urdf_path, [0, 0, 1.0])

        # Get controllable joints (exclude fixed joints)
        self.joint_indices = [i for i in range(p.getNumJoints(self.humanoid))
                              if p.getJointInfo(self.humanoid, i)[2] != p.JOINT_FIXED]
        
        for i in range(p.getNumJoints(self.humanoid)):
            jinfo = p.getJointInfo(self.humanoid, i)
            print(i, jinfo[1].decode('utf-8'), jinfo[12])  # joint name + axis


        n_joints = len(self.joint_indices)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(n_joints * 2,), dtype=np.float32)

    def reset(self, initial_pose=None, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.humanoid = p.loadURDF(self.urdf_path, [0, 0, 1.0])

        # Reset joints to initial pose if provided
        if initial_pose is not None:
            for i, theta in zip(self.joint_indices, initial_pose):
                p.resetJointState(self.humanoid, i, float(theta))

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # Apply action as torques
        for i, torque in zip(self.joint_indices, action):
            p.setJointMotorControl2(
                self.humanoid,
                i,
                p.TORQUE_CONTROL,
                force=float(torque) * 10.0  # scale factor for torque
            )

        p.stepSimulation()

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_termination()
        return obs, reward, done, False, {}

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
