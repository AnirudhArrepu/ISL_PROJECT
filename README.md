# ISL_PROJECT

## Modules

### Module 1
- extracting a 25 point skeleton from a multi person image. 
- costructing an angle joint represetnation of the skeleton
  
### Module 2
- Constructing an environment compatible with pybullet for physics simulation encapsulating the openai's gym environment
  
### Module 3
- constructing a reward function, action spaces, observation space for a DQN agent to learn to walk from the pose given in teh input image at module 1

## File Structure
- `setup.py`: initialises the package of this current pipleine. compatible to install using `pip install -e .`
- `/poses`: contains the initial poses the agent has been trained on.
- `\src\agent.py`: contains the DQN agent implementation
- `\src\humanoidenv.py`: contains the pybullet env
- `\src\skeleton.py`: extracts a 25 key point from the pose given in the image, constructs an angle representation and passes it to the agent to (learn) to walk
- `src\test.py`: contains the pipeline to run the aforesaid modules
- `humanoid.urdf`: urdf code for the body the agent uses to walk
- `\reports`: contains reports of weekly updates. 

