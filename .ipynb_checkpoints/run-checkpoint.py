import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Create the environment
env = gym.make('ALE/Pong-v5', render_mode="human")
env = AtariWrapper(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

# Load the trained model
model = PPO.load("pong.zip", env=env)

# Test the model
obs = env.reset()

for _ in range(1000):  # Run for 1000 steps
    action, _ = model.predict(obs)  # Predict action
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
