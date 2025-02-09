import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)
env = gym.make('ALE/Pong-v5', render_mode="rgb_array")
env = AtariWrapper(env)  
env = DummyVecEnv([lambda: env])  
env = VecFrameStack(env, n_stack=4)  

model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0005, 
            n_steps=2048, batch_size=256, n_epochs=200)
model.learn(total_timesteps=128000)  
model.save("pong")

env.close()

print(env)
