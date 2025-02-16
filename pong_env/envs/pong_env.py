from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    stay = 0
    up = 1
    down = 2


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        pass

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def _render_frame(self):
        pass
