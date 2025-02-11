from gymnasium.envs.registration import register

register(
    id="pong_env/GridWorld-v0",
    entry_point="pong_env.envs:GridWorldEnv",
)
