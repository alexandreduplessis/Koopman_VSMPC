from gym.envs.registration import register

register(
    id='vs-v0',
    entry_point='gym_vs.envs:VsEnv',
)