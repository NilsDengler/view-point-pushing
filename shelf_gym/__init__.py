from gym.envs.registration import register

register(
    id='ViewPointEnv-v0',
    entry_point='shelf_gym.vpp.vpp_rl:VewPointTask',
    kwargs={'render': False}
)
register(
    id='ViewPointEnvGUI-v0',
    entry_point='shelf_gym.vpp.vpp_rl:VewPointTask',
    kwargs={'render': True}
)
