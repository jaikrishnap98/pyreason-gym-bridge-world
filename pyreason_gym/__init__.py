from gym.envs.registration import register

register(
    id='PyReasonBridgeWorld-v0',
    entry_point='pyreason_gym.envs:BridgeWorldEnv'
)
