import gymnasium as gym
from config import env_id, num_envs

def make_env(env_id, capture_video):
  env = gym.make(env_id, render_mode = 'rgb_array')
  if capture_video:
    env = gym.wrappers.RecordVideo(env, 'video')
  env = gym.wrappers.RecordEpisodeStatistics(env)
  return env

envs = gym.vector.SyncVectorEnv(
[lambda: make_env(env_id, False) for i in range(num_envs)],
autoreset_mode = gym.vector.AutoresetMode.SAME_STEP
)
