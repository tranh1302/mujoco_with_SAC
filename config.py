env_id = 'HalfCheetah-v5'

num_envs = 1

total_timesteps = 1_000_000

learning_rate = 3e-4

buffer_size = 1_000_000

gamma = 0.99

tau = 0.005

policy_update_period = 1

batch_size = 256

num_step_before_training = 25000

model_path = 'SAC.pth'

checkpoint_dir = 'checkpoints'

checkpoint_interval = 50_000

resume = True

gpu = False