import torch
import gymnasium as gym
import time

from config import env_id, model_path
from model import Actor
from make_env import envs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actor = Actor(envs).to(device)
actor.load_state_dict(torch.load(model_path, map_location=device))
actor.eval()

env = gym.make(env_id, render_mode='human')
obs, _ = env.reset()
total = 0.0
done = False
while not done:
    with torch.no_grad():
        action, _ = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
    obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
    total += reward
    done = terminated or truncated
    time.sleep(0.05)

print(f'episode return: {total:.1f}')
env.close()
