import torch
import gymnasium as gym

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
        a, _ = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
    obs, r, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
    total += r
    done = term or trunc

print(f'episode return: {total:.1f}')
env.close()
