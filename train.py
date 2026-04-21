import torch
from torch import nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer
import time
import os

from config import resume, checkpoint_dir, checkpoint_interval, model_path, total_timesteps, learning_rate, buffer_size, gamma, tau, policy_update_period, batch_size, num_step_before_training
from model import Critic, Actor
from make_env import envs

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

def np2torch(a):
  return torch.as_tensor(a, dtype=torch.float32, device=device)

target_entropy = -float(np.prod(envs.single_action_space.shape))
log_alpha = torch.zeros(1, device=device, requires_grad=True)
alpha_optimizer = torch.optim.Adam([log_alpha], lr=learning_rate)

actor = Actor(envs).to(device)

q_network1 = Critic(envs).to(device)
q_network2 = Critic(envs).to(device)
q_target1 = Critic(envs).to(device)
q_target2 = Critic(envs).to(device)

q_target1.load_state_dict(q_network1.state_dict())
q_target2.load_state_dict(q_network2.state_dict())

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)

q1_optimizer = torch.optim.Adam(q_network1.parameters(), lr=learning_rate)
q2_optimizer = torch.optim.Adam(q_network2.parameters(), lr=learning_rate)

rb = ReplayBuffer(
    buffer_size = buffer_size,
    observation_space = envs.single_observation_space,
    action_space = envs.single_action_space,
    device = device
)

start_time = time.time()
episode_returns = []
obs, _ = envs.reset()
start_step, best_return = 0, -float('inf')
if resume and os.path.exists(checkpoint_dir):
    c = torch.load(checkpoint_dir, map_location=device)
    actor.load_state_dict(c['actor']); q_network1.load_state_dict(c['q1']); q_network2.load_state_dict(c['q2'])
    q_target1.load_state_dict(c['q1_t']); q_target2.load_state_dict(c['q2_t'])
    log_alpha.data.copy_(c['log_alpha'])
    start_step, best_return = c['step'] + 1, c['best']
    print(f'resumed @ {start_step}, best={best_return:.1f}')

for global_step in range(total_timesteps):
    if global_step < num_step_before_training:
        actions = np.array([envs.single_action_space.sample() for i in range(envs.num_envs)])
    else:
        with torch.no_grad():
            action, _ = actor.sample(np2torch(obs))
            actions = action.cpu().numpy()

    next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

    if 'final_info' in infos:
        ret = infos['final_info']['episode']['r'][0]
        print(f'global_step = {global_step}, episode return: {ret}')
        episode_returns.append(ret)

    real_next_obs = next_obs.copy()
    for i, done in enumerate(truncateds):
        if done:
            real_next_obs[i] = infos['final_obs'][i]
    rb.add(obs, real_next_obs, actions, rewards, terminateds, [infos])

    obs = next_obs
    
    if global_step > num_step_before_training:
        data = rb.sample(batch_size)
        alpha = log_alpha.exp().detach()

        with torch.no_grad():
            action_next, log_prob_next = actor.sample(np2torch(data.next_observations))
            q_next_values1 = q_target1(np2torch(data.next_observations), action_next)
            q_next_values2 = q_target2(np2torch(data.next_observations), action_next)
            td_target = data.rewards.flatten() + gamma * (torch.min(q_next_values1, q_next_values2).view(-1) - alpha * log_prob_next.view(-1)) * (1 - data.dones.flatten())
        
        q1_pred = q_network1(np2torch(data.observations), np2torch(data.actions))
        loss_q1 = nn.functional.mse_loss(td_target.squeeze(), q1_pred.squeeze())

        q1_optimizer.zero_grad()
        loss_q1.backward()
        q1_optimizer.step()

        q2_pred = q_network2(np2torch(data.observations), np2torch(data.actions))
        loss_q2 = nn.functional.mse_loss(td_target.squeeze(), q2_pred.squeeze())

        q2_optimizer.zero_grad()
        loss_q2.backward()
        q2_optimizer.step()

        if global_step % policy_update_period == 0:
            action_new, log_prob_new = actor.sample(np2torch(data.observations))
            q1_ = q_network1(np2torch(data.observations), action_new)
            q2_ = q_network2(np2torch(data.observations), action_new)
            actor_loss = -(torch.min(q1_, q2_) - alpha * log_prob_new).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            alpha_loss = -(log_alpha * (log_prob_new.detach() + target_entropy)).mean()
            alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_optimizer.step()

            for param, target_param in zip(q_network1.parameters(), q_target1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(q_network2.parameters(), q_target2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    if global_step % 100 == 0:
        print(f'steps per second: {int(global_step/(time.time() - start_time))}')
    
    if global_step % checkpoint_interval == 0:
        torch.save({'actor': actor.state_dict(), 'q1': q_network1.state_dict(), 'q2': q_network2.state_dict(),
                    'q1_t': q_target1.state_dict(), 'q2_t': q_target2.state_dict(),
                    'log_alpha': log_alpha.detach(), 'step': global_step, 'best': best_return}, checkpoint_dir)

envs.close()
torch.save(actor.state_dict(), model_path)

print('total time: ', (time.time() - start_time)/60)


