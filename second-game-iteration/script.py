# -*- coding: utf-8 -*-

import math
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordVideo
import torch.nn.functional as F

from game import ZoneCaptureEnv

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 128
GAMMA = 0.995
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 450000
TAU = 0.005
LR = 3e-4

training_env = ZoneCaptureEnv(type="training")
#render_env = ZoneCaptureEnv(render_mode="human", type="training")
render_env = RecordVideo(
    ZoneCaptureEnv(render_mode="rgb_array", type="training"),
    video_folder="second-game-iteration/videos-double-DQN",
    name_prefix="zone_controller",
    episode_trigger=lambda episode_id: True,
    disable_logger=True,
)

n_actions = training_env.action_space.n
n_observations = training_env.observation_space.shape[0]

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(250000)

steps_done = 0
episode_agent_scores = []
num_episodes = 1201


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        # greedy option
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # random exploration
        return torch.tensor([[training_env.action_space.sample()]], device=device, dtype=torch.long)


def plot_durations(show_result=False):
    plt.figure(1)
    scores_s = torch.tensor(episode_agent_scores, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Agent Score')
    plt.plot(scores_s.numpy())
    # Take 100 episode averages and plot them too
    if len(scores_s) >= 100:
        means = scores_s.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)


    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_actions = policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
   
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def watch_episode(policy_net, device):
    global render_env
    initial_state, info = render_env.reset()
    state = torch.tensor(initial_state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        with torch.no_grad():
            action = policy_net(state).max(1).indices.view(1, 1)  # greedy, no exploration
        observation, _, terminated, truncated, _ = render_env.step(action.item())
        if terminated or truncated:
            print(f"  -> watched episode lasted {t + 1} steps")
            break
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


for i_episode in range(num_episodes):
    state, info = training_env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = training_env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated or truncated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        if steps_done % 4 == 0:
            optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_agent_scores.append(info["agent_control"])
            plot_durations()

            if i_episode % 200 == 0:      # every 00 episodes, record greedy run
                print(f"[episode {i_episode}] recording greedy run...")
                watch_episode(policy_net, device) 

                # Save the model's state_dict
                model_filename = f"policy_net_ep_{i_episode}.pth"
                torch.save(policy_net.state_dict(), model_filename)
                print(f"  -> Saved policy_net to {model_filename}")           

            break


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
render_env.close()
