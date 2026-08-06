import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame

from game import ZoneCaptureEnv, read_keyboard_action, GAP_TO_WIN

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


def play_against_agent(model_path="second-game-iteration/policy_net_ep_1200.pth"):
    env = ZoneCaptureEnv(render_mode="human", type="agent-testing")
    
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = "cpu"
    
    model = DQN(n_observations, n_actions).to(device)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Could not find '{model_path}'. Please ensure it is in the same directory.")
        sys.exit(1)
        
    model.eval()
    
    obs, info = env.reset()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                
        # agent picks best action
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            agent_action = model(state_tensor).max(1).indices.item()
            
        # human-player action
        player_action = read_keyboard_action()
        
        action_tuple = (agent_action, player_action)
        obs, _, terminated, truncated, info = env.step(action_tuple)
        
        if terminated or truncated:
            reason = f"Score Gap >= {GAP_TO_WIN}" if terminated else "Time Limit Reached"
            print(f"\nEpisode finished! [{reason}]")
            print(f"P1 (AI) Score: {info['agent_control']:.1f} | P2 (You) Score: {info['opponent_control']:.1f}")
            running = False
            
    env.close()

if __name__ == "__main__":
    play_against_agent()
    sys.exit(0)