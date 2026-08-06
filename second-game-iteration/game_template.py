import math
import sys

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

# Confs env
FPS = 60
DT = 1.0 / FPS                          # fixed timestep, seconds

MAX_DURATION_SECONDS = 10.0             # truncate

# Action space (agent):
ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_UP_LEFT = 4
ACTION_UP_RIGHT = 5
ACTION_DOWN = 6
ACTION_DOWN_LEFT = 7
ACTION_DOWN_RIGHT = 8
# Note: diagonal movement must not be faster than cardinal movement

class BallPlayer:
    # class for a single ball - for both player and opponent
    def __init__(self):
        # set initial properties
        pass

    # in this class you can also place a simple, somewhat random
    # bot logic - for training
    # it is important it has some varience to it so the agent
    # does not just learn how bot moves entirely

class ZoneCaptureEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()

        #self.render_mode = render_mode
        #self.width = SCREEN_WIDTH
        #self.height = SCREEN_HEIGHT

        # Gymnasium-required spaces
        self.action_space = spaces.Discrete(9)

        high = np.array(
            [
            # observations that define state - max values
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                # observations that define state - max values
            ], 
            dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # internal state
        # create 2 balls:
        # self.agent = new BallPlayer()
        # self.opponent = new BallPlayer()

        # episode info
        self.elapsed_time = 0.0
        self.steps = 0
        self.terminated = False
        self.truncated = False

        # render resources
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self, seed=None, options=None):
        """Set intial state"""
        super().reset(seed=seed) 

        self.platform_x = self.width / 2.0
        self.platform_vx = 0.0

        # Spawn balls (fixed positions)

        self.elapsed_time = 0.0
        self.steps = 0
        self.terminated = False
        self.truncated = False

        observations = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observations, info

    def _get_obs(self):
        obs = np.array(
            [
                self.agent.x,
                # ...
            ], dtype=np.float32,
        )
        # normalise all except control_state (takes only -1, 0 or 1) and hot_state (binary)
        return obs

    def _get_info(self):
        return {"elapsed_time": self.elapsed_time, "steps": self.steps, } # also:
        # return info such as how much accummulated control for each agent/opponent

    def calc_reward(self):
        """
        Reward function
        """
        reward = 0.0
        # DO NOT IMPLEMENT IT FOR NOW - THIS WILL BE DONE SEPERATELY ONCE PROVEN GAME WORKS AS INTENDED
        return reward

    def step(self, action):
        """
        Gymnasium step: current state + action -> next state.
        """
        # calculate next state 
        
        self.elapsed_time += DT
        self.steps += 1

        if self.elapsed_time >= MAX_DURATION_SECONDS and not self.terminated:
            self.truncated = True
        
        reward = self.calc_reward()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, self.terminated, self.truncated, self._get_info()


    def render(self):
        """
        Render the current state.
        Only supposed to be run on manual play or learning progress check run
        """
        WHITE = (240, 240, 240)
        BLACK = (15, 15, 20)
        GREEN = (80, 200, 120)
        BLUE = (90, 140, 230)

        self.render_resources()

        # render field
        # render zone
        # render agent
        # render opponent
        # render HUD: should include elapsed time, and control statistics for each player

        # "human" will be used for manual play or live progress showcase
        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        # "rgb_array" for video recording at the late stages
        elif self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(canvas)
            return np.transpose(frame, axes=(1, 0, 2))


    def render_resources(self):
        """
        Helper for render(). Only opens a real display window in "human"
        mode, so "rgb_array" mode works headlessly.
        """
        if self.font is not None:
            return
        pygame.font.init()
        # stuff like font
        if self.render_mode == "human":
            pygame.init()
            # pygame.display.set_caption()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None



# to play myself
def read_keyboard_action():
    pass


# to play myself (as the agent) against the bot - for game testing
def run_manual():
    env = ZoneCaptureEnv(render_mode="human")
    env.reset() # with a bot

    running = True
    total_reward = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        action = read_keyboard_action()
        observations, reward, terminated, truncated, info = env.step(action)
        total_reward += reward # currently will be stale until reward calculation introduced

        if terminated or truncated:
            # stop and print in terminal results
            pass

    env.close()


# runnable - to play myself
if __name__ == "__main__":
    run_manual()
    sys.exit(0)