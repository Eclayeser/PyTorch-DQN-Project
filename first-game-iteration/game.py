"""
Bounce Platform (Gymnasium Environment) - First Game Iteration

Control a platform at the bottom of the screen (left/right) to keep a
constantly-moving ball from hitting the floor. The ball bounces off the
side walls, the top wall, and the platform (simple mirror bounce: only the
ball's y-velocity flips, x-velocity is unaffected).

Episode outcomes:
- terminated = True  -> the ball touched the bottom of the screen (fail)
- truncated  = True  -> the ball survives MAX_DURATION_SECONDS (success)

Runnable directly
"""

import math
import sys

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

# Confs env
FPS = 60
DT = 1.0 / FPS                          # fixed timestep, seconds
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 440
PLATFORM_WIDTH = 90
PLATFORM_HEIGHT = 14
PLATFORM_Y = SCREEN_HEIGHT - 40
PLATFORM_ACCEL = 1800.0
PLATFORM_FRICTION = 1500.0
PLATFORM_MAX_SPEED = 380.0              # capped
BALL_RADIUS = 8
BALL_SPEED = 700.0                      # constant speed magnitude, px/s

MAX_DURATION_SECONDS = 10.0             # truncate (success) after this long


# Action space:
ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2

class BouncePlatformEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT

        # Gymnasium-required spaces
        self.action_space = spaces.Discrete(3)

        high = np.array(
            [
                self.width,            # platform_x
                PLATFORM_MAX_SPEED,    # platform_vx
                self.width,            # ball_x
                self.height,           # ball_y
                BALL_SPEED,            # ball_vx
                BALL_SPEED,            # ball_vy
            ],
            dtype=np.float32,
        )
        low = np.array([0.0, -PLATFORM_MAX_SPEED, 0.0, 0.0, -BALL_SPEED, -BALL_SPEED], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # internal state
        self.platform_x = 0.0
        self.platform_vx = 0.0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0

        # episode info
        self.elapsed_time = 0.0
        self.steps = 0
        self.terminated = False
        self.truncated = False

        # render resources
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self, seed=None):
        """Set intial state"""
        super().reset(seed=seed) 

        self.platform_x = self.width / 2.0
        self.platform_vx = 0.0

        self.ball_x = self.width / 2.0
        self.ball_y = self.height / 3.0

        # Start moving diagonally upwards at a constant speed; randomize
        # which side it heads toward first using the seeded Gymnasium RNG.
        direction = self.np_random.choice([-1, 1])
        component = BALL_SPEED / math.sqrt(2)
        self.ball_vx = float(direction * component)
        self.ball_vy = -component  # negative y = upward on screen

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
        return np.array(
            [
                self.platform_x,
                self.platform_vx,
                self.ball_x,
                self.ball_y,
                self.ball_vx,
                self.ball_vy,
            ], dtype=np.float32,
        )

    def _get_info(self):
        return {"elapsed_time": self.elapsed_time, "steps": self.steps}

    def step(self, action):
        """
        Gymnasium step: current state + action -> next state.
        """

        self.apply_platform_physics(action)
        self.apply_ball_physics()

        self.elapsed_time += DT
        self.steps += 1

        if self.elapsed_time >= MAX_DURATION_SECONDS and not self.terminated:
            self.truncated = True
        
        # reward calculation
        if self.terminated:
            reward = -10.0
        else:
            reward = 0.05

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, self.terminated, self.truncated, self._get_info()


    def apply_platform_physics(self, action):
        ACTION_TO_DIRECTION = {
            ACTION_NONE: 0,
            ACTION_LEFT: -1,
            ACTION_RIGHT: 1,
        }

        direction = ACTION_TO_DIRECTION[action]

        if direction != 0:
            self.platform_vx += direction * PLATFORM_ACCEL * DT
        else:
            # Friction: decelerate toward zero without overshooting past it.
            if self.platform_vx > 0.0:
                self.platform_vx = max(0.0, self.platform_vx - PLATFORM_FRICTION * DT)
            elif self.platform_vx < 0.0:
                self.platform_vx = min(0.0, self.platform_vx + PLATFORM_FRICTION * DT)

        # Cap speed
        self.platform_vx = max(-PLATFORM_MAX_SPEED, min(PLATFORM_MAX_SPEED, self.platform_vx))

        self.platform_x += self.platform_vx * DT

        half_w = PLATFORM_WIDTH / 2.0
        if self.platform_x < half_w:
            self.platform_x = half_w
            self.platform_vx = 0.0
        elif self.platform_x > self.width - half_w:
            self.platform_x = self.width - half_w
            self.platform_vx = 0.0


    def apply_ball_physics(self):
        self.ball_x += self.ball_vx * DT
        self.ball_y += self.ball_vy * DT

        # side walls
        if self.ball_x - BALL_RADIUS <= 0:
            self.ball_x = BALL_RADIUS
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + BALL_RADIUS >= self.width:
            self.ball_x = self.width - BALL_RADIUS
            self.ball_vx = -abs(self.ball_vx)

        # top wall
        if self.ball_y - BALL_RADIUS <= 0:
            self.ball_y = BALL_RADIUS
            self.ball_vy = abs(self.ball_vy)

        # platform (simple mirror bounce: only y-velocity flips)
        half_w = PLATFORM_WIDTH / 2.0
        if (
            self.ball_vy > 0
            and self.ball_y + BALL_RADIUS >= PLATFORM_Y
            and self.ball_y - BALL_RADIUS <= PLATFORM_Y + PLATFORM_HEIGHT
            and (self.platform_x - half_w - BALL_RADIUS)
            <= self.ball_x
            <= (self.platform_x + half_w + BALL_RADIUS)
        ):
            self.ball_y = PLATFORM_Y - BALL_RADIUS
            self.ball_vy = -abs(self.ball_vy)

        # Bottom of the screen -> termination (missed the ball)
        if self.ball_y + BALL_RADIUS >= self.height:
            self.terminated = True


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

        canvas = self.screen
        canvas.fill(BLACK)

        # platform
        half_w = PLATFORM_WIDTH / 2.0
        platform_rect = pygame.Rect(
            self.platform_x - half_w, PLATFORM_Y, PLATFORM_WIDTH, PLATFORM_HEIGHT
        )
        pygame.draw.rect(canvas, BLUE, platform_rect)

        # ball
        pygame.draw.circle(
            canvas, GREEN, (int(self.ball_x), int(self.ball_y)), BALL_RADIUS
        )

        # HUD
        hud_text = (
            f"time: {self.elapsed_time:4.1f}s / {MAX_DURATION_SECONDS:.0f}s   "
        )
        hud = self.font.render(hud_text, True, WHITE)
        canvas.blit(hud, (10, 10))

        
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        return None


    def render_resources(self):
            """
            Helper for render()
            """
            if self.screen is not None:
                return
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption("Bounce Platform")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 20)


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None



# to play myself
def read_keyboard_action():
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a]
    right = keys[pygame.K_d]

    if left and not right:
        return ACTION_LEFT
    if right and not left:
        return ACTION_RIGHT
    return ACTION_NONE


# to play myself
def run_manual():
    env = BouncePlatformEnv(render_mode="human")
    env.reset()

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
        total_reward += reward

        if terminated or truncated:
            outcome = "SUCCESS - survived the full 10 seconds" if truncated else "FAIL - the ball hit the floor"
            print(f"Game over: {outcome}")
            print(f"Duration: {info['steps']} steps ({info['elapsed_time']:.2f} seconds)")
            print(f"Total reward: {total_reward:.2f}")
            break

    env.close()


# runnable - to play myself
if __name__ == "__main__":
    run_manual()
    sys.exit(0)