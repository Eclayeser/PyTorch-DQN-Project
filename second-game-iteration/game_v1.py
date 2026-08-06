import math
import sys
import random

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

# --- Config & Constants ---
FPS = 60
DT = 1.0 / FPS                          # fixed timestep, seconds
MAX_DURATION_SECONDS = 60.0             # Extended to 60s for a fuller episode

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

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

# Mapping actions to direction vectors (normalized so diagonals aren't faster)
ACTION_VECTORS = {
    ACTION_NONE: (0.0, 0.0),
    ACTION_LEFT: (-1.0, 0.0),
    ACTION_RIGHT: (1.0, 0.0),
    ACTION_UP: (0.0, -1.0),
    ACTION_DOWN: (0.0, 1.0),
    ACTION_UP_LEFT: (-0.7071, -0.7071),
    ACTION_UP_RIGHT: (0.7071, -0.7071),
    ACTION_DOWN_LEFT: (-0.7071, 0.7071),
    ACTION_DOWN_RIGHT: (0.7071, 0.7071),
}

# Colors
WHITE = (240, 240, 240)
BLACK = (15, 15, 20)
GREEN = (80, 200, 120)
BLUE = (90, 140, 230)
RED = (230, 90, 90)
ZONE_NEUTRAL = (200, 200, 200)
ZONE_HOT = (255, 165, 0)
ZONE_CONTESTED = (150, 150, 150)


class BallPlayer:
    """Class for a single ball - for both player and opponent."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.radius = 20.0
        self.speed = 1500.0  # Acceleration
        self.friction = 0.85
        self.color = color
        self.control_score = 0.0

    def apply_action(self, action_idx):
        dx, dy = ACTION_VECTORS.get(action_idx, (0.0, 0.0))
        self.vx += dx * self.speed * DT
        self.vy += dy * self.speed * DT

    def step(self):
        # Apply velocity
        self.x += self.vx * DT
        self.y += self.vy * DT

        # Apply friction
        self.vx *= self.friction
        self.vy *= self.friction

        # Screen boundaries (Passive bounding)
        if self.x < self.radius:
            self.x = self.radius
            self.vx *= -0.5
        elif self.x > SCREEN_WIDTH - self.radius:
            self.x = SCREEN_WIDTH - self.radius
            self.vx *= -0.5

        if self.y < self.radius:
            self.y = self.radius
            self.vy *= -0.5
        elif self.y > SCREEN_HEIGHT - self.radius:
            self.y = SCREEN_HEIGHT - self.radius
            self.vy *= -0.5

    def get_bot_action(self, zone_x, zone_y):
        """Simple bot logic with variance for training."""
        # 10% chance to do a random action (adds variance so agent doesn't overfit)
        if random.random() < 0.10:
            return random.randint(0, 8)

        # Otherwise, track towards the center of the zone
        dx = zone_x - self.x
        dy = zone_y - self.y
        
        # Add a little jitter to the target
        dx += random.uniform(-20, 20)
        dy += random.uniform(-20, 20)
        
        dist = math.hypot(dx, dy)
        if dist < 5.0:
            return ACTION_NONE
            
        # Determine best cardinal/diagonal direction
        dx /= dist
        dy /= dist
        
        # Determine discrete action based on angle
        angle = math.atan2(dy, dx)
        deg = math.degrees(angle)
        
        if -22.5 <= deg < 22.5: return ACTION_RIGHT
        elif 22.5 <= deg < 67.5: return ACTION_DOWN_RIGHT
        elif 67.5 <= deg < 112.5: return ACTION_DOWN
        elif 112.5 <= deg < 157.5: return ACTION_DOWN_LEFT
        elif deg >= 157.5 or deg < -157.5: return ACTION_LEFT
        elif -157.5 <= deg < -112.5: return ACTION_UP_LEFT
        elif -112.5 <= deg < -67.5: return ACTION_UP
        elif -67.5 <= deg < -22.5: return ACTION_UP_RIGHT
        return ACTION_NONE


def resolve_collision(b1: BallPlayer, b2: BallPlayer):
    """Elastic collision with physical blocking/deflection."""
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist = math.hypot(dx, dy)
    min_dist = b1.radius + b2.radius

    if 0 < dist < min_dist:
        # Push apart to prevent overlap
        overlap = min_dist - dist
        nx = dx / dist
        ny = dy / dist

        b1.x -= nx * (overlap / 2)
        b1.y -= ny * (overlap / 2)
        b2.x += nx * (overlap / 2)
        b2.y += ny * (overlap / 2)

        # Exchange velocity along the normal (assuming equal mass)
        v1n = b1.vx * nx + b1.vy * ny
        v2n = b2.vx * nx + b2.vy * ny

        b1.vx += (v2n - v1n) * nx
        b1.vy += (v2n - v1n) * ny
        b2.vx += (v1n - v2n) * nx
        b2.vy += (v1n - v2n) * ny


class ZoneCaptureEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT

        # Gymnasium-required spaces
        self.action_space = spaces.Discrete(9)

        # 15 Features as requested:
        # [self_x, self_y, self_vx, self_vy, 
        #  opp_x, opp_y, opp_vx, opp_vy, 
        #  zone_x, zone_y, zone_vx, zone_vy, 
        #  control_state, hot_state, steps_remaining]
        high = np.array([1.0] * 15, dtype=np.float32)
        low = np.array([-1.0] * 15, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Zone parameters
        self.zone_radius = 50.0
        self.zone_x = 0.0
        self.zone_y = 0.0
        self.zone_vx = 0.0
        self.zone_vy = 0.0
        
        # Zone timings
        self.hot_duration = 3.0
        self.neutral_duration = 7.0
        self.time_until_transition = self.neutral_duration
        self.is_hot = False

        self.control_state = 0  # 1 (self), -1 (opponent), 0 (contested/neutral)

        # Episode info
        self.elapsed_time = 0.0
        self.steps = 0
        self.max_steps = int(MAX_DURATION_SECONDS * FPS)
        self.terminated = False
        self.truncated = False

        # Render resources
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self, seed=None, options=None):
        """Set intial state"""
        super().reset(seed=seed) 
        if seed is not None:
            random.seed(seed)

        # Spawn players
        self.agent = BallPlayer(self.width * 0.2, self.height / 2.0, BLUE)
        self.opponent = BallPlayer(self.width * 0.8, self.height / 2.0, RED)

        # Reset Zone
        self.elapsed_time = 0.0
        self._update_zone_position(0.0)
        self.is_hot = False
        self.time_until_transition = self.neutral_duration
        self.control_state = 0

        self.steps = 0
        self.terminated = False
        self.truncated = False

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def _update_zone_position(self, dt):
        """Moves zone in a drifting Lissajous curve figure-8 pattern."""
        old_x, old_y = self.zone_x, self.zone_y
        
        # Predictable path based on elapsed time
        t = self.elapsed_time * 0.5
        self.zone_x = self.width / 2.0 + (self.width * 0.3) * math.cos(t)
        self.zone_y = self.height / 2.0 + (self.height * 0.3) * math.sin(t * 2.0)

        if dt > 0:
            self.zone_vx = (self.zone_x - old_x) / dt
            self.zone_vy = (self.zone_y - old_y) / dt
        else:
            self.zone_vx, self.zone_vy = 0.0, 0.0

    def _get_obs(self):
        """Assemble and normalize the 15 features."""
        # Normalize positions (0 to 1) -> mapped to (-1 to 1) roughly
        nx = lambda x: (x / self.width) * 2 - 1
        ny = lambda y: (y / self.height) * 2 - 1
        
        # Normalize velocities (assuming max reasonable speed is ~1000)
        nv = lambda v: np.clip(v / 1000.0, -1.0, 1.0)
        
        # Hot state: 1.0 if hot, else countdown (0.0 to 1.0)
        hot_val = 1.0 if self.is_hot else (1.0 - (self.time_until_transition / self.neutral_duration))
        
        # Steps remaining (0.0 to 1.0)
        steps_rem = max(0, self.max_steps - self.steps) / self.max_steps

        obs = np.array([
            nx(self.agent.x), ny(self.agent.y), nv(self.agent.vx), nv(self.agent.vy),
            nx(self.opponent.x), ny(self.opponent.y), nv(self.opponent.vx), nv(self.opponent.vy),
            nx(self.zone_x), ny(self.zone_y), nv(self.zone_vx), nv(self.zone_vy),
            float(self.control_state), 
            float(hot_val), 
            float(steps_rem)
        ], dtype=np.float32)
        return obs

    def _get_info(self):
        return {
            "elapsed_time": self.elapsed_time,
            "steps": self.steps,
            "agent_control": self.agent.control_score,
            "opponent_control": self.opponent.control_score,
            "zone_hot": self.is_hot
        }

    def calc_reward(self):
        """
        Reward function. 
        Intentionally left blank for separate implementation.
        """
        reward = 0.0
        return reward

    def step(self, action):
        """Gymnasium step: current state + action -> next state."""
        # 1. Bot action
        bot_action = self.opponent.get_bot_action(self.zone_x, self.zone_y)

        # 2. Apply actions
        self.agent.apply_action(action)
        self.opponent.apply_action(bot_action)

        # 3. Physics step
        self.agent.step()
        self.opponent.step()
        resolve_collision(self.agent, self.opponent)

        # 4. Zone step
        self.elapsed_time += DT
        self._update_zone_position(DT)

        # 5. Hot/Cold transition logic
        self.time_until_transition -= DT
        if self.time_until_transition <= 0:
            self.is_hot = not self.is_hot
            self.time_until_transition = self.hot_duration if self.is_hot else self.neutral_duration

        # 6. Control State resolution
        ag_in_zone = math.hypot(self.agent.x - self.zone_x, self.agent.y - self.zone_y) <= (self.zone_radius + self.agent.radius)
        op_in_zone = math.hypot(self.opponent.x - self.zone_x, self.opponent.y - self.zone_y) <= (self.zone_radius + self.opponent.radius)

        if ag_in_zone and op_in_zone:
            self.control_state = 0  # Contested
        elif ag_in_zone:
            self.control_state = 1
            self.agent.control_score += (2.0 if self.is_hot else 1.0) * DT
        elif op_in_zone:
            self.control_state = -1
            self.opponent.control_score += (2.0 if self.is_hot else 1.0) * DT
        else:
            self.control_state = 0  # Empty/Neutral

        # 7. Time/step constraints
        self.steps += 1
        if self.elapsed_time >= MAX_DURATION_SECONDS and not self.terminated:
            self.truncated = True
        
        reward = self.calc_reward()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, self.terminated, self.truncated, self._get_info()

    def render(self):
        """Render the current state."""
        self.render_resources()

        # Render field
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(BLACK)

        # Draw Grid (Optional aesthetic)
        for x in range(0, self.width, 100):
            pygame.draw.line(canvas, (30, 30, 35), (x, 0), (x, self.height))
        for y in range(0, self.height, 100):
            pygame.draw.line(canvas, (30, 30, 35), (0, y), (self.width, y))

        # Render Zone
        if self.control_state == 0 and not (math.hypot(self.agent.x-self.zone_x, self.agent.y-self.zone_y) <= self.zone_radius+self.agent.radius and math.hypot(self.opponent.x-self.zone_x, self.opponent.y-self.zone_y) <= self.zone_radius+self.opponent.radius):
             zone_color = ZONE_HOT if self.is_hot else ZONE_NEUTRAL
        elif self.control_state == 0:
             zone_color = ZONE_CONTESTED
        elif self.control_state == 1:
             zone_color = BLUE
        else:
             zone_color = RED

        pygame.draw.circle(canvas, zone_color, (int(self.zone_x), int(self.zone_y)), int(self.zone_radius), 3 if not self.is_hot else 0)
        if not self.is_hot: # Fill faintly if neutral
             s = pygame.Surface((int(self.zone_radius*2), int(self.zone_radius*2)), pygame.SRCALPHA)
             pygame.draw.circle(s, (*zone_color, 50), (self.zone_radius, self.zone_radius), self.zone_radius)
             canvas.blit(s, (self.zone_x - self.zone_radius, self.zone_y - self.zone_radius))

        # Render Agents
        pygame.draw.circle(canvas, self.agent.color, (int(self.agent.x), int(self.agent.y)), int(self.agent.radius))
        pygame.draw.circle(canvas, self.opponent.color, (int(self.opponent.x), int(self.opponent.y)), int(self.opponent.radius))

        # Render HUD
        if self.font is not None:
            time_text = self.font.render(f"Time: {self.elapsed_time:.1f}s / {MAX_DURATION_SECONDS}s", True, WHITE)
            state_text = self.font.render(f"Zone: {'HOT' if self.is_hot else 'NEUTRAL'} ({self.time_until_transition:.1f}s)", True, ZONE_HOT if self.is_hot else WHITE)
            score_text = self.font.render(f"P1: {self.agent.control_score:.1f} | P2: {self.opponent.control_score:.1f}", True, WHITE)
            
            canvas.blit(time_text, (10, 10))
            canvas.blit(state_text, (10, 40))
            canvas.blit(score_text, (10, 70))

        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(canvas)
            return np.transpose(frame, axes=(1, 0, 2))

    def render_resources(self):
        """Helper for render(). Setup screen and fonts."""
        if self.font is not None:
            return
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 24)
        
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Zone Control AI Arena")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None


# --- Manual Play Handlers ---

def read_keyboard_action():
    """Maps WASD / Arrow keys to Action Space."""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    if up and left: return ACTION_UP_LEFT
    if up and right: return ACTION_UP_RIGHT
    if down and left: return ACTION_DOWN_LEFT
    if down and right: return ACTION_DOWN_RIGHT
    if up: return ACTION_UP
    if down: return ACTION_DOWN
    if left: return ACTION_LEFT
    if right: return ACTION_RIGHT
    return ACTION_NONE


def run_manual():
    """Run loop for manual testing."""
    env = ZoneCaptureEnv(render_mode="human")
    env.reset()

    running = True
    total_reward = 0.0

    print("--- Zone Control Manual Test ---")
    print("Use Arrow Keys or WASD to move.")
    print("Press ESC to exit.")

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
            print(f"Episode finished! P1 Score: {info['agent_control']:.1f} | P2 (Bot) Score: {info['opponent_control']:.1f}")
            running = False

    env.close()

if __name__ == "__main__":
    run_manual()
    sys.exit(0)