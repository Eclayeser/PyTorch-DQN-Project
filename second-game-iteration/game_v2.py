import math
import sys

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

# --- Config & Constants ---
FPS = 60
DT = 1.0 / FPS                          # fixed timestep, seconds
MAX_DURATION_SECONDS = 30.0             # truncate
GAP_TO_WIN = 6.0                        # gap in score to win automatically

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


class BallPlayer:
    """Class for a single ball - for both player and opponent."""
    def __init__(self, x, y, radius, color):
        self.radius = radius
        self.speed = 1750.0
        self.friction = 0.85
        self.color = color
        self.reset(x, y)

        
    def apply_action(self, action_idx):
        dx, dy = ACTION_VECTORS.get(action_idx, (0.0, 0.0))
        self.vx += dx * self.speed * DT
        self.vy += dy * self.speed * DT


    def step(self):
        """Update position, apply friction, and handle screen boundaries."""
        # vel
        self.x += self.vx * DT
        self.y += self.vy * DT
        # friction
        self.vx *= self.friction
        self.vy *= self.friction

        # bounce off screen edges
        # Reverses direction and removes half the speed upon impact
        BOUNCE_DAMPING = -0.5 

        # Left / Right walls
        if self.x < self.radius:
            self.x = self.radius
            self.vx *= BOUNCE_DAMPING
            
        elif self.x > (SCREEN_WIDTH - self.radius):
            self.x = SCREEN_WIDTH - self.radius
            self.vx *= BOUNCE_DAMPING

        # Top / Bottom walls
        if self.y < self.radius:
            self.y = self.radius
            self.vy *= BOUNCE_DAMPING
            
        elif self.y > (SCREEN_HEIGHT - self.radius):
            self.y = SCREEN_HEIGHT - self.radius
            self.vy *= BOUNCE_DAMPING


    def reset(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.control_score = 0.0


    def get_bot_action(self, zone_x, zone_y, np_random, is_zone_hot, zone_radius):
        """Simple bot logic with variance for training."""    
        # 10% chance to take a random action
        if np_random.random() < 0.10:
            return int(np_random.integers(0, 9))

        # Add jitter so the bot doesn't track perfectly
        delta_x = zone_x - self.x + np_random.uniform(-20.0, 20.0)
        delta_y = zone_y - self.y + np_random.uniform(-20.0, 20.0)
        distance = math.hypot(delta_x, delta_y)

        # maintain a safe distance when hot
        if is_zone_hot:
            distance -= zone_radius * 1.75

        if distance < 5.0:
            return ACTION_NONE
            
        angle_degrees = math.degrees(math.atan2(delta_y, delta_x))
        if -22.5 <= angle_degrees < 22.5: return ACTION_RIGHT
        elif 22.5 <= angle_degrees < 67.5: return ACTION_DOWN_RIGHT
        elif 67.5 <= angle_degrees < 112.5: return ACTION_DOWN
        elif 112.5 <= angle_degrees < 157.5: return ACTION_DOWN_LEFT
        elif angle_degrees >= 157.5 or angle_degrees < -157.5: return ACTION_LEFT
        elif -157.5 <= angle_degrees < -112.5: return ACTION_UP_LEFT
        elif -112.5 <= angle_degrees < -67.5: return ACTION_UP
        elif -67.5 <= angle_degrees < -22.5: return ACTION_UP_RIGHT
        
        return ACTION_NONE


class ZoneCaptureEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, type="bot-testing", render_mode=None):
        super().__init__()

        BLUE = (90, 140, 230)
        YELLOW = (240, 210, 80)
        BALL_RADIUS = 20.0
        
        self.render_mode = render_mode
        self.type = type        # 3 types: agent-bot [training], player-bot [bot-testing], player-agent [agent-testing]
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT

        # Gymnasium-required spaces
        self.action_space = spaces.Discrete(9)
        high = np.array([1.0] * 14, dtype=np.float32)
        low = np.array([-1.0] * 14, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Players
        self.agent = BallPlayer(self.width * 0.2, self.height * 0.25, BALL_RADIUS, BLUE)
        self.opponent = BallPlayer(self.width * 0.2, self.height * 0.75, BALL_RADIUS, YELLOW)
        self.min_ball_touch_dist = BALL_RADIUS * 2

        # Zone
        self.zone_radius = 50.0
        self.hot_duration = 3.0
        self.neutral_duration = 7.0
        self.min_zone_touch_dist = self.zone_radius + BALL_RADIUS

        # Episode info
        self.max_steps = int(MAX_DURATION_SECONDS * FPS)

        # Render resources
        self.screen = None
        self.clock = None
        self.font = None

        self.reset()


    def reset(self, seed=None, options=None):
        """Set initial state and setup gym np_random."""
        super().reset(seed=seed) 

        # Reset objects
        self._update_zone_position(0.0)
        self.agent.reset(self.width * 0.2, self.height * 0.25)
        self.opponent.reset(self.width * 0.2, self.height * 0.75)
        self.ag_in_zone = False
        self.op_in_zone = False

        # Reset episod info
        self.elapsed_time = 0.0
        self.is_hot = False
        self.time_until_transition = self.neutral_duration
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.control_state = 0                                      # 1 (self), -1 (opponent), 0 (contested/neutral)
        
        if self.render_mode == "human":
            self.render()

        return self.get_obs(), self._get_info()


    def _update_zone_position(self, dt):
        """Moves zone in a drifting Lissajous curve figure-8 pattern."""
        if dt == 0:
            self.zone_x, self.zone_y = 0.0, 0.0
            self.zone_vx, self.zone_vy = 0.0, 0.0

        else:
            old_x, old_y = self.zone_x, self.zone_y
            
            t = self.elapsed_time * 0.5
            self.zone_x = self.width / 2.0 + (self.width * 0.3) * math.cos(t)
            self.zone_y = self.height / 2.0 + (self.height * 0.3) * math.sin(t * 2.0)

            self.zone_vx = (self.zone_x - old_x) / dt
            self.zone_vy = (self.zone_y - old_y) / dt


    def get_obs(self):
        """Normalize and return observations."""
        MAX_AGENT_V = 200.0
        MAX_ZONE_V = 220.0

        if self.is_hot:
            hot_val = 1.0
        else:
            hot_val = 1.0 - (self.time_until_transition / self.neutral_duration)

        #note: *2 - 1 is used to shift range from [0, 1] to [-1, 1]
        obs = [
            (self.agent.x / self.width)*2 - 1,                      # agent x pos
            (self.agent.y / self.height)*2 - 1,                     # agent y pos
            np.clip(self.agent.vx / MAX_AGENT_V, -1.0, 1.0),        # agent x vel
            np.clip(self.agent.vy / MAX_AGENT_V, -1.0, 1.0),        # agent y vel
            (self.opponent.x / self.width)*2 - 1,                   # opponent x pos
            (self.opponent.y / self.height)*2 - 1,                  # opponent y pos
            np.clip(self.opponent.vx / MAX_AGENT_V, -1.0, 1.0),     # opponent x vel
            np.clip(self.opponent.vy / MAX_AGENT_V, -1.0, 1.0),     # opponent y vel
            (self.zone_x / self.width)*2 - 1,                       # zone x pos
            (self.zone_y / self.height)*2 - 1,                      # zone y pos
            np.clip(self.zone_vx / MAX_ZONE_V, -1.0, 1.0),          # zone x vel
            np.clip(self.zone_vy / MAX_ZONE_V, -1.0, 1.0),          # zone y vel
            self.control_state,                                     # control state
            hot_val                                                 # index to represent hot state anticipation
        ]

        return np.array(obs, dtype=np.float32)


    def _get_info(self):
        return {
            "elapsed_time": self.elapsed_time,
            "steps": self.steps,
            "agent_control": self.agent.control_score,
            "opponent_control": self.opponent.control_score,
            "zone_hot": self.is_hot
        }


    def calc_reward(self, ag_to_zone_dist):
        """
        Reward function. 
        Intentionally left blank for separate implementation.
        """
        reward = 0.0

        # Agent and Opponent battle
        if not self.ag_in_zone:
            reward -= 1.0 * DT  
        else:
            if self.ag_in_zone and not self.op_in_zone:
                reward += 1.0 * DT
            elif self.op_in_zone and not self.ag_in_zone:
                reward -= 1.0 * DT


        # Penalise for being in the hot zone
        if self.is_hot:
            proximity = (ag_to_zone_dist / self.min_zone_touch_dist)
            if proximity <= 1:
                reward -= 0.1 * (1 - proximity) * DT

        # Winning / Losing 
        if self.terminated or self.truncated:
            score_gap = self.agent.control_score - self.opponent.control_score
            if score_gap >= 10.0:
                reward += 20.0
            elif score_gap <= -10.0:
                reward -= 20.0

        return reward


    def resolve_collision(self, b1, b2):
        """Resolves an elastic collision by preventing overlap and exchanging momentum"""
        delta_x = b2.x - b1.x
        delta_y = b2.y - b1.y
        distance = math.hypot(delta_x, delta_y)

        if 0 < distance < self.min_ball_touch_dist:
            # Pos correction
            overlap = self.min_ball_touch_dist - distance
            half_overlap = overlap / 2.0
            
            normal_x = delta_x / distance
            normal_y = delta_y / distance

            b1.x -= normal_x * half_overlap
            b1.y -= normal_y * half_overlap
            b2.x += normal_x * half_overlap
            b2.y += normal_y * half_overlap

            # Vel resolution
            vel1_normal = (b1.vx * normal_x) + (b1.vy * normal_y)
            vel2_normal = (b2.vx * normal_x) + (b2.vy * normal_y)

            vel_diff1 = vel2_normal - vel1_normal
            vel_diff2 = vel1_normal - vel2_normal

            b1.vx += vel_diff1 * normal_x
            b1.vy += vel_diff1 * normal_y  
            b2.vx += vel_diff2 * normal_x
            b2.vy += vel_diff2 * normal_y


    def step(self, opponent_action_passed = 0, agent_action_passed = 0):
        """Gymnasium step: current state + action -> next state."""

        # Deciding actions based on type
        if self.type == "bot-testing": # agent is bot, opponent is you
            agent_action = self.agent.get_bot_action(self.zone_x, self.zone_y, self.np_random, self.is_hot, self.zone_radius)
            opponent_action = opponent_action_passed

        elif self.type == "agent-testing": # agent is AI agent, opponent is you
            agent_action = agent_action_passed
            opponent_action = opponent_action_passed

        else: # training: agent is AI agent, opponent is bot
            agent_action = agent_action_passed
            opponent_action = self.opponent.get_bot_action(self.zone_x, self.zone_y, self.np_random, self.is_hot, self.zone_radius)

        # Apply actions
        self.agent.apply_action(agent_action)
        self.opponent.apply_action(opponent_action)

        # Physics step
        self.agent.step()
        self.opponent.step()
        self.resolve_collision(self.agent, self.opponent)

        # Zone step
        self.elapsed_time += DT
        self._update_zone_position(DT)

        # Hot/Neutral transition switch
        self.time_until_transition -= DT
        if self.time_until_transition <= 0:
            self.is_hot = not self.is_hot
            self.time_until_transition = self.hot_duration if self.is_hot else self.neutral_duration

        # Control State & Scoring resolution
        ag_to_zone_dist = math.hypot(self.agent.x-self.zone_x, self.agent.y-self.zone_y)
        op_to_zone_dist = math.hypot(self.opponent.x - self.zone_x, self.opponent.y - self.zone_y)
        self.ag_in_zone = ag_to_zone_dist <= self.min_zone_touch_dist
        self.op_in_zone = op_to_zone_dist <= self.min_zone_touch_dist

        score_modifier = 1.0
        if self.ag_in_zone and self.op_in_zone:
            self.control_state = 0  # Contested
            if self.is_hot:
                self.agent.control_score -= score_modifier * DT
                self.opponent.control_score -= score_modifier * DT

        elif self.ag_in_zone:
            self.control_state = 1
            if self.is_hot:
                score_modifier = -1.0
            self.agent.control_score += score_modifier * DT
        elif self.op_in_zone:
            self.control_state = -1
            if self.is_hot:
                score_modifier = -1.0
            self.opponent.control_score += score_modifier * DT
        else:
            self.control_state = 0  # Empty/Neutral

        # Check for final outcome
        self.steps += 1
        
        if abs(self.agent.control_score - self.opponent.control_score) >= GAP_TO_WIN:
            self.terminated = True

        if self.elapsed_time >= MAX_DURATION_SECONDS and not self.terminated:
            self.truncated = True

        # + reward
        reward = self.calc_reward(ag_to_zone_dist)

        if self.render_mode == "human":
            self.render()

        return self.get_obs(), reward, self.terminated, self.truncated, self._get_info()

    def render(self):
        """Render the current state."""
        WHITE = (240, 240, 240)
        BLACK = (15, 15, 20)
        ZONE_CONTESTED = (150, 150, 150)
        ZONE_HOT = (255, 60, 60)
        ZONE_NEUTRAL = (250, 250, 250)

        if self.screen is None:
            self.render_resources()

        # Render field
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(BLACK)
        for x in range(0, self.width, 100):
            pygame.draw.line(canvas, (30, 30, 35), (x, 0), (x, self.height))
        for y in range(0, self.height, 100):
            pygame.draw.line(canvas, (30, 30, 35), (0, y), (self.width, y))

        # Render Zone
        if self.is_hot:
            zone_color = ZONE_HOT
        elif self.control_state == 0:
             if self.ag_in_zone and self.op_in_zone:
                 zone_color = ZONE_CONTESTED
             else:
                zone_color = ZONE_NEUTRAL
        elif self.control_state == 1:
             zone_color = self.agent.color
        else:
             zone_color = self.opponent.color

        pygame.draw.circle(canvas, zone_color, (int(self.zone_x), int(self.zone_y)), int(self.zone_radius), 3 if not self.is_hot else 0)
        if not self.is_hot: # Fill faintly if neutral
             s = pygame.Surface((self.zone_radius*2, self.zone_radius*2), pygame.SRCALPHA)
             pygame.draw.circle(s, (*zone_color, 50), (self.zone_radius, self.zone_radius), self.zone_radius)
             canvas.blit(s, (self.zone_x - self.zone_radius, self.zone_y - self.zone_radius))

        # Render Agents
        pygame.draw.circle(canvas, self.agent.color, (int(self.agent.x), int(self.agent.y)), int(self.agent.radius))
        pygame.draw.circle(canvas, self.opponent.color, (int(self.opponent.x), int(self.opponent.y)), int(self.opponent.radius))

        # Render HUD
        if self.font is not None:
            time_text = self.font.render(f"Time: {self.elapsed_time:.1f}s / {MAX_DURATION_SECONDS}s", True, WHITE)
            state_text = self.font.render(f"Zone: {'HOT (PENALTY)' if self.is_hot else 'NEUTRAL'} ({self.time_until_transition:.1f}s)", True, ZONE_HOT if self.is_hot else WHITE)
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
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 24)
        
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Zone Control")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None


# play against bot
def run_manual():
    """Run loop for manual testing."""
    env = ZoneCaptureEnv(render_mode="human", type="bot-testing")

    running = True
    total_reward = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        action = read_keyboard_action()
        _, reward, terminated, truncated, info = env.step(opponent_action_passed=action)
        total_reward += reward 

        if terminated or truncated:
            reason = "Score Gap >= 10" if terminated else "Time Limit Reached"
            print(f"Episode finished! [{reason}]")
            print(f"P1 Score: {info['agent_control']:.1f} | P2 (Bot) Score: {info['opponent_control']:.1f}")
            print(f"Your total reward: {total_reward:.1f}")
            running = False

    env.close()

def read_keyboard_action():
    """Maps WASD to Action Space."""
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

if __name__ == "__main__":
    run_manual()
    sys.exit(0)