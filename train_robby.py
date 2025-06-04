# Robby the Robot Q-learning Simulator
# --------------------------------------
# This program simulates a reinforcement learning agent (Robby)
# using Q-learning to pick up soda cans in a 10x10 grid world.
# Each episode Robby is placed randomly and must maximize his
# reward by picking up cans while avoiding walls and mistakes.

import numpy as np
import random
import matplotlib.pyplot as plt
import sys

# Constants
GRID_SIZE = 10                 # 10x10 interior grid
EPISODES = 5000              # Number of training episodes
STEPS = 200        # Steps per episode
ALPHA = 0.2                    # Learning rate
GAMMA = 0.9                    # Discount factor
EPSILON_START = 0.1            # Initial exploration rate
EPSILON_DECAY = 0.001          # Epsilon decay amount
EPSILON_DECAY_INTERVAL = 50    # Decay frequency in episodes

# Rewards
REWARD_CAN = +10
REWARD_WALL = -5
REWARD_EMPTY = -1
REWARD_MOVE = 0

# Grid encoding
CAN = 0
EMPTY = 1
WALL = 2

# Q-table dimensions
NUM_STATES = 3 ** 5

# [0=N, 1=S, 2=E, 3=W, 4=PickUp]
NUM_ACTIONS = 5  

class RobbyEnv:
    def __init__(self, grid_size=GRID_SIZE, can_probability=0.5):
        self.grid_size = grid_size
        self.can_prob = can_probability
        self._build_grid()
        self.reset()

    # Full grid with walls
    def _build_grid(self):
        self.grid = np.full((self.grid_size + 2, self.grid_size + 2), WALL, dtype=int)
    
    # Resets the environment 
    def reset(self):
        self._build_grid()
        for row in range(1, self.grid_size + 1):
            for col in range(1, self.grid_size + 1):
                self.grid[row, col] = CAN if random.random() < self.can_prob else EMPTY
        while True:
            x = random.randint(1, self.grid_size)
            y = random.randint(1, self.grid_size)
            if self.grid[y, x] in (CAN, EMPTY):
                self.x = x
                self.y = y
                break
        return self.state_index()

    # Sensor methods 
    def Current(self):
        return self.grid[self.y, self.x]

    def North(self):
        if self.y + 1 >= self.grid.shape[0]:
            return WALL
        return self.grid[self.y + 1, self.x]

    def South(self):
        if self.y - 1 < 0:
            return WALL
        return self.grid[self.y - 1, self.x]

    def East(self):
        if self.x + 1 >= self.grid.shape[1]:
            return WALL
        return self.grid[self.y, self.x + 1]

    def West(self):
        if self.x - 1 < 0:
            return WALL
        return self.grid[self.y, self.x - 1]

    # Returns sensor readings
    def _get_sensors(self):
        return [self.Current(), self.North(), self.South(), self.East(), self.West()]

    def state_index(self):
        # Encodes sensor values as a base-3 integer (0-242)
        sensors = self._get_sensors()
        idx = sum(val * (3 ** i) for i, val in enumerate(sensors))
        return idx

    # Starts action
    def step(self, action):
        picked_up = False
        if action == 0:  # North
            new_x, new_y = self.x, self.y + 1
        elif action == 1:  # South
            new_x, new_y = self.x, self.y - 1
        elif action == 2:  # East
            new_x, new_y = self.x + 1, self.y
        elif action == 3:  # West
            new_x, new_y = self.x - 1, self.y
        elif action == 4:  # PickUp
            if self.grid[self.y, self.x] == CAN:
                self.grid[self.y, self.x] = EMPTY
                return self.state_index(), REWARD_CAN, True
            else:
                return self.state_index(), REWARD_EMPTY, False
        else:
            raise ValueError("Invalid action")

        if self.grid[new_y, new_x] == WALL:
            return self.state_index(), REWARD_WALL, False
        else:
            self.x, self.y = new_x, new_y
            return self.state_index(), REWARD_MOVE, False

    # Grid showing Robby's position
    def render(self):
        print("Grid Snapshot (R=Robby, C=Can, .=Empty, #=Wall):")
        for row in range(self.grid.shape[0]):
            line = []
            for col in range(self.grid.shape[1]):
                if (col, row) == (self.x, self.y):
                    line.append('R')
                else:
                    val = self.grid[row, col]
                    line.append({WALL: '#', EMPTY: '.', CAN: 'C'}[val])
            print(" ".join(line))
        print()

def train_and_test():
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))
    env = RobbyEnv()
    epsilon = EPSILON_START
    plotted_eps, plotted_rewards = [], []

    # Training loop
    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0

        if ep % EPSILON_DECAY_INTERVAL == 0:
            epsilon = max(0.0, epsilon - EPSILON_DECAY)

        for _ in range(STEPS):
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = np.argmax(Q[state])
            next_state, reward, _ = env.step(action)
            Q[state, action] += ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            total_reward += reward

        if ep % 100 == 0:
            plotted_eps.append(ep)
            plotted_rewards.append(total_reward)
            print(f"[Train] Episode {ep:4d} | Reward: {total_reward:4d} | Epsilon: {epsilon:.3f}")
            env.render() 


    # Testing phase
    test_rewards = []
    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        for _ in range(STEPS):
            action = np.argmax(Q[state])
            state, reward, _ = env.step(action)
            total_reward += reward
        test_rewards.append(total_reward)

        if ep % 1000 == 0:
            print(f"[Test] Episode {ep:4d} | Reward: {total_reward:4d}")
            env.render()
            
    print("\n[Test] Average Reward:", np.mean(test_rewards), flush=True)
    print("[Test] Std Dev:", np.std(test_rewards), flush=True)

    # Plot training reward
    plt.plot(plotted_eps, plotted_rewards, marker='o')
    plt.title("Training Reward vs. Episode")
    plt.xlabel("Episode (one point/100 eps)")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_and_test()
